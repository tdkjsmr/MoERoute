"""标准路由与固定 DeepSeek-V3 分组路由的参考和融合入口。"""

import torch

GROUPED_SCALE = 2.5


def reference(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """独立 FP64 参考：全专家 Softmax，再选取并重归一化。"""
    # 稳定排序保留原编号顺序，明确相同 logits 时较小编号优先。
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
    probabilities = torch.softmax(logits.double(), dim=-1)
    weights = probabilities.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids.to(torch.int32)


def fused_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """连续 CUDA FP32 [T,E] → FP32 权重、int32 编号 [T,K]。"""
    # 延迟导入使数学参考可以独立使用；融合算子依赖提前编译的扩展。
    import _moe_router

    return _moe_router.fused_topk(logits, k)


def fused_topk_out(logits: torch.Tensor, k: int, weights: torch.Tensor,
                   ids: torch.Tensor) -> None:
    """标准路由写入已有输出；输出与输入、彼此之间均不得重叠。"""
    import _moe_router

    _moe_router.fused_topk_out(logits, k, weights, ids)


def grouped_pytorch(logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 分步对手：256 专家、8 组留 4 组、选 8 项，普通样本按分数降序。"""
    probabilities = logits.sigmoid()
    corrected = (probabilities + bias).reshape(-1, 8, 32)
    group_scores = corrected.topk(2, dim=-1).values.sum(dim=-1)
    groups = group_scores.topk(4, dim=-1).indices
    retained = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    eligible = corrected.masked_fill(~retained.unsqueeze(-1), -torch.inf).reshape(-1, 256)
    ids = eligible.topk(8, dim=-1, sorted=True).indices
    selected = probabilities.gather(1, ids)
    return selected / selected.sum(dim=-1, keepdim=True) * GROUPED_SCALE, ids.to(torch.int32)


def grouped_reference(logits: torch.Tensor, bias: torch.Tensor, *, fp64: bool = True
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """稳定排序对照；升精度发生在 sigmoid 和偏置加法之前，同分取小组号/专家号。"""
    values, correction = (logits.double(), bias.double()) if fp64 else (logits, bias)
    probabilities = values.sigmoid()
    corrected = (probabilities + correction).reshape(-1, 8, 32)
    ranked = corrected.sort(dim=-1, descending=True, stable=True).values
    group_scores = ranked[..., :2].sum(dim=-1)
    groups = group_scores.argsort(dim=-1, descending=True, stable=True)[:, :4]
    retained = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    eligible = corrected.masked_fill(~retained.unsqueeze(-1), -torch.inf).reshape(-1, 256)
    ids = eligible.argsort(dim=-1, descending=True, stable=True)[:, :8]
    selected = probabilities.gather(1, ids)
    # 官方定义不加 epsilon；选中原始分数总量为零时产生 NaN，不提供有效权重。
    return selected / selected.sum(dim=-1, keepdim=True) * GROUPED_SCALE, ids.to(torch.int32)


def fused_grouped_topk(logits: torch.Tensor, bias: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """连续 FP32 [T,256] 和 [256] 偏置，返回 FP32 权重与 int32 编号 [T,8]。"""
    import _moe_router

    return _moe_router.fused_grouped_topk(logits, bias)
