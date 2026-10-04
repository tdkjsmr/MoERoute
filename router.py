"""标准 MoE 路由：按 logits 选专家，返回重归一化权重与专家编号。"""

import torch


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
