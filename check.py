"""云端最小正确性对照；没有性能计时，也不加载模型。"""

import torch

from router import fused_topk, reference


def check_case(name: str, logits: torch.Tensor, k: int) -> None:
    weights, ids = fused_topk(logits, k)
    expected_weights, expected_ids = reference(logits, k)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.shape == ids.shape == (logits.shape[0], k)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.double(), expected_weights, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[:, 0]),
                               rtol=1e-5, atol=1e-6)
    ordered = ids.sort(dim=-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()
    error = (weights.double() - expected_weights).abs().max().item()
    print(f"[PASS] {name}：T={logits.shape[0]}，E={logits.shape[1]}，K={k}，权重最大误差={error:.8g}")


def main() -> None:
    torch.manual_seed(0)
    print(f"[环境] GPU={torch.cuda.get_device_name(0)}，PyTorch={torch.__version__}，CUDA运行时={torch.version.cuda}")
    # 六个形状覆盖各 E/K，并用 17 行覆盖最后一个线程块只有一行的情况。
    for rows, experts, k in [(1, 64, 2), (17, 64, 8), (128, 128, 2),
                             (17, 128, 8), (1024, 256, 2), (128, 256, 8)]:
        check_case("随机，种子0", torch.randn(rows, experts, device="cuda"), k)

    check_case("全相等，小编号优先", torch.zeros(17, 256, device="cuda"), 8)
    boundary = torch.full((1, 128), -10.0, device="cuda")
    boundary[0, -7:] = torch.arange(10.0, 17.0, device="cuda")
    boundary[0, [0, 31, 32, 63]] = 5.0
    check_case("第8名跨线程并列", boundary, 8)

    # FP32 Softmax 可能把这些不同 logits 舍入成相同概率，选取仍按原 logits。
    close = torch.linspace(-1e-8, 1e-8, 64, device="cuda").reshape(1, 64)
    check_case("接近但不相等", close, 2)
    large = torch.linspace(-10000.0, 10000.0, 256, device="cuda").repeat(17, 1)
    check_case("大幅有限正负值", large, 8)
    extreme = torch.full((1, 64), -torch.finfo(torch.float32).max, device="cuda")
    extreme[0, 63] = torch.finfo(torch.float32).max
    check_case("极端有限差值，指数下溢为零", extreme, 8)
    print("[完成] 11个正确性用例通过；没有验证成熟融合基线或性能")


if __name__ == "__main__":
    main()
