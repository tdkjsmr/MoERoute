"""路由正确性对照；没有性能计时，也不加载模型。"""

import argparse

import torch

from router import (GROUPED_SCALE, fused_grouped_topk, fused_topk, fused_topk_out,
                    grouped_reference, reference)


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
    output_weights = torch.full_like(weights, torch.nan)
    output_ids = torch.full_like(ids, -1)
    addresses = (output_weights.data_ptr(), output_ids.data_ptr())
    fused_topk_out(logits, k, output_weights, output_ids)
    torch.testing.assert_close(output_weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(output_ids, ids, rtol=0, atol=0)
    assert addresses == (output_weights.data_ptr(), output_ids.data_ptr())
    print(f"[PASS] {name}：T={logits.shape[0]}，E={logits.shape[1]}，K={k}，权重最大误差={error:.8g}")


def check_standard() -> None:
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
    # 连续视图可以共享底层存储，但实际字节区间不得重叠。
    logits = torch.randn(1, 256, device="cuda")
    original = logits.clone()
    weights = torch.empty(1, 8, device="cuda")
    ids = torch.empty(1, 8, device="cuda", dtype=torch.int32)
    invalid = [(weights[:, :7], ids), (weights.double(), ids),
               (logits.flatten()[:8].view(1, 8), ids),
               (weights, logits.view(torch.int32).flatten()[:8].view(1, 8)),
               (weights, weights.view(torch.int32))]
    for output_weights, output_ids in invalid:
        try:
            fused_topk_out(logits, 8, output_weights, output_ids)
        except RuntimeError:
            pass
        else:
            raise AssertionError("非法输出或别名未被拒绝")
        torch.testing.assert_close(logits, original, rtol=0, atol=0)
    storage = torch.empty(16, device="cuda")
    fused_topk_out(logits, 8, storage[:8].view(1, 8), storage[8:].view(torch.int32).view(1, 8))
    expected_weights, expected_ids = fused_topk(logits, 8)
    torch.testing.assert_close(storage[:8].view(1, 8), expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(storage[8:].view(torch.int32).view(1, 8), expected_ids, rtol=0, atol=0)
    print("[PASS] out 输出形状/类型/三种重叠被拒绝，不重叠的共享存储视图可用")
    print("[完成] 标准路由11用例及out内容、地址与别名检查；不是性能验证")


def check_grouped_case(name, logits, bias, *, near_boundary=False):
    weights, ids = fused_grouped_topk(logits, bias)
    fp32_weights, fp32_ids = grouped_reference(logits, bias, fp64=False)
    fp64_weights, fp64_ids = grouped_reference(logits, bias)
    torch.testing.assert_close(ids, fp32_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, fp32_weights, rtol=1e-5, atol=1e-6)
    if near_boundary:
        # FP64 重新计算的是同一组已选专家的权重，不拿另一精度的排序冒充 FP32 契约。
        selected = logits.double().sigmoid().gather(1, ids.long())
        fp64_weights = selected / selected.sum(dim=-1, keepdim=True) * GROUPED_SCALE
        assert not torch.equal(fp32_ids, fp64_ids), "近边界样本未构成预期的跨精度排序差异"
        print(f"[诊断] 近边界 FP32编号={fp32_ids.tolist()}，FP64编号={fp64_ids.tolist()}；不加epsilon")
    else:
        torch.testing.assert_close(ids, fp64_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.double(), fp64_weights, rtol=1e-5, atol=1e-6)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.shape == ids.shape == (logits.shape[0], 8)
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.full_like(weights[:, 0], GROUPED_SCALE),
                               rtol=1e-5, atol=1e-6)
    assert (ids.sort(-1).values[:, 1:] != ids.sort(-1).values[:, :-1]).all()
    corrected = (logits.sigmoid() + bias).reshape(-1, 8, 32)
    groups = corrected.topk(2, dim=-1).values.sum(-1).argsort(dim=-1, descending=True, stable=True)[:, :4]
    assert ((ids.long() // 32).unsqueeze(-1) == groups.unsqueeze(1)).any(-1).all()
    error = (weights.double() - fp64_weights).abs().max().item()
    print(f"[PASS] 分组{name}：T={logits.shape[0]}，保留组、编号、原始分数权重、scale通过，误差={error:.8g}")
    return weights, ids


def check_grouped() -> None:
    zero_bias = torch.zeros(256, device="cuda")
    for rows in (1, 17, 128, 4096):
        check_grouped_case("随机", torch.randn(rows, 256, device="cuda"),
                           torch.randn(256, device="cuda") * 0.1)
    check_grouped_case("零偏置", torch.randn(17, 256, device="cuda"), zero_bias)
    _, ids = check_grouped_case("全同分", torch.zeros(17, 256, device="cuda"), zero_bias)
    torch.testing.assert_close(ids, torch.arange(8, device="cuda", dtype=torch.int32).expand(17, -1))

    logits = torch.full((1, 256), -6.0, device="cuda")
    logits[0, 0] = 4.0
    logits[:, 32:160] = 0.0
    _, ids = check_grouped_case("筛组排除全局第一名", logits, zero_bias)
    assert 0 not in ids[0].tolist()
    torch.testing.assert_close(ids[0], torch.arange(32, 40, device="cuda", dtype=torch.int32))

    logits = torch.linspace(-4.0, 4.0, 256, device="cuda").repeat(17, 1)
    bias = zero_bias.clone()
    bias[:128] = 2.0
    bias[0] = 8.0
    weights, ids = check_grouped_case("偏置只影响选择", logits, bias)
    assert (ids[:, 0] == 0).all() and (weights[:, 0] < weights.max(-1).values).all()
    assert not torch.equal(ids, grouped_reference(logits, zero_bias, fp64=False)[1])
    check_grouped_case("负修正分数", logits, torch.full_like(zero_bias, -3.0))
    check_grouped_case("有限极值", torch.linspace(-10000.0, 10000.0, 256, device="cuda").repeat(17, 1), zero_bias)
    for sign in (-1, 1):
        check_grouped_case("有限偏置导致组和溢出", torch.zeros(1, 256, device="cuda"),
                           torch.full_like(zero_bias, sign * torch.finfo(torch.float32).max))

    near = torch.zeros(1, 256, device="cuda")
    near[0, -1] = 1e-8
    check_grouped_case("跨精度近边界", near, zero_bias, near_boundary=True)
    empty_mass = torch.full((1, 256), -torch.finfo(torch.float32).max, device="cuda")
    weights, ids = fused_grouped_topk(empty_mass, zero_bias)
    expected_weights, expected_ids = grouped_reference(empty_mass, zero_bias, fp64=False)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    assert torch.isnan(weights).all() and torch.isnan(expected_weights).all()
    print("[诊断] 零质量产生NaN，与原定义一致；该输入不属于有效权重支持域")
    print("[完成] 固定分组路由检查；尚未验证真实模型输入、成熟分组融合基线或性能")


def main() -> None:
    parser = argparse.ArgumentParser(description="标准/out与固定分组路由的云端检查")
    parser.add_argument("--routing", choices=["all", "standard", "grouped"], default="all")
    args = parser.parse_args()
    torch.manual_seed(0)
    print(f"[环境] GPU={torch.cuda.get_device_name(0)}，PyTorch={torch.__version__}，CUDA运行时={torch.version.cuda}")
    if args.routing in ("all", "standard"):
        check_standard()
    if args.routing in ("all", "grouped"):
        check_grouped()


if __name__ == "__main__":
    main()
