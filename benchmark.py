"""云端路由调用基线：融合算子对照 FP32 PyTorch 分步实现，不测模型。"""

import argparse
import statistics
import time

import torch

from router import fused_topk, reference


def pytorch_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """全专家 Softmax、按原 logits 选取、收集并重归一化。"""
    probabilities = torch.softmax(logits, dim=-1)
    # 按原 logits 而非舍入后的概率排序，与融合算子的选择依据相同。
    ids = torch.topk(logits, k, dim=-1, sorted=True).indices
    weights = probabilities.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    # 两条路径都输出 int32，转换成本也计入分步基线。
    return weights, ids.to(torch.int32)


def measure(fn, logits, k, iterations, start, end) -> tuple[float, float]:
    """返回每次调用的 Event/墙钟微秒数；同步只放在整组边界。"""
    torch.cuda.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        # 不保存全部输出，避免随迭代次数积累显存；两侧都正常分配结果。
        fn(logits, k)
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - begin) * 1e6 / iterations
    event_us = start.elapsed_time(end) * 1e3 / iterations
    return event_us, wall_us


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="MoE 路由融合与 PyTorch 分步调用基线")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128, 1024, 4096])
    parser.add_argument("--experts", nargs="+", type=int, choices=[64, 128, 256], default=[64, 128, 256])
    parser.add_argument("--topk", nargs="+", type=int, choices=[2, 8], default=[2, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens) < 1 or min(args.iterations, args.groups) < 1 or args.warmup < 1:
        parser.error("Token 数、预热、每组次数和组数必须为正整数")

    torch.manual_seed(0)
    print(f"[环境] GPU={torch.cuda.get_device_name(0)}，PyTorch={torch.__version__}，CUDA运行时={torch.version.cuda}")
    print(f"[配置] FP32 正态随机输入，种子=0；T={args.tokens}，E={args.experts}，K={args.topk}")
    print(f"[配置] 每形状每路径预热={args.warmup} 次，测量={args.groups} 组，每组={args.iterations} 次；组间交替顺序")
    print("[口径] 输入提前准备，检查不计时；两侧包含正常输出分配，无 Graph、编译融合或预分配输出")
    print("[口径] Event 为流上首尾事件间隔，可能含宿主提交空隙；墙钟含 Python 调用、分配与组末等待，均不是纯内核时间")
    print("[边界] 反复读取同一输入可能命中硬件缓存；分步基线不保证并列编号顺序，特殊并列规则由 check.py 验证")

    paths = [("PyTorch", pytorch_topk), ("融合", fused_topk)]
    # 事件在计时外创建并初始化，避免把首次事件初始化计入第一组。
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    for tokens in args.tokens:
        for experts in args.experts:
            for k in args.topk:
                logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
                expected_weights, expected_ids = reference(logits, k)
                # 两条路径分别对照独立 FP64 参考，不用慢参考参与性能竞争。
                for name, fn in paths:
                    weights, ids = fn(logits, k)
                    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                    torch.testing.assert_close(weights.double(), expected_weights, rtol=1e-5, atol=1e-6)
                del weights, ids, expected_weights, expected_ids
                for _, fn in paths:
                    for _ in range(args.warmup):
                        fn(logits, k)
                torch.cuda.synchronize()

                samples = {name: [] for name, _ in paths}
                print(f"[形状] T={tokens}，E={experts}，K={k}；两路径计时外编号与权重对照通过")
                for group in range(args.groups):
                    # 交替先测哪条路径；保留全部组均值，不只报告最快的一组。
                    order = paths if group % 2 == 0 else paths[::-1]
                    for name, fn in order:
                        event_us, wall_us = measure(fn, logits, k, args.iterations, start, end)
                        samples[name].append((event_us, wall_us))
                        print(f"[样本] 组={group + 1}，{name}：Event={event_us:.3f} us，墙钟={wall_us:.3f} us")
                medians = {}
                for name, _ in paths:
                    event_us = statistics.median(item[0] for item in samples[name])
                    wall_us = statistics.median(item[1] for item in samples[name])
                    medians[name] = (event_us, wall_us)
                    print(f"[中位数] T={tokens} E={experts} K={k}，{name}：Event={event_us:.3f} us，墙钟={wall_us:.3f} us")
                event_ratio = medians["PyTorch"][0] / medians["融合"][0]
                wall_ratio = medians["PyTorch"][1] / medians["融合"][1]
                print(f"[配对] 分步/融合耗时比：Event={event_ratio:.3f}×，墙钟={wall_ratio:.3f}×；大于1表示本次融合较快")
    print("[完成] 路由调用基线；不是成熟融合内核对比、模型推理延迟或服务吞吐")


if __name__ == "__main__":
    main()
