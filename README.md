# MoERoute：MoE 路由融合算子

把每个 Token 的专家分数转换为 Top-K 专家编号与重归一化权重。一个 CUDA Kernel 完成选择和权重处理，无需加载模型。

支持连续 CUDA FP32 输入 `[T,E]`，`T≥1`、`E=64/128/256`、`K=2/8`；启动网格受 CUDA x 维上限约束。输入约定为有限值，输出权重 FP32、编号 int32，均为 `[T,K]`。按原始 logits 降序选取，相等时较小编号优先。

数学参考执行全专家 Softmax、选取和重归一化；融合实现利用分母消去，先选 Top-K 再只对 K 项做稳定 Softmax。此变换要求重归一化开启。本版本只覆盖前向标准路由。

## 构建与检查

先激活 CUDA 服务器已有的 PyTorch 虚拟环境。首次源码推送后，克隆并进入独立仓库执行：

```bash
git clone https://github.com/tdkjsmr/MoERoute.git
cd MoERoute
export OMP_NUM_THREADS=1
python setup.py build_ext --inplace
python check.py
```

源码目标为 RTX 3090（sm_86）、C++17；此前基线已在 Python 3.12、PyTorch 2.5.1+cu124、CUDA Toolkit 12.4 环境完成云端编译与 11 个正确性用例，E/K 编译期特化版已通过 30 形状基准的计时外输出对照。当前蝶形归约改动尚待云端重新编译和验证。

```python
import torch
from router import fused_topk

logits = torch.randn(17, 128, device="cuda", dtype=torch.float32)
weights, expert_ids = fused_topk(logits, k=8)
```

`routing.cu` 包含绑定、启动入口和内核；`router.py` 提供 FP64 数学参考；`check.py` 对照编号、权重和归一化。每 Warp 处理一行，每线程最多持有 8 项，通过分数/编号成对的蝶形归约重复选取，各 lane 直接得到赢家，无需额外广播。六种 E/K 组合在编译时特化，展开候选扫描、移除及归一化循环，不改变选择规则。

## 性能基线

在已激活的同一 CUDA 环境中运行。CUDA 源码更新后须先重新编译扩展；扩展与源码一致时可直接测量：

```bash
git rev-parse HEAD
python benchmark.py
```

默认测量 30 个 T/E/K 组合，各预热 20 次、测量 5 组，每组 100 次。可先用 `python benchmark.py --tokens 1 1024 --experts 128 --topk 8 --groups 3` 做短测。

两条路径共用输入，并在计时外核对输出。PyTorch 分步基线使用 FP32 全专家 Softmax、原 logits Top-K、收集与重归一化，包含 int32 编号转换；不以 FP64 正确性参考充当性能对手。记录 Event 流时间与包含调用、分配和末尾等待的墙钟时间，输出全部组样本及中位数。重复输入可能命中缓存，Event 也可能包含提交空隙；这不是纯内核时间、成熟融合基线或模型推理性能。

时间线采集：`--profile` 仅接受单个 T/E/K，先检查与预热，再用 CUDA Profiler API 开关及 NVTX 标记采集各路径；不打印正式耗时。Nsight 原始文件放在已忽略的 `profiles/raw/`：

```bash
mkdir -p profiles/raw
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profiles/raw/router-t4096-e256-k8-01 \
  python benchmark.py --profile --tokens 4096 --experts 256 --topk 8
```

先用本机 `nsys profile --help` 确认采集选项可用；重跑换报告编号，不覆盖旧报告。分析时查看 Kernel 持续时间与提交间隙，不能用带 Profiler 的时间替换普通基准结果。

可选成熟基线：只在**已有** vLLM 0.29.0 环境中执行 `python benchmark.py --paths pytorch vllm`，无需加载模型或 MoERoute 扩展。调用原生 `topk_softmax` 并开启内部重归一化，计入三份输出/辅助张量分配。接口来源：[vLLM 0.29.0](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/_custom_ops.py)。不自动安装依赖；不同 PyTorch/CUDA 环境的结果不能直接组成 MoERoute/vLLM 配对加速比。当前只是接口适配，仍待云端运行验证；特殊并列语义未作为跨实现等价契约。
