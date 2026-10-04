# MoERoute：MoE 路由融合算子

把每个 Token 的专家分数转换为 Top-K 专家编号与重归一化权重。一个 CUDA Kernel 完成选择和权重处理，无需加载模型。

支持连续 CUDA FP32 输入 `[T,E]`，`T≥1`、`E=64/128/256`、`K=2/8`；启动网格受 CUDA x 维上限约束。输入约定为有限值，输出权重 FP32、编号 int32，均为 `[T,K]`。按原始 logits 降序选取，相等时较小编号优先。

数学参考执行全专家 Softmax、选取和重归一化；融合实现利用分母消去，先选 Top-K 再只对 K 项做稳定 Softmax。此变换要求重归一化开启。本版本只覆盖前向标准路由，不包含 Gate/Expert GEMM、Dispatch/Combine、反向或完整模型接入。

## 构建与检查

先激活 CUDA 服务器已有的 PyTorch 虚拟环境，再克隆并构建。CUDA Toolkit 主版本应与 PyTorch 的 CUDA 运行时匹配，`CUDA_HOME` 和 `PATH` 指向对应工具链；项目不自动安装依赖或切换环境。

```bash
git clone https://github.com/tdkjsmr/MoERoute.git
cd MoERoute
export OMP_NUM_THREADS=1
python setup.py build_ext --inplace
python check.py
```

源码目标为 RTX 3090（sm_86），C++ 语言标准由当前 PyTorch 构建工具选择。已验证环境为 PyTorch 2.13.0+cu130、CUDA Toolkit 13.0.48、RTX 3090；11 个正确性用例通过，覆盖随机形状、并列、接近值和极端有限值。不同工具链下的独立测量不用于计算配对加速比。

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

默认测量 `T=1/16/128/1024/4096`、`E=64/128/256`、`K=2/8` 共 30 个组合，各预热 20 次、测量 5 组，每组 100 次。PyTorch 分步基线执行 FP32 全专家 Softmax、原 logits Top-K、收集、重归一化和 int32 编号转换；FP64 数学参考只用于计时外正确性检查。

可选成熟基线要求已有 vLLM 0.29.0 环境，与当前 PyTorch/CUDA 工具链匹配后重新构建 MoERoute，再进行同进程三路径对测：

```bash
python benchmark.py --paths pytorch fused vllm
```

vLLM 调用原生 [`topk_softmax`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/_custom_ops.py)，开启 `renormalize=True`，计入三份输出/辅助张量分配。所有路径共用输入，分别对照 FP64 参考；特殊并列语义不作为跨实现等价契约。

### 已测结果

报告编号 `M4-3090-cu130`（本页归档标识），源码提交 [0e40b789a64452d3f5186b626b307ea9cadb2a4a](https://github.com/tdkjsmr/MoERoute/commit/0e40b789a64452d3f5186b626b307ea9cadb2a4a)。环境为上述 RTX 3090/cu130 组合及 vLLM 0.29.0；FP32 正态输入、种子 0，未启用 Graph、torch.compile 或预分配输出。

11 个正确性用例和 30 个形状的三路径输出对照通过；450 条组样本与 90 条中位数核对一致。下表为每组平均调用墙钟耗时的五组中位数，单位 μs；比值大于 1 表示 MoERoute 较快。

| T / E / K | PyTorch | MoERoute | vLLM | PyTorch / MoERoute | vLLM / MoERoute |
|---|---:|---:|---:|---:|---:|
| 1 / 256 / 8 | 59.362 | 7.280 | 22.118 | 8.15× | 3.04× |
| 1024 / 256 / 8 | 59.593 | 7.457 | 22.495 | 7.99× | 3.02× |
| 4096 / 64 / 8 | 59.876 | 8.440 | 22.404 | 7.09× | 2.65× |
| 4096 / 128 / 8 | 67.858 | 9.757 | 22.750 | 6.95× | 2.33× |
| 4096 / 256 / 2 | 115.668 | 7.584 | 22.768 | 15.25× | 3.00× |
| 4096 / 256 / 8 | 120.425 | 12.679 | 22.799 | 9.50× | 1.80× |

本次 30 个形状中，MoERoute 调用墙钟中位数为 7.212～12.679 μs；PyTorch/MoERoute 耗时比为 6.95～15.25×，vLLM/MoERoute 为 1.80～3.09×。

边界：计时包含正常调用、输出分配和组末等待，vLLM 还包含辅助输出；Event 也可能含宿主提交空隙。反复读取同一输入可能命中缓存；五组仅为少量重复，三路径顺序只首尾反转、融合固定居中。结果不是纯内核加速比、完整 MoE/模型推理性能或服务吞吐；不同软件栈的独立结果不能组成配对加速比。

时间线采集：`--profile` 仅接受单个 T/E/K，先检查与预热，再用 CUDA Profiler API 开关及 NVTX 标记采集各路径；不打印正式耗时。Nsight 原始文件放在已忽略的 `profiles/raw/`：

```bash
mkdir -p profiles/raw
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profiles/raw/router-t4096-e256-k8-01 \
  python benchmark.py --profile --tokens 4096 --experts 256 --topk 8
```

先用本机 `nsys profile --help` 确认采集选项可用；重跑换报告编号，不覆盖旧报告。分析时查看 Kernel 持续时间与提交间隙，不能用带 Profiler 的时间替换普通基准结果。
