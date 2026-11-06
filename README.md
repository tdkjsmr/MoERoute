# MoERoute：MoE 路由融合算子

标准 Softmax 路由把每个 Token 的专家 logits 转换为 Top-K 专家编号和重归一化权重。DeepSeek-V3 固定配置分组路由用带修正偏置的 sigmoid 分数筛组选专家，再用原始 sigmoid 分数计算权重；两条路径各用一个 CUDA Kernel 完成，无需加载模型。

标准路由支持连续 CUDA FP32 输入 `[T,E]`，`T≥1`、`E=64/128/256`、`K=2/8`，输入约定为有限值，按原始 logits 降序选取，同分优先小编号。输出统一为 FP32 权重和 int32 编号 `[T,K]`。

标准融合实现利用重归一化的分母消去，先选 Top-K，再只对 K 项做稳定 Softmax。项目提供前向路由算子，不包含 Gate/Expert GEMM、Dispatch/Combine、反向或完整模型接入。

## 构建与检查

先激活已有 PyTorch 环境，CUDA Toolkit 主版本与 PyTorch CUDA 运行时匹配，`CUDA_HOME` 和 `PATH` 指向对应工具链。以下直接克隆含分组实现的开发分支：

```bash
git clone --branch codex/grouped-routing https://github.com/tdkjsmr/MoERoute.git
cd MoERoute
export OMP_NUM_THREADS=1
python setup.py build_ext --inplace
python check.py
```

目标 GPU 为 RTX 3090（sm_86），C++ 标准由 PyTorch 构建工具选择。`check.py` 默认运行标准/out 与分组检查，也可用 `--routing standard` 或 `--routing grouped` 分别执行。

```python
import torch
from router import fused_topk, fused_grouped_topk

logits = torch.randn(17, 256, device="cuda", dtype=torch.float32)
weights, expert_ids = fused_topk(logits, k=8)
bias = torch.zeros(256, device="cuda", dtype=torch.float32)
grouped_weights, grouped_ids = fused_grouped_topk(logits, bias)
```

`routing.cu` 包含绑定、启动入口和内核，`router.py` 提供 FP32 分步实现与 FP64 数学参考。每 Warp 处理一行，分数/编号成对蝶形归约；标准路由按六种 E/K 组合编译期特化，展开候选扫描、移除及归一化。

## 固定分组路由

语义来源为 DeepSeek-V3 固定提交 `9b4e9788` 的 [Gate](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L498-L557) 与 [671B 配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json)，官方代码为 [MIT 许可](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/LICENSE-CODE)。本项目据其有偏置分支独立实现。

输入为 FP32 `[T,256]` logits 与 FP32 `[256]` 偏置，固定 8 组留 4 组、K=8、route_scale=2.5。先求 `p=sigmoid(logits)`，用 `p+bias` 的组内前两项之和选组，再按修正分数选专家；权重从未加偏置的 `p` 收集，归一化后乘 2.5。同分优先小组号/专家号，编号按修正分数降序输出，偏置仅作为输入使用。

有效权重要求有限输入及选中 sigmoid 分数总量非零；零质量保留原定义的 `0/0→NaN`。FP64 参考在 sigmoid 和偏置运算前升精度，检查将零质量及近边界跨精度排序差异单列诊断。

## 基准命令

CUDA 源码变化后先重新编译；切换 PyTorch 环境后用 `python setup.py build_ext --inplace --force` 重建扩展。默认标准矩阵为 `T=1/16/128/1024/4096`、`E=64/128/256`、`K=2/8` 共 30 个组合，各预热 20 次、测量 5 组，每组 100 次：

```bash
git rev-parse HEAD
python benchmark.py
python benchmark.py --routing grouped
```

分组默认测 `T=1/128/4096`、E256/K8，对照匹配语义的 FP32 PyTorch 分步实现；不以 FP64 检查参考充当性能对手。

已有 vLLM 0.29.0 环境中，匹配工具链并重新构建扩展后，可运行标准路由三路径基准或正常/out 配对：

```bash
python benchmark.py --paths pytorch fused vllm
python benchmark.py --compare-out --groups 8
```

标准 vLLM 入口为 [`topk_softmax`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/_custom_ops.py)，`renormalize=True`。`--compare-out` 默认测 T1/1024/4096、E256/K8，轮换双方正常与预分配入口；`fused_topk_out(logits,k,weights,ids)` 要求同设备、连续的 `[T,K]` 输出，输出与输入及彼此均不重叠，vLLM 也预留必要的辅助编号张量。

## 验证与结果

### 分组与 out 版本

源码提交 [48bf4d5853470d4e26d985e84457c2ef824b5300](https://github.com/tdkjsmr/MoERoute/commit/48bf4d5853470d4e26d985e84457c2ef824b5300)，GPU 为 RTX 3090。PyTorch 2.5.1+cu124、CUDA 运行时 12.4 的完整检查通过：标准 11 个用例、out 内容/地址/别名检查、分组 13 个有效用例。

以下为独立 cu130 调用墙钟基准：PyTorch 2.13.0+cu130、CUDA 运行时 13.0、vLLM 0.29.0，T1/E256/K8，预热 20 次、8 组×100 次。单位 µs：

| MoERoute 正常 | vLLM 正常 | MoERoute out | vLLM out |
|---:|---:|---:|---:|
| 7.727 | 22.288 | 4.253 | 10.135 |

调用数据取每组平均耗时的中位数，输入与正确性检查不计时；正常入口包含输出分配，out 提前分配所需输出及辅助张量。墙钟含调用和组末等待，Event 也可能含提交空隙；未使用 Graph 或 torch.compile，重复输入可能命中缓存。normal/out 还存在入口包装和检查差异，不能直接相减当作纯分配成本；调用结果不是纯内核或完整模型性能。

独立 cu12 Nsight 采集的分组 T128/E256/K8 中，100 次调用对应 PyTorch 1800 个内核、融合 100 个，即平均每次由 18 个内核降为 1 个；该采集与上述 cu130 调用结果分别记录，不混合推算宿主或设备贡献。

### 历史标准路由正常调用

源码提交 [0e40b789a64452d3f5186b626b307ea9cadb2a4a](https://github.com/tdkjsmr/MoERoute/commit/0e40b789a64452d3f5186b626b307ea9cadb2a4a)，RTX 3090、PyTorch 2.13.0+cu130、vLLM 0.29.0。30 个形状的三路径输出对照通过，450 条组样本、90 条中位数核对一致；相对标准 vLLM `topk_softmax` 正常调用，vLLM/MoERoute 耗时比为 1.80～3.09×。这是正常输出分配口径的路由调用结果，不用于分组或 out 版本。

## 时间线采集

`--profile` 只采集一个 T/E/K，检查与预热在采集窗口外，不输出正式基准耗时。下面每次建立唯一目录，避免覆盖旧报告：

```bash
mkdir -p profiles/raw
task_profile_dir=$(mktemp -d "$PWD/profiles/raw/router-nsys-XXXXXXXX")
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="$task_profile_dir/router-t4096-e256-k8" \
  python benchmark.py --profile --tokens 4096 --experts 256 --topk 8
```
