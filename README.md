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

源码目标为 RTX 3090（sm_86）、C++17；已在 Python 3.12、PyTorch 2.5.1+cu124、CUDA Toolkit 12.4 环境完成云端编译与 11 个正确性用例。

```python
import torch
from router import fused_topk

logits = torch.randn(17, 128, device="cuda", dtype=torch.float32)
weights, expert_ids = fused_topk(logits, k=8)
```

`routing.cu` 包含绑定、启动入口和内核；`router.py` 提供 FP64 数学参考；`check.py` 对照编号、权重和归一化。基线每 Warp 处理一行，每线程最多持有 8 项，通过成对归约重复选取。当前没有性能结果。

## 性能基线

在已激活的同一 CUDA 环境中运行，不需要重新编译扩展：

```bash
git rev-parse HEAD
python benchmark.py
```

默认测量 30 个 T/E/K 组合，各预热 20 次、测量 5 组，每组 100 次。可先用 `python benchmark.py --tokens 1 1024 --experts 128 --topk 8 --groups 3` 做短测。

两条路径共用输入，并在计时外核对输出。PyTorch 分步基线使用 FP32 全专家 Softmax、原 logits Top-K、收集与重归一化，包含 int32 编号转换；不以 FP64 正确性参考充当性能对手。记录 Event 流时间与包含调用、分配和末尾等待的墙钟时间，输出全部组样本及中位数。重复输入可能命中缓存，Event 也可能包含提交空隙；这不是纯内核时间、成熟融合基线或模型推理性能。
