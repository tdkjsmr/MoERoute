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

源码目标为 RTX 3090（sm_86）、C++17；起始环境为 Python 3.12、PyTorch 2.5.1+cu124、CUDA Toolkit 12.4。首轮等待云端验证。

```python
import torch
from router import fused_topk

logits = torch.randn(17, 128, device="cuda", dtype=torch.float32)
weights, expert_ids = fused_topk(logits, k=8)
```

`routing.cu` 包含绑定、启动入口和内核；`router.py` 提供 FP64 数学参考；`check.py` 对照编号、权重和归一化。基线每 Warp 处理一行，每线程最多持有 8 项，通过成对归约重复选取。当前没有性能结果。
