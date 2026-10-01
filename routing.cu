#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>  // 显式提供 CUDART_INF_F 等 CUDA 数学常量。
#include <cmath>
#include <vector>

namespace moe_router {
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;
constexpr unsigned kFullMask = 0xffffffffu;

// 值和编号始终一起比较：降序按值，完全相等时小编号胜出。
__device__ __forceinline__ bool better(float value, int id, float best, int best_id) {
  return value > best || (value == best && id < best_id);
}

__global__ void routing_kernel(const float* logits, float* weights, int* ids,
                               int64_t rows, int experts, int k) {
  const int lane = threadIdx.x % kWarp;
  const int64_t row = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  // 一整 Warp 负责同一行，因此尾部退出不会留下半个 Warp 参与 shuffle。
  if (row >= rows) return;

  // lane 负责 lane、lane+32……；E≤256，每线程至多保存八个候选。
  float values[8];
  float selected[8];
  const int count = experts / kWarp;
  for (int slot = 0; slot < count; ++slot) {
    values[slot] = logits[row * experts + lane + slot * kWarp];
  }

  for (int rank = 0; rank < k; ++rank) {
    float best = -CUDART_INF_F;
    int best_id = experts;
    for (int slot = 0; slot < count; ++slot) {
      const int id = lane + slot * kWarp;
      if (better(values[slot], id, best, best_id)) {
        best = values[slot];
        best_id = id;
      }
    }

    // 所有 32 个线程都到达这里；归约后 lane 0 持有整行赢家。
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const float other = __shfl_down_sync(kFullMask, best, offset);
      const int other_id = __shfl_down_sync(kFullMask, best_id, offset);
      if (better(other, other_id, best, best_id)) {
        best = other;
        best_id = other_id;
      }
    }
    const float winner = __shfl_sync(kFullMask, best, 0);
    const int winner_id = __shfl_sync(kFullMask, best_id, 0);
    if (lane == 0) {
      selected[rank] = winner;
      ids[row * k + rank] = winner_id;
    }
    // 只由赢家所在的线程删去其寄存器候选，下一轮不会重复选择。
    if (lane == winner_id % kWarp) values[winner_id / kWarp] = -CUDART_INF_F;
  }

  if (lane == 0) {
    // 重归一化约去全专家 Softmax 分母，只需对选中 K 项求指数。
    // selected[0] 是最大值；极端有限差值可变为负无穷，exp 得到合法的零。
    const float maximum = selected[0];
    float sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
      selected[rank] = expf(selected[rank] - maximum);
      sum += selected[rank];
    }
    for (int rank = 0; rank < k; ++rank) weights[row * k + rank] = selected[rank] / sum;
  }
}

std::vector<at::Tensor> fused_topk(const at::Tensor& logits, int64_t k) {
  // 仅在公开扩展入口检查元数据；输入有限性属于调用契约，不取回设备标量。
  TORCH_CHECK(logits.is_cuda(), "logits 必须在 CUDA 设备上");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits 必须为 FP32");
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(), "logits 必须是连续二维张量");
  const int64_t rows = logits.size(0);
  const int64_t experts = logits.size(1);
  TORCH_CHECK(rows >= 1 && rows <= int64_t(2147483647) * kWarpsPerBlock,
              "Token 数必须为正且启动网格不能超过 CUDA 的 x 维上限");
  TORCH_CHECK(experts == 64 || experts == 128 || experts == 256, "E 只支持 64/128/256");
  TORCH_CHECK(k == 2 || k == 8, "K 只支持 2/8");

  const c10::cuda::CUDAGuard guard(logits.device());
  auto weights = at::empty({rows, k}, logits.options());
  auto ids = at::empty({rows, k}, logits.options().dtype(at::kInt));
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  routing_kernel<<<blocks, kWarp * kWarpsPerBlock, 0, stream.stream()>>>(
      logits.data_ptr<float>(), weights.data_ptr<float>(), ids.data_ptr<int>(), rows,
      static_cast<int>(experts), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weights, ids};
}
}  // 结束路由命名空间。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_topk", &moe_router::fused_topk,
             "标准 Top-K 路由与选中权重重归一化", py::arg("logits"), py::arg("k"));
}
