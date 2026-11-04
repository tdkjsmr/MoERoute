#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
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
constexpr float kGroupedScale = 2.5f;

// 值和编号始终一起比较：降序按值，完全相等时小编号胜出。
__device__ __forceinline__ bool better(float value, int id, float best, int best_id) {
  return value > best || (value == best && id < best_id);
}

// 所有 lane 同时参与；分数和编号成对合并，同分优先小编号。
__device__ __forceinline__ void warp_best(float& best, int& best_id) {
  #pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(kFullMask, best, offset);
    const int other_id = __shfl_xor_sync(kFullMask, best_id, offset);
    if (better(other, other_id, best, best_id)) {
      best = other;
      best_id = other_id;
    }
  }
}

// 六种已支持的 E/K 组合分别编译，不在设备端使用动态循环边界。
template <int Experts, int K>
__global__ void routing_kernel(const float* logits, float* weights, int* ids,
                               int64_t rows) {
  const int lane = threadIdx.x % kWarp;
  const int64_t row = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  // 一整 Warp 负责同一行，因此尾部退出不会留下半个 Warp 参与 shuffle。
  if (row >= rows) return;

  // lane 负责 lane、lane+32……；E≤256，每线程至多保存八个候选。
  constexpr int count = Experts / kWarp;
  float values[count];
  float selected[K];
  #pragma unroll
  for (int slot = 0; slot < count; ++slot) {
    values[slot] = logits[row * Experts + lane + slot * kWarp];
  }

  #pragma unroll
  for (int rank = 0; rank < K; ++rank) {
    float best = -CUDART_INF_F;
    int best_id = Experts;
    #pragma unroll
    for (int slot = 0; slot < count; ++slot) {
      const int id = lane + slot * kWarp;
      if (better(values[slot], id, best, best_id)) {
        best = values[slot];
        best_id = id;
      }
    }

    // 蝶形归约：每轮与 lane ^ offset 交换，五轮后所有 lane 都持有整行赢家。
    // 分数与编号成对归约，仍按分数降序、相等时小编号优先，无需再广播。
    warp_best(best, best_id);
    if (lane == 0) {
      selected[rank] = best;
      ids[row * K + rank] = best_id;
    }
    // 展开后每个候选槽的下标固定，避免按赢家编号动态寻址数组。
    // 仍只有赢家对应的一项被移除；具体寄存器分配由编译器决定。
    #pragma unroll
    for (int slot = 0; slot < count; ++slot) {
      if (best_id == lane + slot * kWarp) values[slot] = -CUDART_INF_F;
    }
  }

  if (lane == 0) {
    // 重归一化约去全专家 Softmax 分母，只需对选中 K 项求指数。
    // selected[0] 是最大值；极端有限差值可变为负无穷，exp 得到合法的零。
    const float maximum = selected[0];
    float sum = 0.0f;
    #pragma unroll
    for (int rank = 0; rank < K; ++rank) {
      selected[rank] = expf(selected[rank] - maximum);
      sum += selected[rank];
    }
    #pragma unroll
    for (int rank = 0; rank < K; ++rank) weights[row * K + rank] = selected[rank] / sum;
  }
}

// 固定分组：每个槽对应一组的 32 个专家，偏置只改变选择，不改变输出原分数。
__global__ void grouped_routing_kernel(const float* logits, const float* bias,
                                       float* weights, int* ids, int64_t rows) {
  const int lane = threadIdx.x % kWarp;
  const int64_t row = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) return;
  float probabilities[8], scores[8], selected[8];
  float group_candidate = -CUDART_INF_F;
  #pragma unroll
  for (int group = 0; group < 8; ++group) {
    const int expert = group * kWarp + lane;
    probabilities[group] = 1.0f / (1.0f + expf(-logits[row * 256 + expert]));
    scores[group] = probabilities[group] + bias[expert];
    float first = scores[group];
    int first_id = lane;
    warp_best(first, first_id);
    float second = lane == first_id ? -CUDART_INF_F : scores[group];
    int second_id = lane;
    warp_best(second, second_id);
    if (lane == group) group_candidate = first + second;
  }
  unsigned retained = 0;
  int candidate_id = lane < 8 ? lane : 8;
  #pragma unroll
  for (int rank = 0; rank < 4; ++rank) {
    float best = group_candidate;
    int best_id = candidate_id;
    warp_best(best, best_id);
    retained |= 1u << best_id;
    if (lane == best_id) {
      group_candidate = -CUDART_INF_F;
      // 有限负偏置的两项之和也可能为 -Inf，编号哨兵避免重复选中已移除组。
      candidate_id = 8;
    }
  }
  #pragma unroll
  for (int group = 0; group < 8; ++group) {
    if (!(retained & (1u << group))) scores[group] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int rank = 0; rank < 8; ++rank) {
    float best = -CUDART_INF_F, local_probability = 0.0f;
    int best_id = 256;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
      const int expert = group * kWarp + lane;
      if (better(scores[group], expert, best, best_id)) {
        best = scores[group];
        best_id = expert;
        local_probability = probabilities[group];
      }
    }
    warp_best(best, best_id);
    const float original = __shfl_sync(kFullMask, local_probability, best_id % kWarp);
    if (lane == 0) {
      ids[row * 8 + rank] = best_id;
      selected[rank] = original;
    }
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (best_id == group * kWarp + lane) scores[group] = -CUDART_INF_F;
    }
  }
  if (lane == 0) {
    float sum = 0.0f;
    #pragma unroll
    for (int rank = 0; rank < 8; ++rank) sum += selected[rank];
    // 保留官方零分母行为，不使用 Softmax 化简或 epsilon 回退。
    #pragma unroll
    for (int rank = 0; rank < 8; ++rank) weights[row * 8 + rank] = selected[rank] / sum * kGroupedScale;
  }
}

// 专家数由入口分派，K 只在宿主端分支；两条路径保持相同启动布局。
template <int Experts>
void launch_routing(const float* logits, float* weights, int* ids, int64_t rows,
                    int64_t k, unsigned blocks, cudaStream_t stream) {
  if (k == 2) {
    routing_kernel<Experts, 2><<<blocks, kWarp * kWarpsPerBlock, 0, stream>>>(
        logits, weights, ids, rows);
  } else {
    routing_kernel<Experts, 8><<<blocks, kWarp * kWarpsPerBlock, 0, stream>>>(
        logits, weights, ids, rows);
  }
}

void check_logits(const at::Tensor& logits) {
  // 仅在公开扩展入口检查元数据；输入有限性属于调用契约，不取回设备标量。
  TORCH_CHECK(logits.is_cuda(), "logits 必须在 CUDA 设备上");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits 必须为 FP32");
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(), "logits 必须是连续二维张量");
  const int64_t rows = logits.size(0);
  const int64_t experts = logits.size(1);
  TORCH_CHECK(rows >= 1 && rows <= int64_t(2147483647) * kWarpsPerBlock,
              "Token 数必须为正且启动网格不能超过 CUDA 的 x 维上限");
  TORCH_CHECK(experts == 64 || experts == 128 || experts == 256, "E 只支持 64/128/256");
}

void run_standard(const at::Tensor& logits, int64_t k, const at::Tensor& weights,
                  const at::Tensor& ids) {
  // 两个公开入口已设置设备，启动逻辑共用，不重复分配或重新设置 guard。
  const int64_t rows = logits.size(0), experts = logits.size(1);
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const float* input = logits.data_ptr<float>();
  float* output = weights.data_ptr<float>();
  int* output_ids = ids.data_ptr<int>();
  switch (experts) {
    case 64: launch_routing<64>(input, output, output_ids, rows, k, blocks, stream.stream()); break;
    case 128: launch_routing<128>(input, output, output_ids, rows, k, blocks, stream.stream()); break;
    case 256: launch_routing<256>(input, output, output_ids, rows, k, blocks, stream.stream()); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_topk_out(const at::Tensor& logits, int64_t k, const at::Tensor& weights,
                    const at::Tensor& ids) {
  check_logits(logits);
  TORCH_CHECK(k == 2 || k == 8, "K 只支持 2/8");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == logits.size(0) && weights.size(1) == k &&
              ids.sizes() == weights.sizes(), "输出形状必须为 [T,K]");
  TORCH_CHECK(weights.scalar_type() == at::kFloat && ids.scalar_type() == at::kInt,
              "输出类型必须为 FP32 权重与 int32 编号");
  TORCH_CHECK(weights.device() == logits.device() && ids.device() == logits.device() &&
              weights.is_contiguous() && ids.is_contiguous(), "输出必须连续且与输入同设备");
  at::assert_no_overlap(logits, weights);
  at::assert_no_overlap(logits, ids);
  at::assert_no_overlap(weights, ids);
  const c10::cuda::CUDAGuard guard(logits.device());
  run_standard(logits, k, weights, ids);
}

std::vector<at::Tensor> fused_topk(const at::Tensor& logits, int64_t k) {
  check_logits(logits);
  TORCH_CHECK(k == 2 || k == 8, "K 只支持 2/8");
  const c10::cuda::CUDAGuard guard(logits.device());
  auto weights = at::empty({logits.size(0), k}, logits.options());
  auto ids = at::empty({logits.size(0), k}, logits.options().dtype(at::kInt));
  run_standard(logits, k, weights, ids);
  return {weights, ids};
}

std::vector<at::Tensor> fused_grouped_topk(const at::Tensor& logits, const at::Tensor& bias) {
  check_logits(logits);
  TORCH_CHECK(logits.size(1) == 256, "分组路由只支持 E=256、8 组留 4 组、K=8");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 256 && bias.scalar_type() == at::kFloat &&
              bias.device() == logits.device() && bias.is_contiguous(), "偏置必须为同设备连续 FP32 [256]");
  const c10::cuda::CUDAGuard guard(logits.device());
  const int64_t rows = logits.size(0);
  auto weights = at::empty({rows, 8}, logits.options());
  auto ids = at::empty({rows, 8}, logits.options().dtype(at::kInt));
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  grouped_routing_kernel<<<blocks, kWarp * kWarpsPerBlock, 0, stream.stream()>>>(
      logits.data_ptr<float>(), bias.data_ptr<float>(), weights.data_ptr<float>(),
      ids.data_ptr<int>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weights, ids};
}
}  // 结束路由命名空间。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_topk", &moe_router::fused_topk,
             "标准 Top-K 路由与选中权重重归一化", py::arg("logits"), py::arg("k"));
  module.def("fused_topk_out", &moe_router::fused_topk_out,
             "标准路由写入预分配且不重叠的输出", py::arg("logits"), py::arg("k"),
             py::arg("weights"), py::arg("ids"));
  module.def("fused_grouped_topk", &moe_router::fused_grouped_topk,
             "固定分组偏置 sigmoid 路由", py::arg("logits"), py::arg("bias"));
}
