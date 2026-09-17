#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t kNumRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_bf16_mega_moe_backward_impl(void* dx,
                                  float* dw1_weights, float* dw2_weights, float* dtopk_weights,
                                  const void* dy,
                                  const uint32_t num_tokens,
                                  const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "sm100_bf16_mega_moe_backward_impl is not implemented yet");
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

}
