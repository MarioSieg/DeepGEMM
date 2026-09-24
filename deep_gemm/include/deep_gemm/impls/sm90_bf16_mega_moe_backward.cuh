#pragma once
#undef DG_DEVICE_PRINTF
#define DG_DEVICE_PRINTF(...) do {} while (0)

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

namespace sm90_mega_moe_backward {

template <bool kFastMath>
[[nodiscard]] __device__ __forceinline__ float sigmoid(float x) {
    if constexpr (kFastMath) return 1.0f / (1.0f + __expf(-x));
    else return 1.0f / (1.0f + expf(-x));
}

CUTLASS_DEVICE void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

CUTLASS_DEVICE uint32_t pack_bf16x2(const float& a, const float& b) {
    const auto h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

} // namespace sm90_mega_moe_backward
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t kNumRingTokens,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kNumStages,
    typename dw_t,
    bool kDwNatural,
    bool kL1Natural,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumPasses = kHasShared ? kNumSharedExperts : 1,
    uint32_t kGran = 8
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_bf16_mega_moe_backward_impl(
    void* __restrict__ dx,
    dw_t* __restrict__ dw1_weights,
    dw_t* __restrict__ dw2_weights,
    float* __restrict__ dtopk_weights,
    dw_t* __restrict__ shared_dw1_weights,
    dw_t* __restrict__ shared_dw2_weights,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w1_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w1_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_w1_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_w1_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_w2_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_x_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_x_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dy_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dy_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dz_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dz_mn,
    const __grid_constant__ cute::TmaDescriptor tensor_map_hw_k
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    using namespace sm90_mega_moe_backward;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using bf16_t = cutlass::bfloat16_t;
    using BlockDesc = layout::MegaMoEBackwardBuffer::BlockDesc;
    constexpr uint32_t kNumThreads = 384;
    constexpr uint32_t kNumProducerThreads = 128;
    constexpr uint32_t kNumConsumerWarpgroups = 2;
    constexpr uint32_t kNumConsumerThreadsPerWG = 128;
    constexpr uint32_t kNumGatherWarps = 2;
    constexpr uint32_t kSchedulerWarpIdx = 0, kTMAWarpIdx = 1, kGatherWarpStartIdx = 2;
    constexpr uint32_t kConsumerBarrierStartIdx = 1;
    constexpr uint32_t BLOCK_M = layout::MegaMoEBackwardBuffer::kBlockM;
    constexpr uint32_t kNumZSlots = layout::MegaMoEBackwardBuffer::kNumZSlots;
    constexpr uint32_t MMA_M = 128;
    constexpr uint32_t MMA_N = 128;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t WGMMA_M = 64;
    constexpr uint32_t WGMMA_K = 16;
    constexpr uint32_t kNumMHalves = MMA_M / WGMMA_M;
    constexpr uint32_t kNumAccumPerHalf = WGMMA_M * MMA_N / 128;
    constexpr uint32_t kNumTokenGroups = MMA_N / 8;
    constexpr uint32_t CHUNK = 32;
    constexpr uint32_t kNumGatherChunks = BLOCK_M / CHUNK;
    constexpr uint32_t kSwizzleMode = 128;
    constexpr uint32_t kNumItemSlots = 4;
    constexpr uint32_t DX_STAGE_TOKENS = 16;
    constexpr uint32_t kNumDxChunks = BLOCK_M / DX_STAGE_TOKENS;
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t I2 = kIntermediateHidden << 1;
    constexpr uint32_t kNumG1Tiles = I2 / MMA_M;
    constexpr uint32_t kNumG2Tiles = kIntermediateHidden / MMA_M;
    constexpr uint32_t kNumHTiles = kHidden / MMA_M;
    constexpr uint32_t kNumKBlocksH = kHidden / BLOCK_K;
    constexpr uint32_t kNumKBlocksI2 = I2 / BLOCK_K;
    constexpr uint32_t kNumDW2Tiles = kNumHTiles * kNumG2Tiles;
    constexpr uint32_t kNumDW1Tiles = kNumG1Tiles * kNumHTiles;
    constexpr uint32_t kNumTilesPerExpert = kNumDW2Tiles + kNumDW1Tiles;
    constexpr uint32_t kNumSharedSlots = kHasShared ? kNumPasses : 0;
    constexpr uint32_t kNumExpertSlots = kNumExpertsPerRank + kNumSharedSlots;
    constexpr uint32_t kNumVecPerRow = (kHidden << 1) >> 4;
    constexpr uint32_t kVecPerLane = kNumVecPerRow >> 5;
    constexpr uint32_t kVecUnroll = (1 & kVecPerLane) == 0 ? 2 : 1;
    constexpr uint32_t kKindGather = 0, kKindZ = 1, kKindDz = 2, kKindDx = 3, kKindDw = 4;
    constexpr uint32_t kGroupBlocks = 32;
    static_assert(kNumZSlots % kGroupBlocks == 0, "Invalid group size");

    static_assert(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    static_assert(kHidden % MMA_M == 0 && kIntermediateHidden % MMA_M == 0, "Invalid hidden sizes");
    static_assert(kIntermediateHidden % kGran == 0, "Invalid intermediate hidden for gate/up interleaving");
    static_assert(kGran == 8, "The fragment layout relies on granularity-8 gate/up interleaving");
    static_assert(kNumTopk <= 32, "Invalid number of topk");
    static_assert(kNumSMs > 1, "Invalid SM count");
    static_assert(BLOCK_M == MMA_N && BLOCK_M % BLOCK_K == 0 && BLOCK_M == kNumConsumerThreadsPerWG, "Invalid token block");
    static_assert((31 & kNumVecPerRow) == 0, "Invalid hidden for the gather");
    static_assert(kNumExpertSlots <= layout::MegaMoEBackwardBuffer::kMaxExpertSlots, "Too many experts per rank");
    static_assert(kNumStages >= 2, "Invalid number of stages");
    constexpr uint32_t kNumProducerRegisters = 88;
    constexpr uint32_t kNumConsumerRegisters = 208;
    static_assert(kNumProducerRegisters * kNumProducerThreads +
                  kNumConsumerRegisters * kNumConsumerWarpgroups * kNumConsumerThreadsPerWG <= 65536, "Too many registers");

    const uint32_t tid = threadIdx.x;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t warp = cutlass::canonical_warp_idx_sync();
    const uint32_t lane = ptx::get_lane_idx();
    constexpr uint32_t kNumGlobalThreads = kNumSMs * kNumThreads;
    const uint32_t global_tid = sm_idx * kNumThreads + tid;

    if (warp == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_w1_k);
        cute::prefetch_tma_descriptor(&tensor_map_w1_mn);
        cute::prefetch_tma_descriptor(&tensor_map_w2_mn);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w1_k);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w1_mn);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w2_mn);
        cute::prefetch_tma_descriptor(&tensor_map_x_k);
        cute::prefetch_tma_descriptor(&tensor_map_x_mn);
        cute::prefetch_tma_descriptor(&tensor_map_dy_k);
        cute::prefetch_tma_descriptor(&tensor_map_dy_mn);
        cute::prefetch_tma_descriptor(&tensor_map_dz_k);
        cute::prefetch_tma_descriptor(&tensor_map_dz_mn);
        cute::prefetch_tma_descriptor(&tensor_map_hw_k);
    }

    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        false,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;
    const auto bw = layout::MegaMoEBackwardBuffer(
        buffer.get_end_ptr(),
        kHidden, kIntermediateHidden, kNumMaxTokensPerRank, kNumTopk, kNumSharedExperts,
        workspace.num_max_pool_tokens, kNumSMs
    );
    auto* x_pool = static_cast<nv_bfloat16*>(bw.x_pool);
    auto* dy_pool = static_cast<nv_bfloat16*>(bw.dy_pool);
    auto* dz_pool = static_cast<nv_bfloat16*>(bw.dz_pool);
    auto* hw_pool = static_cast<nv_bfloat16*>(bw.hw_pool);
    auto* z_scratch = static_cast<float*>(bw.z_scratch);
    const uint64_t pool_stride = bw.num_pool_rows;

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads)
        *workspace.get_expert_send_count_ptr(i) = 0;
    for (uint32_t i = global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 101>(workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); }, false, true);

    for (uint32_t idx = global_tid; idx < num_tokens * kNumTopk; idx += kNumGlobalThreads) {
        const int64_t expert_idx = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[idx];
        if (expert_idx < 0) continue;
        const auto dst_rank = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
        const auto dst_local_expert = static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
        const auto slot = static_cast<uint32_t>(ptx::atomic_add(workspace.get_expert_send_count_ptr(expert_idx), static_cast<uint64_t>(1)));
        const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(dst_local_expert, sym_buffer.rank_idx, slot);
        *sym_buffer.map(dst_ptr, dst_rank) = idx;
    }
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 102>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads) {
        const auto dst_rank = i / kNumExpertsPerRank;
        const auto dst_local_expert = i % kNumExpertsPerRank;
        const auto count = static_cast<uint32_t>(*workspace.get_expert_send_count_ptr(i));
        *sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert), dst_rank) = count;
        ptx::atomic_add_sys(sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert), dst_rank), static_cast<uint64_t>(count));
    }
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 103>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });

    if (sm_idx == 0 && tid == 0) {
        uint32_t pool_base = 0, meta_base = 0, num_blocks = 0;
        for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
            const auto valid_m = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            const auto blocks_e = math::ceil_div(valid_m, BLOCK_M);
            bw.expert_pool_base[e] = pool_base;
            bw.expert_num_blocks[e] = blocks_e;
            bw.expert_done[e] = 0;
            bw.expert_a2_target[e] = blocks_e * kNumG2Tiles;
            for (uint32_t off = 0; off < valid_m; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = e;
                bw.block_desc[num_blocks].pool_begin = pool_base + off;
                bw.block_desc[num_blocks].meta_begin = meta_base + off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, valid_m - off);
                ++ num_blocks;
            }
            pool_base += blocks_e * BLOCK_M;
            meta_base += valid_m;
        }
        const uint32_t num_routed_blocks = num_blocks;
        if constexpr (kHasShared) {
            const auto blocks_s = math::ceil_div(num_tokens, BLOCK_M);
            for (uint32_t p = 0; p < kNumPasses; ++ p) {
                bw.expert_pool_base[kNumExpertsPerRank + p] = pool_base + p * bw.shared_region_stride;
                bw.expert_num_blocks[kNumExpertsPerRank + p] = blocks_s;
                bw.expert_done[kNumExpertsPerRank + p] = 0;
                bw.expert_a2_target[kNumExpertsPerRank + p] = blocks_s * kNumPasses * kNumG2Tiles;
            }
            for (uint32_t off = 0; off < num_tokens; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = kNumExpertsPerRank;
                bw.block_desc[num_blocks].pool_begin = pool_base + off;
                bw.block_desc[num_blocks].meta_begin = off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, num_tokens - off);
                ++ num_blocks;
            }
            pool_base += kNumPasses * bw.shared_region_stride;
        }
        DG_DEVICE_ASSERT(num_blocks <= bw.max_num_blocks);
        DG_DEVICE_ASSERT(pool_base <= bw.num_pool_rows);
        *bw.num_blocks = num_blocks;
        *bw.num_routed_blocks = num_routed_blocks;
        uint32_t num_items = kNumExpertSlots * kNumTilesPerExpert;
        for (uint32_t b0 = 0; b0 < num_blocks; b0 += kGroupBlocks) {
            const uint32_t nb = cute::min(kGroupBlocks, num_blocks - b0);
            const uint32_t r = num_routed_blocks > b0 ? cute::min(nb, num_routed_blocks - b0) : 0u;
            num_items += nb * kNumGatherChunks + (r + (nb - r) * kNumPasses) * (kNumG1Tiles + kNumG2Tiles) + nb * kNumHTiles;
        }
        *bw.num_items = num_items;
        *bw.next_item = 0;
    }
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    __shared__ uint32_t s_pool_begin[kNumExpertsPerRank];
    __shared__ uint32_t s_rank_prefix[kNumExpertsPerRank][kNumRanks];
    if (tid == 0) {
        uint32_t pool_begin = 0;
        for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
            s_pool_begin[e] = pool_begin;
            uint32_t rank_prefix = 0;
            for (uint32_t r = 0; r < kNumRanks; ++ r) {
                s_rank_prefix[e][r] = rank_prefix;
                rank_prefix += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, e));
            }
            pool_begin += rank_prefix;
        }
    }
    __syncthreads();
    for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
        const auto valid_m_e = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
        const auto pool_begin_e = s_pool_begin[e];
        for (uint32_t local_pos = global_tid; local_pos < valid_m_e; local_pos += kNumGlobalThreads) {
            uint32_t r = 0;
            #pragma unroll
            for (uint32_t rr = 0; rr < kNumRanks; ++ rr)
                if (s_rank_prefix[e][rr] <= local_pos) r = rr;
            const auto slot = local_pos - s_rank_prefix[e][r];
            const auto token_topk = *workspace.get_src_token_topk_idx_ptr(e, r, slot);
            const auto src_token = token_topk / kNumTopk;
            const auto src_topk = token_topk % kNumTopk;
            *workspace.get_token_src_metadata_ptr(pool_begin_e + local_pos) =
                layout::TokenSrcMetadata(r, src_token, src_topk);
            bw.meta_weight[pool_begin_e + local_pos] = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token * kNumTopk + src_topk, r);
        }
    }
    const uint32_t total_blocks = *bw.num_blocks;
    for (uint32_t i = global_tid; i < total_blocks; i += kNumGlobalThreads) {
        bw.block_a0_done[i] = 0;
        bw.block_a1_done[i] = 0;
        bw.block_a2_done[i] = 0;
    }
    for (uint32_t i = global_tid; i < total_blocks * BLOCK_M; i += kNumGlobalThreads)
        bw.block_dtopk[i] = 0.0f;
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });

    struct Item {
        uint32_t kind;
        uint32_t block;
        uint32_t expert;
        uint32_t pool_begin;
        uint32_t x_begin;
        uint32_t valid_m;
        uint32_t tile;
        uint32_t pass;
        uint32_t num_k_blocks;
        uint32_t zslot;
    };
    struct SharedStorage {
        alignas(1024) bf16_t a[kNumStages][MMA_M * BLOCK_K];
        alignas(1024) bf16_t b[kNumStages][MMA_N * BLOCK_K];
        alignas(1024) bf16_t dx_stage[kNumConsumerWarpgroups][kNumMHalves][DX_STAGE_TOKENS * WGMMA_M];
        float route_weight[kNumItemSlots][BLOCK_M];
        uint32_t src_rank[kNumItemSlots][BLOCK_M];
        uint32_t src_token[kNumItemSlots][BLOCK_M];
        uint32_t src_topk[kNumItemSlots][BLOCK_M];
        float dtopk_partial[kNumConsumerWarpgroups][BLOCK_M];
        Item item[kNumItemSlots];
        uint32_t item_valid[kNumItemSlots];
        uint32_t g_block[kNumGatherWarps], g_pool_begin[kNumGatherWarps], g_valid_m[kNumGatherWarps];
        uint32_t g_row_begin[kNumGatherWarps], g_valid[kNumGatherWarps];
        uint32_t g_src_rank[kNumGatherWarps][CHUNK];
        uint32_t g_src_token[kNumGatherWarps][CHUNK];
        Barrier gfull[kNumGatherWarps];
        Barrier gempty[kNumGatherWarps];
        Barrier slot_full[kNumItemSlots];
        Barrier slot_empty[kNumItemSlots];
        Barrier full[kNumStages];
        Barrier empty[kNumStages];
    };
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    constexpr uint32_t kStageABytes = sizeof(smem.a[0]);
    constexpr uint32_t kStageBBytes = sizeof(smem.b[0]);
    constexpr uint32_t kDxHalfBytes = sizeof(smem.dx_stage[0][0]);
    static_assert(kDxHalfBytes % 1024 == 0, "Invalid dx stage alignment");

    if (warp == 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumItemSlots; ++ i) {
                smem.slot_full[i].init(1);
                smem.slot_empty[i].init(1 + kNumConsumerWarpgroups);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumGatherWarps; ++ i) {
                smem.gfull[i].init(1);
                smem.gempty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                smem.full[i].init(1);
                smem.empty[i].init(4);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const auto pg = [](const uint32_t& j) -> uint32_t {
        return (j / kGran) * 2 * kGran + (j % kGran);
    };
    const auto z_slot_ptr = [&](const uint32_t& zslot, const uint32_t& pass) -> float* {
        return z_scratch + (static_cast<uint64_t>(zslot) * kNumPasses + pass) * I2 * BLOCK_M;
    };
    const auto block_passes = [&](const uint32_t& block) -> uint32_t {
        return block >= *bw.num_routed_blocks ? kNumPasses : 1u;
    };
    const auto gather_rows = [&](const uint32_t& gslot, const uint32_t& pool_begin, const uint32_t& valid_m, const uint32_t& row_begin) {
        for (uint32_t rr = 0; rr < CHUNK; rr += 2) {
            const uint32_t rows[2] = {row_begin + rr, row_begin + rr + 1};
            uint4* dst_x[2];
            uint4* dst_dy[2];
            const uint4* src_x[2];
            const uint4* src_dy[2];
            bool valid[2];
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++ h) {
                dst_x[h] = reinterpret_cast<uint4*>(x_pool + static_cast<uint64_t>(pool_begin + rows[h]) * kHidden);
                dst_dy[h] = reinterpret_cast<uint4*>(dy_pool + static_cast<uint64_t>(pool_begin + rows[h]) * kHidden);
                valid[h] = rows[h] < valid_m;
                const auto r_rank = smem.g_src_rank[gslot][rr + h];
                const auto r_token = smem.g_src_token[gslot][rr + h];
                src_x[h] = sym_buffer.map(reinterpret_cast<const uint4*>(
                    buffer.input_token_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
                src_dy[h] = sym_buffer.map(reinterpret_cast<const uint4*>(
                    bw.input_dy_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
            }
            for (uint32_t base = lane; base < kNumVecPerRow; base += 32 * kVecUnroll) {
                uint4 vx[2][kVecUnroll], vy[2][kVecUnroll];
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecUnroll; ++ i) {
                        vx[h][i] = valid[h] ? src_x[h][base + (i << 5)] : make_uint4(0, 0, 0, 0);
                        vy[h][i] = valid[h] ? src_dy[h][base + (i << 5)] : make_uint4(0, 0, 0, 0);
                    }
                }
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecUnroll; ++ i) {
                        dst_x[h][base + (i << 5)] = vx[h][i];
                        dst_dy[h][base + (i << 5)] = vy[h][i];
                    }
                }
            }
        }
        fence_proxy_async_global();
    };
    const auto wait_counter = [&](const uint32_t* counter, const uint32_t& target) {
        while (ptx::ld_acq(counter) < target)
            __nanosleep(128);
    };

    if (warp == kSchedulerWarpIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        uint32_t slot = 0, slot_phase = 0;
        uint32_t gslot = 0, gphase_bits = 0;
        const auto gphase = [&](const uint32_t& g) -> uint32_t { return (gphase_bits >> g) & 1u; };
        for (;;) {
            uint32_t item_idx = 0;
            if (lane == 0)
                item_idx = ptx::atomic_add(bw.next_item, 1u);
            item_idx = __shfl_sync(0xffffffff, item_idx, 0);
            const uint32_t num_blocks = *bw.num_blocks;
            const uint32_t num_routed_blocks = *bw.num_routed_blocks;
            const uint32_t num_block_items = *bw.num_items - kNumExpertSlots * kNumTilesPerExpert;
            const bool done = item_idx >= *bw.num_items;
            Item cur{};
            if (!done) {
                if (item_idx < num_block_items) {
                    const auto group_of = [&](const uint32_t& g, uint32_t& b0_, uint32_t& nb_, uint32_t& r_) {
                        b0_ = g * kGroupBlocks;
                        nb_ = b0_ < num_blocks ? cute::min(kGroupBlocks, num_blocks - b0_) : 0u;
                        r_ = num_routed_blocks > b0_ ? cute::min(nb_, num_routed_blocks - b0_) : 0u;
                    };
                    const auto gather_count = [&](const uint32_t& nb_) { return nb_ * kNumGatherChunks; };
                    const auto compute_count = [&](const uint32_t& nb_, const uint32_t& r_) { return (r_ + (nb_ - r_) * kNumPasses) * (kNumG1Tiles + kNumG2Tiles) + nb_ * kNumHTiles; };
                    uint32_t rem = item_idx, b0 = 0, nb = 0, r = 0;
                    bool is_gather = false;
                    group_of(0, b0, nb, r);
                    if (rem < gather_count(nb)) {
                        is_gather = true;
                    } else {
                        rem -= gather_count(nb);
                        for (uint32_t g = 0;; ++ g) {
                            uint32_t b1, nb1, r1;
                            group_of(g + 1, b1, nb1, r1);
                            if (rem < gather_count(nb1)) {
                                is_gather = true;
                                b0 = b1, nb = nb1, r = r1;
                                break;
                            }
                            rem -= gather_count(nb1);
                            group_of(g, b0, nb, r);
                            if (rem < compute_count(nb, r))
                                break;
                            rem -= compute_count(nb, r);
                        }
                    }
                    const auto decode_tiles = [&](uint32_t idx, const uint32_t& tiles, uint32_t& block, uint32_t& pass, uint32_t& tile) {
                        if (idx < r * tiles) {
                            block = b0 + idx / tiles;
                            pass = 0;
                            tile = idx % tiles;
                        } else {
                            idx -= r * tiles;
                            block = b0 + r + idx / (kNumPasses * tiles);
                            pass = (idx % (kNumPasses * tiles)) / tiles;
                            tile = idx % tiles;
                        }
                    };
                    if (is_gather) {
                        cur.kind = kKindGather;
                        cur.block = b0 + rem / kNumGatherChunks;
                        cur.tile = rem % kNumGatherChunks;
                    } else if (rem < (r + (nb - r) * kNumPasses) * kNumG1Tiles) {
                        cur.kind = kKindZ;
                        decode_tiles(rem, kNumG1Tiles, cur.block, cur.pass, cur.tile);
                        cur.num_k_blocks = kNumKBlocksH;
                    } else if ((rem -= (r + (nb - r) * kNumPasses) * kNumG1Tiles) < (r + (nb - r) * kNumPasses) * kNumG2Tiles) {
                        cur.kind = kKindDz;
                        decode_tiles(rem, kNumG2Tiles, cur.block, cur.pass, cur.tile);
                        cur.num_k_blocks = kNumKBlocksH;
                    } else {
                        rem -= (r + (nb - r) * kNumPasses) * kNumG2Tiles;
                        cur.kind = kKindDx;
                        cur.block = b0 + rem / kNumHTiles;
                        cur.tile = rem % kNumHTiles;
                        cur.num_k_blocks = block_passes(cur.block) * kNumKBlocksI2;
                    }
                    const uint32_t num_passes = block_passes(cur.block);
                    const auto bd = bw.block_desc[cur.block];
                    cur.expert = bd.local_expert;
                    cur.pool_begin = bd.pool_begin;
                    cur.x_begin = bd.pool_begin;
                    cur.valid_m = bd.valid_m;
                    cur.zslot = cur.block % kNumZSlots;
                    if (cur.kind == kKindGather) {
                        smem.gempty[gslot].wait(gphase(gslot) ^ 1);
                        const bool is_shared = cur.expert >= kNumExpertsPerRank;
                        const uint32_t r_idx = cur.tile * CHUNK + lane;
                        uint32_t src_rank = sym_buffer.rank_idx, src_token = 0;
                        if (r_idx < bd.valid_m) {
                            if (is_shared) {
                                src_token = bd.meta_begin + r_idx;
                            } else {
                                const auto meta = *workspace.get_token_src_metadata_ptr(bd.meta_begin + r_idx);
                                src_rank = meta.rank_idx;
                                src_token = meta.token_idx;
                            }
                        }
                        smem.g_src_rank[gslot][lane] = src_rank;
                        smem.g_src_token[gslot][lane] = src_token;
                        if (lane == 0) {
                            smem.g_block[gslot] = cur.block;
                            smem.g_pool_begin[gslot] = cur.pool_begin;
                            smem.g_valid_m[gslot] = cur.valid_m;
                            smem.g_row_begin[gslot] = cur.tile * CHUNK;
                            smem.g_valid[gslot] = 1;
                        }
                        __syncwarp();
                        if (lane == 0)
                            smem.gfull[gslot].arrive();
                        gphase_bits ^= 1u << gslot;
                        gslot ^= 1;
                        continue;
                    }
                    if (lane == 0) {
                        if (cur.kind == kKindZ) {
                            wait_counter(bw.block_a0_done + cur.block, kNumGatherChunks);
                            if (cur.block >= kNumZSlots)
                                wait_counter(bw.block_a2_done + cur.block - kNumZSlots, block_passes(cur.block - kNumZSlots) * kNumG2Tiles);
                        } else if (cur.kind == kKindDz) {
                            wait_counter(bw.block_a1_done + cur.block, num_passes * kNumG1Tiles);
                        } else if (cur.kind == kKindDx) {
                            wait_counter(bw.block_a2_done + cur.block, num_passes * kNumG2Tiles);
                        }
                    }
                } else {
                    const uint32_t t = item_idx - num_block_items;
                    cur.kind = kKindDw;
                    cur.expert = t / kNumTilesPerExpert;
                    cur.tile = t % kNumTilesPerExpert;
                    cur.num_k_blocks = bw.expert_num_blocks[cur.expert] * (BLOCK_M / BLOCK_K);
                    cur.pool_begin = bw.expert_pool_base[cur.expert];
                    cur.x_begin = bw.expert_pool_base[cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank];
                    const uint32_t done_slot = cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank;
                    if (lane == 0)
                        wait_counter(bw.expert_done + done_slot, bw.expert_a2_target[done_slot]);
                }
                __syncwarp();
                fence_proxy_async_global();
            }
            smem.slot_empty[slot].wait(slot_phase ^ 1);
            if (!done && cur.kind != kKindDw) {
                const auto bd = bw.block_desc[cur.block];
                const bool is_shared = cur.expert >= kNumExpertsPerRank;
                for (uint32_t r_idx = lane; r_idx < BLOCK_M; r_idx += 32) {
                    uint32_t src_rank = sym_buffer.rank_idx, src_token = 0, src_topk = kNumTopk;
                    float weight = 0.0f;
                    if (r_idx < bd.valid_m) {
                        if (is_shared) {
                            src_token = bd.meta_begin + r_idx;
                            weight = 1.0f;
                        } else {
                            const auto meta = *workspace.get_token_src_metadata_ptr(bd.meta_begin + r_idx);
                            src_rank = meta.rank_idx;
                            src_token = meta.token_idx;
                            src_topk = meta.topk_idx;
                            weight = bw.meta_weight[bd.meta_begin + r_idx];
                        }
                    }
                    smem.route_weight[slot][r_idx] = weight;
                    smem.src_rank[slot][r_idx] = src_rank;
                    smem.src_token[slot][r_idx] = src_token;
                    smem.src_topk[slot][r_idx] = src_topk;
                }
            }
            if (lane == 0) {
                smem.item[slot] = cur;
                smem.item_valid[slot] = done ? 0u : 1u;
            }
            __syncwarp();
            if (lane == 0)
                smem.slot_full[slot].arrive();
            if (done) {
                #pragma unroll
                for (uint32_t g = 0; g < kNumGatherWarps; ++ g) {
                    smem.gempty[g].wait(gphase(g) ^ 1);
                    if (lane == 0)
                        smem.g_valid[g] = 0;
                    __syncwarp();
                    if (lane == 0)
                        smem.gfull[g].arrive();
                }
                break;
            }
            slot = slot == kNumItemSlots - 1 ? 0 : slot + 1;
            slot_phase ^= (slot == 0);
        }
    } else if (warp >= kGatherWarpStartIdx && warp < kGatherWarpStartIdx + kNumGatherWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        const uint32_t gslot = warp - kGatherWarpStartIdx;
        uint32_t gphase = 0;
        for (;;) {
            smem.gfull[gslot].wait(gphase);
            if (!smem.g_valid[gslot])
                break;
            gather_rows(gslot, smem.g_pool_begin[gslot], smem.g_valid_m[gslot], smem.g_row_begin[gslot]);
            __threadfence();
            __syncwarp();
            if (lane == 0) {
                ptx::atomic_add_rel(bw.block_a0_done + smem.g_block[gslot], 1u);
                smem.gempty[gslot].arrive();
            }
            gphase ^= 1;
        }
    } else if (warp == kTMAWarpIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase ^= (stage == 0);
        };
        const auto issue = [&](const uint32_t& num_bytes) {
            smem.full[stage].arrive_and_expect_tx(num_bytes);
            advance();
        };
        constexpr uint32_t kStageBytes = kStageABytes + kStageBBytes;
        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.item_valid[slot])
                break;
            const auto item = smem.item[slot];
            __syncwarp();
            if (lane == 0)
                smem.slot_empty[slot].arrive();

            const bool is_shared = item.expert >= kNumExpertsPerRank;
            const auto* w1k = is_shared ? &tensor_map_shared_w1_k : &tensor_map_w1_k;
            const auto* w1mn = is_shared ? &tensor_map_shared_w1_mn : &tensor_map_w1_mn;
            const auto* w2mn = is_shared ? &tensor_map_shared_w2_mn : &tensor_map_w2_mn;
            if (item.kind == kKindZ) {
                const uint32_t w1_rows = is_shared ? item.pass * I2 : item.expert * I2;
                for (uint32_t kb = 0; kb < kNumKBlocksH; ++ kb) {
                    smem.empty[stage].wait(phase ^ 1);
                    if (cute::elect_one_sync()) {
                        if constexpr (kL1Natural) {
                            // All shared experts form one `[gate | up]` matrix of `I2 * kNumPasses` rows
                            if (is_shared)
                                tma::copy_gate_up_natural<BLOCK_K, MMA_M, kSwizzleMode, I2 * kNumPasses, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb * BLOCK_K, w1_rows + item.tile * MMA_M);
                            else
                                tma::copy_gate_up_natural<BLOCK_K, MMA_M, kSwizzleMode, I2, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb * BLOCK_K, w1_rows + item.tile * MMA_M);
                        } else
                            tma::copy<BLOCK_K, MMA_M, kSwizzleMode, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb * BLOCK_K, w1_rows + item.tile * MMA_M);
                        tma::copy<BLOCK_K, MMA_N, kSwizzleMode, bf16_t>(&tensor_map_x_k, &smem.full[stage], smem.b[stage], kb * BLOCK_K, item.pool_begin);
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDz) {
                const uint32_t w2_rows = is_shared ? 0u : item.expert * kHidden;
                const uint32_t w2_cols = is_shared ? item.pass * kIntermediateHidden : 0u;
                for (uint32_t kb = 0; kb < kNumKBlocksH; ++ kb) {
                    smem.empty[stage].wait(phase ^ 1);
                    if (cute::elect_one_sync()) {
                        tma::copy<MMA_M, BLOCK_K, kSwizzleMode, bf16_t>(w2mn, &smem.full[stage], smem.a[stage], w2_cols + item.tile * MMA_M, w2_rows + kb * BLOCK_K);
                        tma::copy<BLOCK_K, MMA_N, kSwizzleMode, bf16_t>(&tensor_map_dy_k, &smem.full[stage], smem.b[stage], kb * BLOCK_K, item.pool_begin);
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDx) {
                const uint32_t w1_rows = is_shared ? 0u : item.expert * I2;
                for (uint32_t kb = 0; kb < item.num_k_blocks; ++ kb) {
                    smem.empty[stage].wait(phase ^ 1);
                    if (cute::elect_one_sync()) {
                        if constexpr (kL1Natural) {
                            // All shared experts form one `[gate | up]` matrix of `I2 * kNumPasses` rows
                            if (is_shared)
                                tma::copy_gate_up_natural<MMA_M, BLOCK_K, kSwizzleMode, I2 * kNumPasses, bf16_t>(w1mn, &smem.full[stage], smem.a[stage], item.tile * MMA_M, w1_rows + kb * BLOCK_K);
                            else
                                tma::copy_gate_up_natural<MMA_M, BLOCK_K, kSwizzleMode, I2, bf16_t>(w1mn, &smem.full[stage], smem.a[stage], item.tile * MMA_M, w1_rows + kb * BLOCK_K);
                        } else
                            tma::copy<MMA_M, BLOCK_K, kSwizzleMode, bf16_t>(w1mn, &smem.full[stage], smem.a[stage], item.tile * MMA_M, w1_rows + kb * BLOCK_K);
                        tma::copy<MMA_N, BLOCK_K, kSwizzleMode, bf16_t>(&tensor_map_dz_mn, &smem.full[stage], smem.b[stage],
                                                                        item.pool_begin + (kb / kNumKBlocksI2) * bw.shared_region_stride, (kb % kNumKBlocksI2) * BLOCK_K);
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDw) {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                const uint32_t mt = is_dw2 ? item.tile / kNumG2Tiles : (item.tile - kNumDW2Tiles) / kNumHTiles;
                const uint32_t nt = is_dw2 ? item.tile % kNumG2Tiles : (item.tile - kNumDW2Tiles) % kNumHTiles;
                for (uint32_t kb = 0; kb < item.num_k_blocks; ++ kb) {
                    smem.empty[stage].wait(phase ^ 1);
                    if (cute::elect_one_sync()) {
                        if (is_dw2) {
                            tma::copy<MMA_M, BLOCK_K, kSwizzleMode, bf16_t>(&tensor_map_dy_mn, &smem.full[stage], smem.a[stage], mt * MMA_M, item.x_begin + kb * BLOCK_K);
                            tma::copy<BLOCK_K, MMA_N, kSwizzleMode, bf16_t>(&tensor_map_hw_k, &smem.full[stage], smem.b[stage], item.pool_begin + kb * BLOCK_K, nt * MMA_N);
                        } else {
                            tma::copy<BLOCK_K, MMA_M, kSwizzleMode, bf16_t>(&tensor_map_dz_k, &smem.full[stage], smem.a[stage], item.pool_begin + kb * BLOCK_K, mt * MMA_M);
                            tma::copy<MMA_N, BLOCK_K, kSwizzleMode, bf16_t>(&tensor_map_x_mn, &smem.full[stage], smem.b[stage], nt * MMA_N, item.x_begin + kb * BLOCK_K);
                        }
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            }
            slot = slot == kNumItemSlots - 1 ? 0 : slot + 1;
            slot_phase ^= (slot == 0);
        }
    } else if (warp >= 4) {
        cutlass::arch::warpgroup_reg_alloc<kNumConsumerRegisters>();
        const uint32_t consumer_tid = tid - kNumProducerThreads;
        const uint32_t consumer_wg_idx = __shfl_sync(0xffffffff, consumer_tid / kNumConsumerThreadsPerWG, 0);
        const uint32_t tid_in_wg = consumer_tid % kNumConsumerThreadsPerWG;
        const uint32_t warp_in_wg = tid_in_wg / 32;
        const uint32_t frag_row = warp_in_wg * 16 + lane / 4;      // Row within an M half (+ 8 for the second half)
        const uint32_t frag_col = (lane % 4) * 2;                   // Token column within an 8-token group
        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t item_seq = 0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase ^= (stage == 0);
        };
        const auto a_k = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, MMA_M, BLOCK_K, kSwizzleMode>(smem.a[0], 0, 0);
        const auto a_mn = mma::sm90::make_gmma_desc<cute::UMMA::Major::MN, MMA_M, BLOCK_K, kSwizzleMode>(smem.a[0], 0, 0);
        const auto b_k = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, MMA_N, BLOCK_K, kSwizzleMode>(smem.b[0], 0, 0);
        const auto b_mn = mma::sm90::make_gmma_desc<cute::UMMA::Major::MN, MMA_N, BLOCK_K, kSwizzleMode>(smem.b[0], 0, 0);
        const uint32_t a_k_lo = __shfl_sync(0xffffffff, a_k.reg32_[0], 0);
        const uint32_t a_mn_lo = __shfl_sync(0xffffffff, a_mn.reg32_[0], 0);
        const uint32_t b_k_lo = __shfl_sync(0xffffffff, b_k.reg32_[0], 0);
        const uint32_t b_mn_lo = __shfl_sync(0xffffffff, b_mn.reg32_[0], 0);

        float accum[kNumMHalves][kNumAccumPerHalf];
        const auto run_mainloop = [&]<cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB>(
                std::integral_constant<cute::UMMA::Major, kMajorA>, std::integral_constant<cute::UMMA::Major, kMajorB>,
                const uint32_t& num_k_blocks) {
            using WGMMA = typename mma::sm90::BF16MMASelector<MMA_N, kMajorA, kMajorB>::type;
            static_assert(WGMMA::kNumAccum == kNumAccumPerHalf && WGMMA::M == WGMMA_M && WGMMA::K == WGMMA_K, "Invalid WGMMA");
            auto a_desc = kMajorA == cute::UMMA::Major::K ? a_k : a_mn;
            auto b_desc = kMajorB == cute::UMMA::Major::K ? b_k : b_mn;
            const uint32_t a_lo = kMajorA == cute::UMMA::Major::K ? a_k_lo : a_mn_lo;
            const uint32_t b_lo = kMajorB == cute::UMMA::Major::K ? b_k_lo : b_mn_lo;

            #pragma unroll
            for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccumPerHalf; ++ i)
                    accum[mh][i] = 0.0f;
            }
            for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                const uint32_t a_base = a_lo + stage * (kStageABytes / 16);
                const uint32_t b_base = b_lo + stage * (kStageBBytes / 16);
                smem.full[stage].wait(phase);

                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccumPerHalf; ++ i)
                        ptx::warpgroup_fence_operand(accum[mh][i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++ k) {
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorB, MMA_N, BLOCK_K, kSwizzleMode, nv_bfloat16>(b_base, 0, k * WGMMA_K);
                    #pragma unroll
                    for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, MMA_M, BLOCK_K, kSwizzleMode, nv_bfloat16>(a_base, mh * WGMMA_M, k * WGMMA_K);
                        WGMMA::wgmma(a_desc, b_desc, accum[mh], 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccumPerHalf; ++ i)
                        ptx::warpgroup_fence_operand(accum[mh][i]);
                }
                ptx::warpgroup_wait<0>();
                if (lane == 0)
                    smem.empty[stage].arrive();
                __syncwarp();
                advance();
            }
        };
        constexpr auto kMajorK = std::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>{};
        constexpr auto kMajorMN = std::integral_constant<cute::UMMA::Major, cute::UMMA::Major::MN>{};

        const auto epi_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumConsumerThreadsPerWG, kConsumerBarrierStartIdx + consumer_wg_idx); };
        const auto clamp_gate = [](float g) {
            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                g = cute::min(g, kActivationClamp);
            return g;
        };
        const auto clamp_up = [](float u) {
            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                u = cute::max(cute::min(u, kActivationClamp), -kActivationClamp);
            return u;
        };
        const auto finish_item = [&](const uint32_t& slot) {
            epi_sync();
            if (tid_in_wg == 0)
                smem.slot_empty[slot].arrive();
        };
        const auto load_f32x2 = [](const float* ptr) -> float2 { return __ldcg(reinterpret_cast<const float2*>(ptr)); };

        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.item_valid[slot]) break;
            const auto item = smem.item[slot];
            const bool is_owner = (item_seq % kNumConsumerWarpgroups) == consumer_wg_idx;
            ++ item_seq;
            if (!is_owner) {
                for (uint32_t kb = 0; kb < item.num_k_blocks; ++ kb)
                    advance();
                finish_item(slot);
                slot = slot == kNumItemSlots - 1 ? 0 : slot + 1;
                slot_phase ^= (slot == 0);
                continue;
            }

            const bool is_shared = item.expert >= kNumExpertsPerRank;
            if (item.kind == kKindZ) {
                run_mainloop(kMajorK, kMajorK, kNumKBlocksH);
                auto* z_slot = z_slot_ptr(item.zslot, item.pass);
                const uint32_t pool_row = item.pool_begin + item.pass * bw.shared_region_stride;
                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    const uint32_t prow_gate = item.tile * MMA_M + mh * WGMMA_M + frag_row;
                    const uint32_t prow_up = prow_gate + kGran;
                    const uint32_t j = (prow_gate / (2 * kGran)) * kGran + (prow_gate % kGran);
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumTokenGroups; ++ i) {
                        const uint32_t t = i * 8 + frag_col;
                        const float2 g = make_float2(accum[mh][i * 4 + 0], accum[mh][i * 4 + 1]);
                        const float2 u = make_float2(accum[mh][i * 4 + 2], accum[mh][i * 4 + 3]);
                        *reinterpret_cast<float2*>(z_slot + static_cast<uint64_t>(prow_gate) * BLOCK_M + t) = g;
                        *reinterpret_cast<float2*>(z_slot + static_cast<uint64_t>(prow_up) * BLOCK_M + t) = u;
                        const float2 rw = *reinterpret_cast<const float2*>(&smem.route_weight[slot][t]);
                        const float gc0 = clamp_gate(g.x), gc1 = clamp_gate(g.y);
                        const float hw0 = gc0 * sigmoid<kFastMath>(gc0) * clamp_up(u.x) * rw.x;
                        const float hw1 = gc1 * sigmoid<kFastMath>(gc1) * clamp_up(u.y) * rw.y;
                        *reinterpret_cast<uint32_t*>(hw_pool + static_cast<uint64_t>(j) * pool_stride + pool_row + t) = pack_bf16x2(hw0, hw1);
                    }
                }
                fence_proxy_async_global();
                epi_sync();
                if (tid_in_wg == 0) {
                    __threadfence();
                    ptx::atomic_add_rel(bw.block_a1_done + item.block, 1u);
                    smem.slot_empty[slot].arrive();
                }
            } else if (item.kind == kKindDz) {
                run_mainloop(kMajorMN, kMajorK, kNumKBlocksH);

                const auto* z_slot = z_slot_ptr(item.zslot, item.pass);
                const uint32_t pool_row = item.pool_begin + item.pass * bw.shared_region_stride;
                smem.dtopk_partial[consumer_wg_idx][tid_in_wg] = 0.0f;
                epi_sync();

                #pragma unroll
                for (uint32_t i = 0; i < kNumTokenGroups; ++ i) {
                    const uint32_t t = i * 8 + frag_col;
                    const float2 rw = *reinterpret_cast<const float2*>(&smem.route_weight[slot][t]);
                    float2 partial = make_float2(0.0f, 0.0f);
                    #pragma unroll
                    for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                        #pragma unroll
                        for (uint32_t half = 0; half < 2; ++ half) {
                            const uint32_t irow = item.tile * MMA_M + mh * WGMMA_M + frag_row + half * 8;
                            const uint32_t pgi = pg(irow);
                            const float2 dh = make_float2(accum[mh][i * 4 + half * 2], accum[mh][i * 4 + half * 2 + 1]);
                            const float2 g = load_f32x2(z_slot + static_cast<uint64_t>(pgi) * BLOCK_M + t);
                            const float2 u = load_f32x2(z_slot + static_cast<uint64_t>(pgi + kGran) * BLOCK_M + t);
                            float dz_gate[2], dz_up[2];
                            #pragma unroll
                            for (uint32_t e = 0; e < 2; ++ e) {
                                const float ge = e == 0 ? g.x : g.y, ue = e == 0 ? u.x : u.y;
                                const float dhe = e == 0 ? dh.x : dh.y, rwe = e == 0 ? rw.x : rw.y;
                                bool gate_active = true, up_active = true;
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                    gate_active = ge <= kActivationClamp;
                                    up_active = ue >= -kActivationClamp && ue <= kActivationClamp;
                                }
                                const float gc = clamp_gate(ge), uc = clamp_up(ue);
                                const float sig = sigmoid<kFastMath>(gc);
                                const float silu = gc * sig;
                                const float dsilu = sig * (1.0f + gc * (1.0f - sig));
                                (e == 0 ? partial.x : partial.y) += silu * uc * dhe;
                                const float dhw = dhe * rwe;
                                dz_gate[e] = gate_active ? dhw * uc * dsilu : 0.0f;
                                dz_up[e] = up_active ? dhw * silu : 0.0f;
                            }
                            *reinterpret_cast<uint32_t*>(dz_pool + static_cast<uint64_t>(pgi) * pool_stride + pool_row + t) = pack_bf16x2(dz_gate[0], dz_gate[1]);
                            *reinterpret_cast<uint32_t*>(dz_pool + static_cast<uint64_t>(pgi + kGran) * pool_stride + pool_row + t) = pack_bf16x2(dz_up[0], dz_up[1]);
                        }
                    }
                    partial.x = math::warp_reduce_sum<4, true>(partial.x);
                    partial.y = math::warp_reduce_sum<4, true>(partial.y);
                    if (lane < 4 && !is_shared) {
                        atomicAdd(&smem.dtopk_partial[consumer_wg_idx][t + 0], partial.x);
                        atomicAdd(&smem.dtopk_partial[consumer_wg_idx][t + 1], partial.y);
                    }
                }
                epi_sync();
                if (!is_shared)
                    atomicAdd(bw.block_dtopk + static_cast<uint64_t>(item.block) * BLOCK_M + tid_in_wg, smem.dtopk_partial[consumer_wg_idx][tid_in_wg]);
                fence_proxy_async_global();
                epi_sync();
                if (tid_in_wg == 0) {
                    __threadfence();
                    ptx::atomic_add_rel(bw.block_a2_done + item.block, 1u);
                    ptx::atomic_add_rel(bw.expert_done + (is_shared ? kNumExpertsPerRank : item.expert), 1u);
                    smem.slot_empty[slot].arrive();
                }
            } else if (item.kind == kKindDx) {
                run_mainloop(kMajorMN, kMajorMN, item.num_k_blocks);
                auto* stage_tile = reinterpret_cast<uint8_t*>(smem.dx_stage[consumer_wg_idx]);
                #pragma unroll
                for (uint32_t c = 0; c < kNumDxChunks; ++ c) {
                    if (c > 0)
                        epi_sync();
                    const uint32_t i = c * 2;
                    const uint32_t m = lane / 8;
                    const uint32_t row = (m / 2) * 8 + (lane % 8);
                    const uint32_t col_chunk = warp_in_wg * 2 + (m % 2);
                    #pragma unroll
                    for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                        const uint32_t byte_offset = mh * kDxHalfBytes + math::swizzle_byte_offset<kSwizzleMode>(
                            row * kSwizzleMode + col_chunk * kNumBankGroupBytes);
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            pack_bf16x2(accum[mh][i * 4 + 0], accum[mh][i * 4 + 1]),
                            pack_bf16x2(accum[mh][i * 4 + 2], accum[mh][i * 4 + 3]),
                            pack_bf16x2(accum[mh][i * 4 + 4], accum[mh][i * 4 + 5]),
                            pack_bf16x2(accum[mh][i * 4 + 6], accum[mh][i * 4 + 7]),
                            stage_tile + byte_offset
                        );
                    }
                    epi_sync();
                    #pragma unroll
                    for (uint32_t p = 0; p < DX_STAGE_TOKENS * (MMA_M / 8) / kNumConsumerThreadsPerWG; ++ p) {
                        const uint32_t idx = p * kNumConsumerThreadsPerWG + tid_in_wg;
                        const uint32_t tok_in_chunk = idx / (MMA_M / 8);
                        const uint32_t chunk16 = idx % (MMA_M / 8);
                        const uint32_t tok = c * DX_STAGE_TOKENS + tok_in_chunk;
                        if (tok < item.valid_m) {
                            const uint32_t mh = chunk16 / 8, cc = chunk16 % 8;
                            const uint32_t byte_offset = mh * kDxHalfBytes + math::swizzle_byte_offset<kSwizzleMode>(
                                tok_in_chunk * kSwizzleMode + cc * kNumBankGroupBytes);
                            const auto value = ptx::ld_shared(reinterpret_cast<const uint4*>(stage_tile + byte_offset));
                            auto* remote_dx = reinterpret_cast<uint4*>(sym_buffer.map(
                                bw.dx_slot_buffer.get_rank_buffer(smem.src_topk[slot][tok])
                                    .get_data_buffer(smem.src_token[slot][tok]).template get_base_ptr<nv_bfloat16>(),
                                smem.src_rank[slot][tok]) + item.tile * MMA_M + chunk16 * 8);
                            *remote_dx = value;
                        }
                    }
                }
                if (item.tile == 0 && !is_shared && tid_in_wg < item.valid_m) {
                    auto* remote_dw = sym_buffer.map(
                        bw.dtopk_weight_slot_buffer.get_rank_buffer(smem.src_topk[slot][tid_in_wg])
                            .get_data_buffer(smem.src_token[slot][tid_in_wg]).template get_base_ptr<float>(),
                        smem.src_rank[slot][tid_in_wg]);
                    *remote_dw = __ldcg(bw.block_dtopk + static_cast<uint64_t>(item.block) * BLOCK_M + tid_in_wg);
                }
                finish_item(slot);
            } else {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                if (is_dw2)
                    run_mainloop(kMajorMN, kMajorK, item.num_k_blocks);
                else
                    run_mainloop(kMajorK, kMajorMN, item.num_k_blocks);

                const uint32_t mt = is_dw2 ? item.tile / kNumG2Tiles : (item.tile - kNumDW2Tiles) / kNumHTiles;
                const uint32_t nt = is_dw2 ? item.tile % kNumG2Tiles : (item.tile - kNumDW2Tiles) % kNumHTiles;
                const uint32_t p = is_shared ? item.expert - kNumExpertsPerRank : 0u;
                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++ half) {
                        const uint32_t m = mt * MMA_M + mh * WGMMA_M + frag_row + half * 8;
                        // Natural dW1 rows are `[gate | up]`; all shared experts form one matrix of `kIntermediateHidden * kNumPasses` gate rows
                        const uint32_t m_half = (m / (2 * kGran)) * kGran + (m % kGran);
                        const bool m_is_up = (m % (2 * kGran)) >= kGran;
                        const uint32_t m1 = kDwNatural ? m_half + (m_is_up ? kIntermediateHidden : 0u) : m;
                        const uint32_t shared_m1 = kDwNatural ? p * kIntermediateHidden + m_half + (m_is_up ? kIntermediateHidden * kNumPasses : 0u) : p * I2 + m;
                        dw_t* dst;
                        if (is_dw2) {
                            dst = is_shared
                                ? shared_dw2_weights + static_cast<uint64_t>(m) * (kIntermediateHidden * kNumPasses) + p * kIntermediateHidden + nt * MMA_N
                                : dw2_weights + (static_cast<uint64_t>(item.expert) * kHidden + m) * kIntermediateHidden + nt * MMA_N;
                        } else {
                            dst = is_shared
                                ? shared_dw1_weights + static_cast<uint64_t>(shared_m1) * kHidden + nt * MMA_N
                                : dw1_weights + (static_cast<uint64_t>(item.expert) * I2 + m1) * kHidden + nt * MMA_N;
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumTokenGroups; ++ i) {
                            const uint32_t t = i * 8 + frag_col;
                            const float v0 = accum[mh][i * 4 + half * 2], v1 = accum[mh][i * 4 + half * 2 + 1];
                            if constexpr (cute::is_same_v<dw_t, float>)
                                *reinterpret_cast<float2*>(dst + t) = make_float2(v0, v1);
                            else
                                *reinterpret_cast<uint32_t*>(dst + t) = pack_bf16x2(v0, v1);
                        }
                    }
                }
                finish_item(slot);
            }
            slot = slot == kNumItemSlots - 1 ? 0 : slot + 1;
            slot_phase ^= (slot == 0);
        }
    }

    __threadfence_system();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 199>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads) {
        *workspace.get_expert_send_count_ptr(i) = 0;
        *workspace.get_expert_recv_count_ptr(i / kNumExpertsPerRank, i % kNumExpertsPerRank) = 0;
    }
    for (uint32_t i = global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;

    constexpr uint32_t kHiddenVec = kHidden / 8;
    for (uint64_t linear = static_cast<uint64_t>(sm_idx) * kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens) * kHiddenVec;
         linear += static_cast<uint64_t>(kNumSMs) * kNumThreads) {
        const uint32_t token = linear / kHiddenVec, k8 = linear % kHiddenVec;
        float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const auto accumulate = [&](const nv_bfloat16* slot) {
            const uint4 raw = *reinterpret_cast<const uint4*>(slot + k8 * 8);
            const auto* h = reinterpret_cast<const nv_bfloat162*>(&raw);
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i) {
                const float2 f = __bfloat1622float2(h[i]);
                sum[i * 2] += f.x, sum[i * 2 + 1] += f.y;
            }
        };
        #pragma unroll
        for (uint32_t topk = 0; topk < kNumTopk; ++ topk) {
            const auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token * kNumTopk + topk];
            if (e >= 0)
                accumulate(bw.dx_slot_buffer.get_rank_buffer(topk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>());
        }
        if constexpr (kHasShared)
            accumulate(bw.dx_slot_buffer.get_rank_buffer(kNumTopk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>());
        uint4 out;
        auto* out_h = reinterpret_cast<nv_bfloat162*>(&out);
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i)
            out_h[i] = __floats2bfloat162_rn(sum[i * 2], sum[i * 2 + 1]);
        reinterpret_cast<uint4*>(static_cast<nv_bfloat16*>(dx))[linear] = out;
    }

    for (uint64_t linear = static_cast<uint64_t>(sm_idx) * kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens) * kNumTopk;
         linear += static_cast<uint64_t>(kNumSMs) * kNumThreads) {
        const uint32_t token = linear / kNumTopk, topk = linear % kNumTopk;
        const auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[linear];
        dtopk_weights[linear] = e < 0 ? 0.0f : *bw.dtopk_weight_slot_buffer.get_rank_buffer(topk)
            .get_data_buffer(token).template get_base_ptr<float>();
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm
