#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

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

#ifndef SUPERMEOW_MEGA_BWD_PROFILE
#define SUPERMEOW_MEGA_BWD_PROFILE 0 // Enable this to enable our ugly clock64 profili
#endif
#ifndef SUPERMEOW_MEGA_BWD_PROFILE_LAUNCH
#define SUPERMEOW_MEGA_BWD_PROFILE_LAUNCH 4
#endif
#if SUPERMEOW_MEGA_BWD_PROFILE
#define SUPERMEOW_PROF_DECL(...) unsigned long long __VA_ARGS__
#define SUPERMEOW_PROF_TIME(acc, ...) do { const long long _dg_t0 = clock64(); __VA_ARGS__; (acc) += clock64() - _dg_t0; } while (0)
#define SUPERMEOW_PROF(...) __VA_ARGS__
__device__ uint32_t g_dg_mega_bwd_prof_launch = 0;
#else
#define SUPERMEOW_PROF_DECL(...)
#define SUPERMEOW_PROF_TIME(acc, ...) __VA_ARGS__
#define SUPERMEOW_PROF(...)
#endif

namespace deep_gemm {

template<const bool kFastMath>
[[nodiscard]] __device__ __forceinline__ float sigmoid(float x) {
    if constexpr (kFastMath) return 1.0f / (1.0f + __expf(-x));
    else return 1.0f / (1.0f + expf(-x));
}

CUTLASS_DEVICE void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

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
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumPasses = kHasShared ? kNumSharedExperts : 1,
    uint32_t kGran = 8
>
CUTLASS_GLOBAL __launch_bounds__(256, 1) void
sm100_bf16_mega_moe_backward_impl(
    void* __restrict__ dx,
    dw_t* __restrict__ dw1_weights,
    dw_t* __restrict__ dw2_weights,
    float* __restrict__ dtopk_weights,
    dw_t* __restrict__ shared_dw1_weights,
    dw_t* __restrict__ shared_dw2_weights,
    uint32_t num_tokens,
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using bf16_t = cutlass::bfloat16_t;
    using BlockDesc = layout::MegaMoEBackwardBuffer::BlockDesc;

    constexpr uint32_t kNumThreads = 256;
    constexpr uint32_t BLOCK_M = layout::MegaMoEBackwardBuffer::kBlockM;
    constexpr uint32_t kNumZSlots = layout::MegaMoEBackwardBuffer::kNumZSlots;
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = 128;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t CHUNK = 32;
    constexpr uint32_t kNumChunks = UMMA_N / CHUNK;
    constexpr uint32_t UMMA_N_WIDE = 256;
    constexpr uint32_t kNumChunksWide = UMMA_N_WIDE / CHUNK;
    constexpr uint32_t kNumGatherChunks = BLOCK_M / CHUNK;
    constexpr uint32_t kNumEpilogueThreads = 128;
    constexpr uint32_t kEpilogueBarrierIdx = 1;
    constexpr uint32_t kAccumCols = UMMA_N_WIDE;
    constexpr uint32_t kNumAccumStages = 2;
    constexpr uint32_t kNumTmemCols = kNumAccumStages*kAccumCols;
    constexpr uint32_t I2 = kIntermediateHidden<<1;
    constexpr uint32_t kNumG1Tiles = I2 / UMMA_M;
    constexpr uint32_t kNumG2Tiles = kIntermediateHidden / UMMA_M;
    constexpr uint32_t kNumHTiles = kHidden / UMMA_M;
    constexpr uint32_t kNumKBlocksH = kHidden / BLOCK_K;
    constexpr uint32_t kNumKBlocksI2 = I2 / BLOCK_K;
    constexpr uint32_t kNumHTilesWide = kHidden / UMMA_N_WIDE;
    constexpr uint32_t kNumG2TilesWide = kIntermediateHidden / UMMA_N_WIDE;
    constexpr uint32_t kNumDxTiles = kNumHTilesWide;
    constexpr uint32_t DW_M = 2*UMMA_M;
    constexpr uint32_t kNumDW2Tiles = (kHidden / DW_M)*kNumG2TilesWide;
    constexpr uint32_t kNumDW1Tiles = (I2 / DW_M)*kNumHTilesWide;
    constexpr uint32_t kHostStageBytes = (UMMA_M + UMMA_N)*BLOCK_K*sizeof(bf16_t);
    constexpr uint32_t kNumPipeStages = (kNumStages*kHostStageBytes) / ((DW_M + UMMA_N_WIDE)*BLOCK_K*sizeof(bf16_t));
    constexpr uint32_t kNumTilesPerExpert = kNumDW2Tiles + kNumDW1Tiles;
    constexpr uint32_t kNumSharedSlots = kHasShared ? kNumPasses : 0;
    constexpr uint32_t kNumExpertSlots = kNumExpertsPerRank + kNumSharedSlots;
    constexpr uint32_t kNumVecPerRow = (kHidden<<1)>>4;
    constexpr uint32_t kVecPerLane = kNumVecPerRow>>5;
    constexpr uint32_t kVecUnroll = (7&kVecPerLane) == 0 ? 8 : (3&kVecPerLane) == 0 ? 4 : (1&kVecPerLane) == 0 ? 2 : 1;
    constexpr uint32_t kKindGather = 0, kKindZ = 1, kKindDz = 2, kKindDx = 3, kKindDw = 4;
    constexpr uint32_t kGroupBlocks = 32;
    constexpr uint32_t kNumSlots = 4;
    constexpr uint32_t kMetaTokenBits = 20, kMetaRankBits = 6; // We bit pack token index and rank to save memo
    static_assert(kNumMaxTokensPerRank <= (1u<<kMetaTokenBits) && kNumRanks <= 64 && kNumTopk < 64);
    static_assert(kNumZSlots % kGroupBlocks == 0);
    static_assert(kNumExperts % kNumRanks == 0);
    static_assert(kHidden % UMMA_M == 0 && kIntermediateHidden % UMMA_M == 0);
    static_assert(kIntermediateHidden % kGran == 0);
    static_assert(kNumTopk <= 32);
    static_assert(kNumSMs > 1);
    static_assert(BLOCK_M == UMMA_N && BLOCK_M % BLOCK_K == 0 && BLOCK_M % 4 == 0 && BLOCK_M <= kNumEpilogueThreads);
    static_assert((31&kNumVecPerRow) == 0);
    static_assert(kNumTmemCols <= 512);
    static_assert(kNumExpertSlots <= layout::MegaMoEBackwardBuffer::kMaxExpertSlots);
    static_assert(kNumPipeStages >= 2);
    static_assert(kHidden % UMMA_N_WIDE == 0 && kIntermediateHidden % UMMA_N_WIDE == 0);
    uint32_t tid = threadIdx.x;
    uint32_t sm_idx = blockIdx.x;
    uint32_t warp = tid>>5;
    uint32_t lane = tid&31;
    constexpr uint32_t kNumGlobalThreads = kNumSMs*kNumThreads;
    uint32_t global_tid = sm_idx*kNumThreads + tid;
    SUPERMEOW_PROF(const long long prof_t_start = clock64(); uint32_t prof_launch = *reinterpret_cast<volatile uint32_t*>(&g_dg_mega_bwd_prof_launch);)
    if (warp == 0) { // Prefetch TMA descs
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
    auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        false,
        kNumSharedExperts
    );
    auto workspace = buffer.workspace;
    auto bw = layout::MegaMoEBackwardBuffer(
        buffer.get_end_ptr(),
        kHidden, kIntermediateHidden, kNumMaxTokensPerRank, kNumTopk, kNumSharedExperts,
        workspace.num_max_pool_tokens, kNumSMs
    );
    auto* x_pool = static_cast<nv_bfloat16*>(bw.x_pool);
    auto* dy_pool = static_cast<nv_bfloat16*>(bw.dy_pool);
    auto* dz_pool = static_cast<nv_bfloat16*>(bw.dz_pool);
    auto* hw_pool = static_cast<nv_bfloat16*>(bw.hw_pool);
    auto* z_scratch = static_cast<nv_bfloat16*>(bw.z_scratch);
    uint64_t pool_stride = bw.num_pool_rows;
    #pragma unroll
    for (uint32_t i=global_tid; i < kNumExperts; i += kNumGlobalThreads)
        *workspace.get_expert_send_count_ptr(i)=0;
    #pragma unroll
    for (uint32_t i=global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i)=0;
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 101>( workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); }, false, true);
    for (uint32_t idx=global_tid; idx < num_tokens*kNumTopk; idx += kNumGlobalThreads) {
        int64_t expert_idx = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[idx];
        if (expert_idx < 0) continue;
        auto dst_rank = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
        auto dst_local_expert = static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
        auto slot = static_cast<uint32_t>(ptx::atomic_add(workspace.get_expert_send_count_ptr(expert_idx), static_cast<uint64_t>(1)));
        auto dst_ptr = workspace.get_src_token_topk_idx_ptr(dst_local_expert, sym_buffer.rank_idx, slot);
        *sym_buffer.map(dst_ptr, dst_rank) = idx;
    }
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 102>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
    for (uint32_t i=global_tid; i < kNumExperts; i += kNumGlobalThreads) {
        auto dst_rank = i / kNumExpertsPerRank;
        auto dst_local_expert = i % kNumExpertsPerRank;
        auto count = static_cast<uint32_t>(*workspace.get_expert_send_count_ptr(i));
        *sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert), dst_rank) = count;
        ptx::atomic_add_sys( sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert), dst_rank), static_cast<uint64_t>(count));
    }
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 102+1>(workspace, sym_buffer, sm_idx, tid, [&]() -> void { __syncthreads(); });
    if (sm_idx == 0 && tid == 0) {
        uint32_t pool_base=0, meta_base=0, num_blocks=0;
        for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
            auto valid_m = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            auto blocks_e = math::ceil_div(valid_m, BLOCK_M);
            bw.expert_pool_base[e] = pool_base;
            bw.expert_num_blocks[e] = blocks_e;
            bw.expert_done[e]=0;
            bw.expert_a2_target[e] = blocks_e*kNumG2Tiles;
            for (uint32_t off=0; off < valid_m; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = e;
                bw.block_desc[num_blocks].pool_begin = pool_base + off;
                bw.block_desc[num_blocks].meta_begin = meta_base + off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, valid_m - off);
                ++num_blocks;
            }
            pool_base += blocks_e*BLOCK_M;
            meta_base += valid_m;
        }
        uint32_t num_routed_blocks = num_blocks;
        if constexpr (kHasShared) {
            auto blocks_s = math::ceil_div(num_tokens, BLOCK_M);
            for (uint32_t p=0; p < kNumPasses; ++p) {
                bw.expert_pool_base[kNumExpertsPerRank + p] = pool_base + p*bw.shared_region_stride;
                bw.expert_num_blocks[kNumExpertsPerRank + p] = blocks_s;
                bw.expert_done[kNumExpertsPerRank + p]=0;
                bw.expert_a2_target[kNumExpertsPerRank + p] = blocks_s*kNumPasses*kNumG2Tiles;
            }
            for (uint32_t off=0; off < num_tokens; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = kNumExpertsPerRank;
                bw.block_desc[num_blocks].pool_begin = pool_base + off;
                bw.block_desc[num_blocks].meta_begin = off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, num_tokens - off);
                ++num_blocks;
            }
            pool_base += kNumPasses*bw.shared_region_stride;
        }
        DG_DEVICE_ASSERT(num_blocks <= bw.max_num_blocks);
        DG_DEVICE_ASSERT(pool_base <= bw.num_pool_rows);
        *bw.num_blocks = num_blocks;
        *bw.num_routed_blocks = num_routed_blocks;
        uint32_t num_items = kNumExpertSlots*kNumTilesPerExpert;
        for (uint32_t b0 = 0; b0 < num_blocks; b0 += kGroupBlocks) {
            uint32_t nb = cute::min(kGroupBlocks, num_blocks - b0);
            uint32_t r = num_routed_blocks > b0 ? cute::min(nb, num_routed_blocks - b0) : 0u;
            num_items += nb*kNumGatherChunks + (r + (nb - r)*kNumPasses)*(kNumG1Tiles + kNumG2Tiles) + nb*kNumDxTiles;
        }
        *bw.num_items = num_items;
        *bw.next_item=0;
    }
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    __shared__ uint32_t s_pool_begin[kNumExpertsPerRank];
    __shared__ uint32_t s_rank_prefix[kNumExpertsPerRank][kNumRanks];
    if (tid == 0) {
        uint32_t pool_begin=0;
        for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
            s_pool_begin[e] = pool_begin;
            uint32_t rank_prefix=0;
            for (uint32_t r=0; r < kNumRanks; ++r) {
                s_rank_prefix[e][r] = rank_prefix;
                rank_prefix += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, e));
            }
            pool_begin += rank_prefix;
        }
    }
    __syncthreads();
    for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
        auto valid_m_e = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
        auto pool_begin_e = s_pool_begin[e];
        for (uint32_t local_pos = global_tid; local_pos < valid_m_e; local_pos += kNumGlobalThreads) {
            uint32_t r=0;
            #pragma unroll
            for (uint32_t rr=0; rr < kNumRanks; ++rr)
                if (s_rank_prefix[e][rr] <= local_pos) r = rr;
            auto slot = local_pos - s_rank_prefix[e][r];
            auto token_topk = *workspace.get_src_token_topk_idx_ptr(e, r, slot);
            auto src_token = token_topk / kNumTopk;
            auto src_topk = token_topk % kNumTopk;
            *workspace.get_token_src_metadata_ptr(pool_begin_e + local_pos) =
                layout::TokenSrcMetadata(r, src_token, src_topk);
            bw.meta_weight[pool_begin_e + local_pos] = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token*kNumTopk + src_topk, r);
        }
    }
    uint32_t total_blocks = *bw.num_blocks;
    for (uint32_t i=global_tid; i < total_blocks; i += kNumGlobalThreads) {
        bw.block_a0_done[i] = 0;
        bw.block_a1_done[i] = 0;
        bw.block_a2_done[i] = 0;
    }
    for (uint32_t i=global_tid; i < total_blocks*BLOCK_M; i += kNumGlobalThreads)
        bw.block_dtopk[i] = 0.0f;
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    struct Item final {
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
    struct SharedStorage final {
        alignas(1024) bf16_t a[kNumPipeStages][DW_M*BLOCK_K];
        alignas(1024) bf16_t b[kNumPipeStages][UMMA_N_WIDE*BLOCK_K];
        float route_weight[kNumSlots][BLOCK_M];
        uint32_t src_meta[kNumSlots][BLOCK_M];
        Item item[kNumSlots];
        uint32_t item_valid[kNumSlots];
        uint32_t g_block[2], g_pool_begin[2], g_valid_m[2], g_row_begin[2], g_valid[2];
        uint32_t g_src_rank[2][CHUNK];
        uint32_t g_src_token[2][CHUNK];
        Barrier gfull[2];
        Barrier gempty[2];
        alignas(16) bf16_t epi_stage[4][CHUNK][CHUNK + 8];
        Barrier slot_full[kNumSlots];
        Barrier slot_empty[kNumSlots];
        Barrier full[kNumPipeStages];
        Barrier empty[kNumPipeStages];
        Barrier tmem_full[kNumAccumStages];
        Barrier tmem_empty[kNumAccumStages];
        uint32_t tmem_ptr;
    };
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    constexpr uint32_t kStageABytes = sizeof(smem.a[0]);
    constexpr uint32_t kStageBBytes = sizeof(smem.b[0]);
    static_assert(sizeof(SharedStorage) <= kNumStages*kHostStageBytes+0x4000, "Shared storage exceeds the host allocation");
    auto meta_token = [&](uint32_t slot, uint32_t r) -> uint32_t { return smem.src_meta[slot][r]&((1u<<kMetaTokenBits)-1); };
    auto meta_rank = [&](uint32_t slot, uint32_t r) -> uint32_t { return (smem.src_meta[slot][r]>>kMetaTokenBits)&((1u<<kMetaRankBits)-1); };
    auto meta_topk = [&](uint32_t slot, uint32_t r) -> uint32_t { return smem.src_meta[slot][r]>>(kMetaTokenBits+kMetaRankBits); };
    if (warp == 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i=0; i < kNumSlots; ++i) {
                smem.slot_full[i].init(1);
                smem.slot_empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i=0; i < 2; ++i) {
                smem.gfull[i].init(1);
                smem.gempty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i=0; i < kNumPipeStages; ++i) {
                smem.full[i].init(1);
                smem.empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i=0; i < kNumAccumStages; ++i) {
                smem.tmem_full[i].init(1);
                smem.tmem_empty[i].init(kNumEpilogueThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp == 3) {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    __syncthreads();
    auto pg = [](uint32_t j) -> uint32_t { return ((j / kGran)<<1)*kGran + (j%kGran); };
    auto z_slot_ptr = [&](uint32_t zslot, uint32_t pass) -> nv_bfloat16* { return z_scratch + (static_cast<uint64_t>(zslot)*kNumPasses + pass)*I2*BLOCK_M;};
    auto block_passes = [&](uint32_t block) -> uint32_t { return block >= *bw.num_routed_blocks ? kNumPasses : 1u;};
    auto gather_rows = [&](uint32_t gslot, uint32_t pool_begin, uint32_t valid_m, uint32_t row_begin) {
        for (uint32_t rr=0; rr < CHUNK; rr += 2) {
            uint32_t rows[2] = {row_begin + rr, row_begin + rr + 1};
            uint4* dst_x[2];
            uint4* dst_dy[2];
            const uint4* src_x[2];
            const uint4* src_dy[2];
            bool valid[2];
            #pragma unroll
            for (uint32_t h=0; h < 2; ++h) {
                dst_x[h] = reinterpret_cast<uint4*>(x_pool + static_cast<uint64_t>(pool_begin + rows[h])*kHidden);
                dst_dy[h] = reinterpret_cast<uint4*>(dy_pool + static_cast<uint64_t>(pool_begin + rows[h])*kHidden);
                valid[h] = rows[h] < valid_m;
                auto r_rank = smem.g_src_rank[gslot][rr + h];
                auto r_token = smem.g_src_token[gslot][rr + h];
                src_x[h] = sym_buffer.map(reinterpret_cast<const uint4*>(
                    buffer.input_token_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
                src_dy[h] = sym_buffer.map(reinterpret_cast<const uint4*>(
                    bw.input_dy_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
            }
            for (uint32_t base = lane; base < kNumVecPerRow; base += 32*kVecUnroll) {
                uint4 vx[2][kVecUnroll], vy[2][kVecUnroll];
                #pragma unroll
                for (uint32_t h=0; h < 2; ++h) {
                    #pragma unroll
                    for (uint32_t i=0; i < kVecUnroll; ++i) {
                        vx[h][i] = valid[h] ? src_x[h][base + (i<<5)] : make_uint4(0, 0, 0, 0);
                        vy[h][i] = valid[h] ? src_dy[h][base + (i<<5)] : make_uint4(0, 0, 0, 0);
                    }
                }
                #pragma unroll
                for (uint32_t h=0; h < 2; ++h) {
                    #pragma unroll
                    for (uint32_t i=0; i < kVecUnroll; ++i) {
                        dst_x[h][base + (i<<5)] = vx[h][i];
                        dst_dy[h][base + (i<<5)] = vy[h][i];
                    }
                }
            }
        }
        fence_proxy_async_global();
    };
    auto wait_counter = [&](uint32_t* counter, uint32_t target) -> void { while (ptx::ld_acq(counter) < target) __nanosleep(128);};
    SUPERMEOW_PROF(const long long prof_t_main = clock64(); const bool prof_print = prof_launch == SUPERMEOW_MEGA_BWD_PROFILE_LAUNCH; uint32_t prof_rank = sym_buffer.rank_idx;)
    if (warp == 0) {
        uint32_t slot = 0, slot_phase = 0;
        uint32_t gslot = 0, gphase = 0;
        SUPERMEOW_PROF_DECL(p_dep[5] = {0, 0, 0, 0, 0}, p_n[5] = {0, 0, 0, 0, 0}, p_slot = 0, p_gempty = 0);
        for (;;) {
            uint32_t item_idx = 0;
            if (lane == 0) item_idx = ptx::atomic_add(bw.next_item, 1u);
            item_idx = __shfl_sync(0xffffffff, item_idx, 0);
            uint32_t num_blocks = *bw.num_blocks;
            uint32_t num_routed_blocks = *bw.num_routed_blocks;
            uint32_t num_block_items = *bw.num_items - kNumExpertSlots*kNumTilesPerExpert;
            const bool done = item_idx >= *bw.num_items;
            Item cur{};
            if (!done) {
                if (item_idx < num_block_items) {
                    auto group_of = [&](uint32_t g, uint32_t& b0_, uint32_t& nb_, uint32_t& r_) -> void {
                        b0_ = g*kGroupBlocks;
                        nb_ = b0_ < num_blocks ? cute::min(kGroupBlocks, num_blocks - b0_) : 0u;
                        r_ = num_routed_blocks > b0_ ? cute::min(nb_, num_routed_blocks - b0_) : 0u;
                    };
                    auto gather_count = [&](uint32_t nb_) { return nb_*kNumGatherChunks; };
                    auto compute_count = [&](uint32_t nb_, uint32_t r_) { return (r_ + (nb_ - r_)*kNumPasses)*(kNumG1Tiles + kNumG2Tiles) + nb_*kNumDxTiles; };
                    uint32_t rem = item_idx, b0 = 0, nb = 0, r = 0;
                    bool is_gather = false;
                    group_of(0, b0, nb, r);
                    if (rem < gather_count(nb)) {
                        is_gather = true;
                    } else {
                        rem -= gather_count(nb);
                        for (uint32_t g=0;; ++g) {
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
                    auto decode_tiles = [&](uint32_t idx, uint32_t tiles, uint32_t& block, uint32_t& pass, uint32_t& tile) {
                        if (idx < r*tiles) { block = b0 + idx / tiles; pass = 0; tile = idx % tiles; }
                        else {
                            idx -= r*tiles;
                            block = b0 + r + idx / (kNumPasses*tiles);
                            pass = (idx % (kNumPasses*tiles)) / tiles;
                            tile = idx % tiles;
                        }
                    };
                    if (is_gather) {
                        cur.kind = kKindGather;
                        cur.block = b0 + rem / kNumGatherChunks;
                        cur.tile = rem % kNumGatherChunks;
                    } else if (rem < (r + (nb - r)*kNumPasses)*kNumG1Tiles) {
                        cur.kind = kKindZ;
                        decode_tiles(rem, kNumG1Tiles, cur.block, cur.pass, cur.tile);
                        cur.num_k_blocks = kNumKBlocksH;
                    } else if ((rem -= (r + (nb - r)*kNumPasses)*kNumG1Tiles) < (r + (nb - r)*kNumPasses)*kNumG2Tiles) {
                        cur.kind = kKindDz;
                        decode_tiles(rem, kNumG2Tiles, cur.block, cur.pass, cur.tile);
                        cur.num_k_blocks = kNumKBlocksH;
                    } else {
                        rem -= (r + (nb - r)*kNumPasses)*kNumG2Tiles;
                        cur.kind = kKindDx;
                        cur.block = b0 + rem / kNumDxTiles;
                        cur.tile = rem % kNumDxTiles;
                        cur.num_k_blocks = block_passes(cur.block)*kNumKBlocksI2;
                    }
                    uint32_t num_passes = block_passes(cur.block);
                    auto bd = bw.block_desc[cur.block];
                    cur.expert = bd.local_expert;
                    cur.pool_begin = bd.pool_begin;
                    cur.x_begin = bd.pool_begin;
                    cur.valid_m = bd.valid_m;
                    cur.zslot = cur.block % kNumZSlots;
                    if (cur.kind == kKindGather) {
                        SUPERMEOW_PROF_TIME(p_gempty, smem.gempty[gslot].wait(gphase ^ 1));
                        SUPERMEOW_PROF(++p_n[kKindGather];)
                        const bool is_shared = cur.expert >= kNumExpertsPerRank;
                        uint32_t r = cur.tile*CHUNK + lane;
                        uint32_t src_rank = sym_buffer.rank_idx, src_token = 0;
                        if (r < bd.valid_m) {
                            if (is_shared) {
                                src_token = bd.meta_begin + r;
                            } else {
                                auto meta = *workspace.get_token_src_metadata_ptr(bd.meta_begin + r);
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
                            smem.g_row_begin[gslot] = cur.tile*CHUNK;
                            smem.g_valid[gslot] = 1;
                        }
                        __syncwarp();
                        if (lane == 0)
                            smem.gfull[gslot].arrive();
                        gslot^=1;
                        gphase^=(gslot == 0);
                        continue;
                    }
                    if (lane == 0) {
                        SUPERMEOW_PROF(const long long _dg_t0 = clock64();)
                        if (cur.kind == kKindZ) {
                            wait_counter(bw.block_a0_done + cur.block, kNumGatherChunks);
                            if (cur.block >= kNumZSlots)
                                wait_counter(bw.block_a2_done + cur.block - kNumZSlots, block_passes(cur.block - kNumZSlots)*kNumG2Tiles);
                        } else if (cur.kind == kKindDz) {
                            wait_counter(bw.block_a1_done + cur.block, num_passes*kNumG1Tiles);
                        } else if (cur.kind == kKindDx) {
                            wait_counter(bw.block_a2_done + cur.block, num_passes*kNumG2Tiles);
                        }
                        SUPERMEOW_PROF(p_dep[cur.kind] += clock64() - _dg_t0; ++p_n[cur.kind];)
                    }
                } else {
                    uint32_t t = item_idx - num_block_items;
                    cur.kind = kKindDw;
                    cur.expert = t / kNumTilesPerExpert;
                    cur.tile = t % kNumTilesPerExpert;
                    cur.num_k_blocks = bw.expert_num_blocks[cur.expert]*(BLOCK_M / BLOCK_K);
                    cur.pool_begin = bw.expert_pool_base[cur.expert];
                    cur.x_begin = bw.expert_pool_base[cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank];
                    uint32_t done_slot = cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank;
                    if (lane == 0) {
                        SUPERMEOW_PROF_TIME(p_dep[kKindDw], wait_counter(bw.expert_done + done_slot, bw.expert_a2_target[done_slot]));
                        SUPERMEOW_PROF(++p_n[kKindDw];)
                    }
                }
                __syncwarp();
                fence_proxy_async_global();
            }
            SUPERMEOW_PROF_TIME(p_slot, smem.slot_empty[slot].wait(slot_phase ^ 1));
            if (!done && cur.kind != kKindDw) {
                auto bd = bw.block_desc[cur.block];
                    const bool is_shared = cur.expert >= kNumExpertsPerRank;
                    for (uint32_t r = lane; r < BLOCK_M; r += 32) {
                        uint32_t src_rank = sym_buffer.rank_idx, src_token = 0, src_topk = kNumTopk;
                        float weight = 0.0f;
                        if (r < bd.valid_m) {
                            if (is_shared) {
                                src_token = bd.meta_begin + r;
                                weight = 1.0f;
                            } else {
                                auto meta = *workspace.get_token_src_metadata_ptr(bd.meta_begin + r);
                                src_rank = meta.rank_idx;
                                src_token = meta.token_idx;
                                src_topk = meta.topk_idx;
                                weight = bw.meta_weight[bd.meta_begin + r];
                            }
                        }
                        smem.route_weight[slot][r] = weight;
                        smem.src_meta[slot][r] = src_token | (src_rank << kMetaTokenBits) | (src_topk << (kMetaTokenBits + kMetaRankBits));
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
                smem.gempty[gslot].wait(gphase ^ 1);
                if (lane == 0)
                    smem.g_valid[gslot] = 0;
                __syncwarp();
                if (lane == 0)
                    smem.gfull[gslot].arrive();
                break;
            }
            slot = (slot + 1) % kNumSlots;
            slot_phase^=(slot == 0);
        }
        SUPERMEOW_PROF(if (prof_print && lane == 0) {
            printf("DGPROF r=%u sm=%u role=sched total=%lld dep_z=%llu dep_dz=%llu dep_dx=%llu dep_dw=%llu n_g=%llu n_z=%llu n_dz=%llu n_dx=%llu n_dw=%llu slot_wait=%llu gempty_wait=%llu\n",
                   prof_rank, sm_idx, clock64() - prof_t_main, p_dep[kKindZ], p_dep[kKindDz], p_dep[kKindDx], p_dep[kKindDw],
                   p_n[kKindGather], p_n[kKindZ], p_n[kKindDz], p_n[kKindDx], p_n[kKindDw], p_slot, p_gempty);
        })
    } else if (warp == 3) {
        uint32_t gslot = 0, gphase = 0;
        SUPERMEOW_PROF_DECL(p_gwait = 0, p_gwork = 0);
        for (;;) {
            SUPERMEOW_PROF_TIME(p_gwait, smem.gfull[gslot].wait(gphase));
            if (!smem.g_valid[gslot])
                break;
            SUPERMEOW_PROF_TIME(p_gwork, gather_rows(gslot, smem.g_pool_begin[gslot], smem.g_valid_m[gslot], smem.g_row_begin[gslot]));
            __threadfence();
            __syncwarp();
            if (lane == 0) {
                ptx::atomic_add_rel(bw.block_a0_done + smem.g_block[gslot], 1u);
                smem.gempty[gslot].arrive();
            }
            gslot^=1;
            gphase^=(gslot == 0);
        }
        SUPERMEOW_PROF(if (prof_print && lane == 0) printf("DGPROF r=%u sm=%u role=gather total=%lld wait=%llu work=%llu\n", prof_rank, sm_idx, clock64() - prof_t_main, p_gwait, p_gwork);)
    } else if (warp == 1) {
        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        auto advance = [&]() {
            stage = (stage + 1) % kNumPipeStages;
            phase^=!stage;
        };
        auto issue = [&](uint32_t num_bytes) {
            smem.full[stage].arrive_and_expect_tx(num_bytes);
            advance();
        };
        constexpr uint32_t kStageBytes = UMMA_M*BLOCK_K*2 + UMMA_N*BLOCK_K*2;
        constexpr uint32_t kStageBytesWide = UMMA_M*BLOCK_K*2 + UMMA_N_WIDE*BLOCK_K*2;
        constexpr uint32_t kBHalf = UMMA_N*BLOCK_K;
        constexpr uint32_t kAHalf = UMMA_M*BLOCK_K;
        constexpr uint32_t kStageBytesDw = DW_M*BLOCK_K*2 + UMMA_N_WIDE*BLOCK_K*2;
        SUPERMEOW_PROF_DECL(p_empty = 0, p_slotw = 0);
        for (;;) {
            SUPERMEOW_PROF_TIME(p_slotw, smem.slot_full[slot].wait(slot_phase));
            if (!smem.item_valid[slot])
                break;
            auto item = smem.item[slot];
            const bool is_shared = item.expert >= kNumExpertsPerRank;
            auto* w1k = is_shared ? &tensor_map_shared_w1_k : &tensor_map_w1_k;
            auto* w1mn = is_shared ? &tensor_map_shared_w1_mn : &tensor_map_w1_mn;
            auto* w2mn = is_shared ? &tensor_map_shared_w2_mn : &tensor_map_w2_mn;
            if (item.kind == kKindZ) {
                uint32_t w1_rows = is_shared ? item.pass*I2 : item.expert*I2;
                for (uint32_t kb = 0; kb < kNumKBlocksH; ++kb) {
                    SUPERMEOW_PROF_TIME(p_empty, smem.empty[stage].wait(phase ^ 1));
                    if (cute::elect_one_sync()) {
                        tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb*BLOCK_K, w1_rows + item.tile*UMMA_M);
                        tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_x_k, &smem.full[stage], smem.b[stage], kb*BLOCK_K, item.pool_begin);
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDz) {
                uint32_t w2_rows = is_shared ? 0u : item.expert*kHidden;
                uint32_t w2_cols = is_shared ? item.pass*kIntermediateHidden : 0u;
                for (uint32_t kb = 0; kb < kNumKBlocksH; ++kb) {
                    SUPERMEOW_PROF_TIME(p_empty, smem.empty[stage].wait(phase ^ 1));
                    if (cute::elect_one_sync()) {
                        tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(w2mn, &smem.full[stage], smem.a[stage], w2_cols + item.tile*UMMA_M, w2_rows + kb*BLOCK_K);
                        tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_dy_k, &smem.full[stage], smem.b[stage], kb*BLOCK_K, item.pool_begin);
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDx) {
                uint32_t w1_rows = is_shared ? 0u : item.expert*I2;
                for (uint32_t kb = 0; kb < item.num_k_blocks; ++kb) {
                    SUPERMEOW_PROF_TIME(p_empty, smem.empty[stage].wait(phase ^ 1));
                    if (cute::elect_one_sync()) {
                        tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(&tensor_map_dz_mn, &smem.full[stage], smem.a[stage],
                                                                item.pool_begin + (kb / kNumKBlocksI2)*bw.shared_region_stride, (kb % kNumKBlocksI2)*BLOCK_K);
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++h)
                            tma::copy<UMMA_N, BLOCK_K, 128, bf16_t>(w1mn, &smem.full[stage], smem.b[stage] + h*kBHalf,
                                                                    item.tile*UMMA_N_WIDE + h*UMMA_N, w1_rows + kb*BLOCK_K);
                        issue(kStageBytesWide);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            } else if (item.kind == kKindDw) {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                uint32_t mt = is_dw2 ? item.tile / kNumG2TilesWide : (item.tile - kNumDW2Tiles) / kNumHTilesWide;
                uint32_t nt = is_dw2 ? item.tile % kNumG2TilesWide : (item.tile - kNumDW2Tiles) % kNumHTilesWide;
                for (uint32_t kb = 0; kb < item.num_k_blocks; ++kb) {
                    SUPERMEOW_PROF_TIME(p_empty, smem.empty[stage].wait(phase ^ 1));
                    if (cute::elect_one_sync()) {
                        if (is_dw2) {
                            tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(&tensor_map_dy_mn, &smem.full[stage], smem.a[stage], mt*UMMA_M, item.x_begin + kb*BLOCK_K);
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h)
                                tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(&tensor_map_dy_mn, &smem.full[stage], smem.a[stage] + h*kAHalf,
                                                                        mt*DW_M + h*UMMA_M, item.x_begin + kb*BLOCK_K);
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h)
                                tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_hw_k, &smem.full[stage], smem.b[stage] + h*kBHalf,
                                                                        item.pool_begin + kb*BLOCK_K, nt*UMMA_N_WIDE + h*UMMA_N);
                        } else {
                            tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(&tensor_map_dz_k, &smem.full[stage], smem.a[stage], item.pool_begin + kb*BLOCK_K, mt*UMMA_M);
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h)
                                tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(&tensor_map_dz_k, &smem.full[stage], smem.a[stage] + h*kAHalf,
                                                                        item.pool_begin + kb*BLOCK_K, mt*DW_M + h*UMMA_M);
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h)
                                tma::copy<UMMA_N, BLOCK_K, 128, bf16_t>(&tensor_map_x_mn, &smem.full[stage], smem.b[stage] + h*kBHalf,
                                                                        nt*UMMA_N_WIDE + h*UMMA_N, item.x_begin + kb*BLOCK_K);
                        }
                        issue(kStageBytesWide);
                        issue(kStageBytesDw);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            }
            slot = (slot + 1) % kNumSlots;
            slot_phase^=(slot == 0);
        }
        SUPERMEOW_PROF(if (prof_print && lane == 0)
            printf("DGPROF r=%u sm=%u role=tma total=%lld empty_wait=%llu slot_wait=%llu\n", prof_rank, sm_idx, clock64() - prof_t_main, p_empty, p_slotw);)
    } else if (warp == 2) {
        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t task_idx = 0;
        SUPERMEOW_PROF_DECL(p_full[5] = {0, 0, 0, 0, 0}, p_tempty = 0, p_slotw = 0, p_kb[5] = {0, 0, 0, 0, 0});
        SUPERMEOW_PROF(uint32_t prof_kind = 0;)
        auto advance = [&]() {
            stage = (stage + 1) % kNumPipeStages;
            phase^=!stage;
        };

        auto idesc_kk = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto idesc_mk = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::MN, cute::UMMA::Major::K>();
        auto idesc_km = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::MN>();
        auto idesc_mk_w = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N_WIDE, cute::UMMA::Major::MN, cute::UMMA::Major::K>();
        auto idesc_km_w = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N_WIDE, cute::UMMA::Major::K, cute::UMMA::Major::MN>();
        auto idesc_mm_w = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N_WIDE, cute::UMMA::Major::MN, cute::UMMA::Major::MN>();

        auto a_k = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto a_mn = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto b_k = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_N, BLOCK_K, 128>(smem.b[0], 0, 0);
        auto b_mn = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_N, BLOCK_K, 128>(smem.b[0], 0, 0);
        uint32_t a_k_lo = a_k.lo, a_mn_lo = a_mn.lo, b_k_lo = b_k.lo, b_mn_lo = b_mn.lo;

        auto run_task = [&](auto& a_desc, auto& b_desc, uint32_t a_lo, uint32_t b_lo,
                                  auto advance_a, auto advance_b,
                                  const cute::UMMA::InstrDescriptor& idesc,
                                  uint32_t num_k_blocks) {
            auto accum = task_idx % kNumAccumStages;
            auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            SUPERMEOW_PROF_TIME(p_tempty, smem.tmem_empty[accum].wait(accum_phase ^ 1));
            ptx::tcgen05_after_thread_sync();
            auto runtime_idesc = cute::UMMA::make_runtime_instr_desc(idesc);
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                SUPERMEOW_PROF_TIME(p_full[prof_kind], smem.full[stage].wait(phase));
                SUPERMEOW_PROF(++p_kb[prof_kind];)
                ptx::tcgen05_after_thread_sync();
                uint32_t a_base = a_lo + stage*(kStageABytes / 16);
                uint32_t b_base = b_lo + stage*(kStageBBytes / 16);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                        a_desc.lo = advance_a(a_base, k*UMMA_K);
                        b_desc.lo = advance_b(b_base, k*UMMA_K);
                        ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, accum*kAccumCols, kb > 0 || k > 0, runtime_idesc);
                    }
                }
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty[stage]));
                if (kb == num_k_blocks - 1)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum]));
                __syncwarp();
                advance();
            }
            if (num_k_blocks == 0) {
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum]));
                __syncwarp();
            }
        };
        auto run_task_dw = [&](auto& a_desc, auto& b_desc, uint32_t a_lo, uint32_t b_lo,
                                     auto advance_a, auto advance_b,
                                     const cute::UMMA::InstrDescriptor& idesc,
                                     uint32_t num_k_blocks) {
            uint32_t accum[2];
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++h) {
                accum[h] = task_idx % kNumAccumStages;
                auto accum_phase = (task_idx / kNumAccumStages) & 1;
                ++task_idx;
                SUPERMEOW_PROF_TIME(p_tempty, smem.tmem_empty[accum[h]].wait(accum_phase ^ 1));
            }
            ptx::tcgen05_after_thread_sync();
            auto runtime_idesc = cute::UMMA::make_runtime_instr_desc(idesc);
            constexpr uint32_t kAHalfDesc = (UMMA_M*BLOCK_K*sizeof(bf16_t)) / 16;
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                SUPERMEOW_PROF_TIME(p_full[prof_kind], smem.full[stage].wait(phase));
                SUPERMEOW_PROF(++p_kb[prof_kind];)
                ptx::tcgen05_after_thread_sync();
                uint32_t a_base = a_lo + stage*(kStageABytes / 16);
                uint32_t b_base = b_lo + stage*(kStageBBytes / 16);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                        b_desc.lo = advance_b(b_base, k*UMMA_K);
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++h) {
                            a_desc.lo = advance_a(a_base + h*kAHalfDesc, k*UMMA_K);
                            ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, accum[h]*kAccumCols, kb > 0 || k > 0, runtime_idesc);
                        }
                    }
                }
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty[stage]));
                if (kb == num_k_blocks - 1) {
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum[0]]));
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum[1]]));
                }
                __syncwarp();
                advance();
            }
            if (num_k_blocks == 0) {
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum[0]]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum[1]]));
                __syncwarp();
            }
        };
        auto adv_a_k = [](uint32_t base, uint32_t k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_M, 128, bf16_t>(base, 0, k); };
        auto adv_a_mn = [](uint32_t base, uint32_t k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_M, 128, bf16_t>(base, 0, k); };
        auto adv_b_k = [](uint32_t base, uint32_t k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_N, 128, bf16_t>(base, 0, k); };
        auto adv_b_mn = [](uint32_t base, uint32_t k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_N, 128, bf16_t>(base, 0, k); };

        for (;;) {
            SUPERMEOW_PROF_TIME(p_slotw, smem.slot_full[slot].wait(slot_phase));
            if (!smem.item_valid[slot])
                break;
            auto item = smem.item[slot];
            SUPERMEOW_PROF(prof_kind = item.kind;)
            if (item.kind == kKindZ) {
                run_task(a_k, b_k, a_k_lo, b_k_lo, adv_a_k, adv_b_k, idesc_kk, kNumKBlocksH);
            } else if (item.kind == kKindDz) {
                run_task(a_mn, b_k, a_mn_lo, b_k_lo, adv_a_mn, adv_b_k, idesc_mk, kNumKBlocksH);
            } else if (item.kind == kKindDx) {
                run_task(a_mn, b_mn, a_mn_lo, b_mn_lo, adv_a_mn, adv_b_mn, idesc_mm_w, item.num_k_blocks);
            } else if (item.kind == kKindDw) {
                if (item.tile < kNumDW2Tiles)
                    run_task_dw(a_mn, b_k, a_mn_lo, b_k_lo, adv_a_mn, adv_b_k, idesc_mk_w, item.num_k_blocks);
                else
                    run_task_dw(a_k, b_mn, a_k_lo, b_mn_lo, adv_a_k, adv_b_mn, idesc_km_w, item.num_k_blocks);
            }
            slot = (slot + 1) % kNumSlots;
            slot_phase^=(slot == 0);
        }
        SUPERMEOW_PROF(if (prof_print && lane == 0)
            printf("DGPROF r=%u sm=%u role=mma total=%lld full_z=%llu full_dz=%llu full_dx=%llu full_dw=%llu kb_z=%llu kb_dz=%llu kb_dx=%llu kb_dw=%llu tmem_empty_wait=%llu slot_wait=%llu\n",
                   prof_rank, sm_idx, clock64() - prof_t_main, p_full[kKindZ], p_full[kKindDz], p_full[kKindDx], p_full[kKindDw],
                   p_kb[kKindZ], p_kb[kKindDz], p_kb[kKindDx], p_kb[kKindDw], p_tempty, p_slotw);)
    } else if (warp >= 4) {
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
        uint32_t epi_warp = warp - 4;
        uint32_t epi_tid = tid - 128;
        uint32_t row = epi_warp*32 + lane;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t task_idx = 0;
        SUPERMEOW_PROF_DECL(p_tfull[5] = {0, 0, 0, 0, 0}, p_item[5] = {0, 0, 0, 0, 0}, p_slotw = 0, p_seg[3][4] = {});
        SUPERMEOW_PROF(long long prof_seg_t = 0;)
#if SUPERMEOW_MEGA_BWD_PROFILE
#define SUPERMEOW_PROF_SEG_BEGIN() (prof_seg_t = clock64())
#define SUPERMEOW_PROF_SEG(k, idx) do { const long long _n = clock64(); p_seg[k][idx] += _n - prof_seg_t; prof_seg_t = _n; } while (0)
#else
#define SUPERMEOW_PROF_SEG_BEGIN() ((void)0)
#define SUPERMEOW_PROF_SEG(k, idx) ((void)0)
#endif
        SUPERMEOW_PROF(uint32_t prof_kind = 0;)

        auto epi_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, kEpilogueBarrierIdx); };
        auto begin_task = [&](uint32_t& accum) {
            accum = task_idx % kNumAccumStages;
            auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            SUPERMEOW_PROF_TIME(p_tfull[prof_kind], smem.tmem_full[accum].wait(accum_phase));
            ptx::tcgen05_after_thread_sync();
        };
        auto load_cols = [&](uint32_t accum, uint32_t col, float* v) {
            ptx::tmem_load_32dp32b<32>(accum*kAccumCols + col, reinterpret_cast<uint32_t*>(v));
            cutlass::arch::fence_view_async_tmem_load();
        };
        auto release_tmem = [&](uint32_t accum) {
            ptx::tcgen05_before_thread_sync();
            smem.tmem_empty[accum].arrive();
        };
        auto pack2 = [](const float& a, const float& b) {
            auto h = __floats2bfloat162_rn(a, b);
            return *reinterpret_cast<uint32_t*>(&h);
        };
        auto store_row_bf16 = [&](nv_bfloat16* dst, const float* v) -> void {
            auto* dst4 = reinterpret_cast<uint4*>(dst);
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i)
                dst4[i] = make_uint4(pack2(v[(i<<3) + 0], v[(i<<3) + 1]), pack2(v[(i<<3) + 2], v[(i<<3) + 3]), pack2(v[(i<<3) + 4], v[(i<<3) + 5]), pack2(v[(i<<3) + 6], v[(i<<3) + 7]));
        };
        auto stage_store = [&](const float* v, const auto& row_ptr) {
            auto* stage = &smem.epi_stage[epi_warp][0][0];
            auto* mine = reinterpret_cast<uint4*>(stage + lane*(CHUNK + 8));
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i)
                mine[i] = make_uint4(pack2(v[(i<<3) + 0], v[(i<<3) + 1]), pack2(v[(i<<3) + 2], v[(i<<3) + 3]), pack2(v[(i<<3) + 4], v[(i<<3) + 5]), pack2(v[(i<<3) + 6], v[(i<<3) + 7]));
            __syncwarp();
            #pragma unroll
            for (uint32_t it=0; it < 4; ++it) {
                uint32_t r = it*8 + (lane >> 2), piece = lane & 3;
                nv_bfloat16* dst = row_ptr(r);
                if (dst != nullptr)
                    reinterpret_cast<uint4*>(dst)[piece] = reinterpret_cast<const uint4*>(stage + r*(CHUNK + 8))[piece];
            }
            __syncwarp();
        };
        auto store_row_f32 = [](float* dst, const float* v) -> void {
            auto* dst4 = reinterpret_cast<float4*>(dst);
            #pragma unroll
            for (uint32_t i=0; i < 8; ++i)
                dst4[i] = make_float4(v[i*4], v[i*4 + 1], v[i*4 + 2], v[i*4 + 3]);
        };
        auto load_row_bf16 = [](const nv_bfloat16* src, float* v) -> void {
            auto* src4 = reinterpret_cast<const uint4*>(src);
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i) {
                const uint4 q = __ldcg(src4 + i);
                auto* h = reinterpret_cast<const nv_bfloat162*>(&q);
                #pragma unroll
                for (uint32_t j=0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(h[j]);
                    v[(i<<3) + j*2] = f.x, v[(i<<3) + j*2 + 1] = f.y;
                }
            }
        };
        auto clamp_gate = [](float g) {
            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                g = cute::min(g, kActivationClamp);
            return g;
        };
        auto clamp_up = [](float u) {
            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                u = cute::max(cute::min(u, kActivationClamp), -kActivationClamp);
            return u;
        };
        auto finish_item = [&](uint32_t slot) {
            epi_sync();
            if (epi_tid == 0)
                smem.slot_empty[slot].arrive();
        };

        for (;;) {
            SUPERMEOW_PROF_TIME(p_slotw, smem.slot_full[slot].wait(slot_phase));
            if (!smem.item_valid[slot]) break;
            auto item = smem.item[slot];
            SUPERMEOW_PROF(prof_kind = item.kind; const long long prof_t_item = clock64();)
            const bool is_shared = item.expert >= kNumExpertsPerRank;
            if (item.kind == kKindZ) {
                uint32_t accum;
                begin_task(accum);
                auto* z_slot = z_slot_ptr(item.zslot, item.pass);
                uint32_t pool_row = item.pool_begin + item.pass*bw.shared_region_stride;
                SUPERMEOW_PROF_SEG_BEGIN();
                #pragma unroll
                for (uint32_t c=0; c < kNumChunks; ++c) {
                    float z[CHUNK];
                    load_cols(accum, c*CHUNK, z);
                    if (c == kNumChunks - 1)
                        release_tmem(accum);
                    SUPERMEOW_PROF_SEG(0, 0);
                    uint32_t prow0 = item.tile*UMMA_M + epi_warp*32;
                    stage_store(z, [&](uint32_t r) { return z_slot + static_cast<uint64_t>(prow0 + r)*BLOCK_M + c*CHUNK; });
                    SUPERMEOW_PROF_SEG(0, 1);
                    float hw[CHUNK];
                    #pragma unroll
                    for (uint32_t t = 0; t < CHUNK; ++t) {
                        const float partner = __shfl_xor_sync(0xffffffff, z[t], kGran);
                        const float g = clamp_gate(z[t]);
                        const float u = clamp_up(partner);
                        hw[t] = g*sigmoid<kFastMath>(g)*u*smem.route_weight[slot][c*CHUNK + t];
                    }
                    stage_store(hw, [&](uint32_t r) -> nv_bfloat16* {
                        uint32_t pr = prow0 + r;
                        if ((pr % (2*kGran)) >= kGran)
                            return nullptr;
                        return hw_pool + static_cast<uint64_t>((pr / (2*kGran))*kGran + (pr % kGran))*pool_stride + pool_row + c*CHUNK;
                    });
                    SUPERMEOW_PROF_SEG(0, 2);
                }
                fence_proxy_async_global();
                epi_sync();
                if (epi_tid == 0) {
                    __threadfence();
                    ptx::atomic_add_rel(bw.block_a1_done + item.block, 1u);
                    smem.slot_empty[slot].arrive();
                }
                SUPERMEOW_PROF_SEG(0, 3);
            } else if (item.kind == kKindDz) {
                uint32_t accum;
                begin_task(accum);
                auto* z_slot = z_slot_ptr(item.zslot, item.pass);
                uint32_t pool_row = item.pool_begin + item.pass*bw.shared_region_stride;
                uint32_t i=item.tile*UMMA_M + row;
                uint32_t pgi = pg(i);
                SUPERMEOW_PROF_SEG_BEGIN();
                #pragma unroll
                for (uint32_t c=0; c < kNumChunks; ++c) {
                    float dh[CHUNK];
                    load_cols(accum, c*CHUNK, dh);
                    if (c == kNumChunks - 1)
                        release_tmem(accum);
                    SUPERMEOW_PROF_SEG(1, 0);
                    float g[CHUNK], u[CHUNK];
                    load_row_bf16(z_slot + static_cast<uint64_t>(pgi)*BLOCK_M + c*CHUNK, g);
                    load_row_bf16(z_slot + static_cast<uint64_t>(pgi + kGran)*BLOCK_M + c*CHUNK, u);
                    float partial[CHUNK], dz_gate[CHUNK], dz_up[CHUNK];
                    #pragma unroll
                    for (uint32_t t = 0; t < CHUNK; ++t) {
                        bool gate_active = true, up_active = true;
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                            gate_active = g[t] <= kActivationClamp;
                            up_active = u[t] >= -kActivationClamp && u[t] <= kActivationClamp;
                        }
                        const float gc = clamp_gate(g[t]), uc = clamp_up(u[t]);
                        const float sig = sigmoid<kFastMath>(gc);
                        const float silu = gc*sig;
                        const float dsilu = sig*(1.0f + gc*(1.0f - sig));
                        partial[t] = silu*uc*dh[t];
                        const float dhw = dh[t]*smem.route_weight[slot][c*CHUNK + t];
                        dz_gate[t] = gate_active ? dhw*uc*dsilu : 0.0f;
                        dz_up[t] = up_active ? dhw*silu : 0.0f;
                    }
                    uint32_t i0 = item.tile*UMMA_M + epi_warp*32;
                    stage_store(dz_gate, [&](uint32_t r) { return dz_pool + static_cast<uint64_t>(pg(i0 + r))*pool_stride + pool_row + c*CHUNK; });
                    stage_store(dz_up, [&](uint32_t r) { return dz_pool + static_cast<uint64_t>(pg(i0 + r) + kGran)*pool_stride + pool_row + c*CHUNK; });
                    SUPERMEOW_PROF_SEG(1, 1);
                    #pragma unroll
                    for (uint32_t step = 0; step < 5; ++step) {
                        uint32_t off = 16u >> step;
                        const bool upper = (lane & off) != 0;
                        #pragma unroll
                        for (uint32_t t = 0; t < off; ++t) {
                            const float send = upper ? partial[t] : partial[t + off];
                            const float keep = upper ? partial[t + off] : partial[t];
                            partial[t] = keep + __shfl_xor_sync(0xffffffff, send, off);
                        }
                    }
                    const float mine = partial[0];
                    if (!is_shared)
                        atomicAdd(bw.block_dtopk + static_cast<uint64_t>(item.block)*BLOCK_M + c*CHUNK + lane, mine);
                    SUPERMEOW_PROF_SEG(1, 2);
                }
                fence_proxy_async_global();
                epi_sync();
                if (epi_tid == 0) {
                    __threadfence();
                    ptx::atomic_add_rel(bw.block_a2_done + item.block, 1u);
                    ptx::atomic_add_rel(bw.expert_done + (is_shared ? kNumExpertsPerRank : item.expert), 1u);
                    smem.slot_empty[slot].arrive();
                }
                SUPERMEOW_PROF_SEG(1, 3);
            } else if (item.kind == kKindDx) {
                uint32_t accum;
                begin_task(accum);
                SUPERMEOW_PROF_SEG_BEGIN();
                #pragma unroll
                for (uint32_t c=0; c < kNumChunksWide; ++c) {
                    float v[CHUNK];
                    load_cols(accum, c*CHUNK, v);
                    if (c == kNumChunksWide - 1)
                        release_tmem(accum);
                    stage_store(v, [&](uint32_t r) -> nv_bfloat16* {
                        uint32_t t = epi_warp*32 + r;
                        if (t >= item.valid_m)
                            return nullptr;
                        return sym_buffer.map(bw.dx_slot_buffer.get_rank_buffer(meta_topk(slot, t))
                                                  .get_data_buffer(meta_token(slot, t)).template get_base_ptr<nv_bfloat16>(),
                                              meta_rank(slot, t)) + item.tile*UMMA_N_WIDE + c*CHUNK;
                    });
                }
                SUPERMEOW_PROF_SEG(2, 0);
                if (item.tile == 0 && !is_shared && epi_tid < item.valid_m) {
                    auto* remote_dw = sym_buffer.map(
                        bw.dtopk_weight_slot_buffer.get_rank_buffer(meta_topk(slot, epi_tid))
                            .get_data_buffer(meta_token(slot, epi_tid)).template get_base_ptr<float>(),
                        meta_rank(slot, epi_tid));
                    *remote_dw = __ldcg(bw.block_dtopk + static_cast<uint64_t>(item.block)*BLOCK_M + epi_tid);
                }
                SUPERMEOW_PROF_SEG(2, 1);
                finish_item(slot);
                SUPERMEOW_PROF_SEG(2, 2);
            } else {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                uint32_t mt = is_dw2 ? item.tile / kNumG2TilesWide : (item.tile - kNumDW2Tiles) / kNumHTilesWide;
                uint32_t nt = is_dw2 ? item.tile % kNumG2TilesWide : (item.tile - kNumDW2Tiles) % kNumHTilesWide;
                uint32_t p = is_shared ? item.expert - kNumExpertsPerRank : 0u;
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t m = mt*DW_M + h*UMMA_M + row;
                    uint32_t m1 = kDwNatural ? (m / (2*kGran))*kGran + (m % kGran) + ((m % (2*kGran)) >= kGran ? kIntermediateHidden : 0u) : m;
                    dw_t* dst;
                    if (is_dw2) {
                        dst = is_shared
                            ? shared_dw2_weights + static_cast<uint64_t>(m)*(kIntermediateHidden*kNumPasses) + p*kIntermediateHidden + nt*UMMA_N_WIDE
                            : dw2_weights + (static_cast<uint64_t>(item.expert)*kHidden + m)*kIntermediateHidden + nt*UMMA_N_WIDE;
                    } else {
                        dst = is_shared
                            ? shared_dw1_weights + (static_cast<uint64_t>(p)*I2 + m1)*kHidden + nt*UMMA_N_WIDE
                            : dw1_weights + (static_cast<uint64_t>(item.expert)*I2 + m1)*kHidden + nt*UMMA_N_WIDE;
                    }
                    uint32_t accum;
                    begin_task(accum);
                    #pragma unroll
                    for (uint32_t c=0; c < kNumChunksWide; ++c) {
                        float v[CHUNK];
                        if (item.num_k_blocks == 0) {
                            #pragma unroll
                            for (uint32_t t = 0; t < CHUNK; ++t)
                                v[t] = 0.0f;
                        } else {
                            load_cols(accum, c*CHUNK, v);
                        }
                        if (c == kNumChunksWide - 1)
                            release_tmem(accum);
                        if constexpr (cute::is_same_v<dw_t, float>)
                            store_row_f32(dst + c*CHUNK, v);
                        else
                            store_row_bf16(reinterpret_cast<nv_bfloat16*>(dst + c*CHUNK), v);
                    }
                }
                finish_item(slot);
            }
            SUPERMEOW_PROF(p_item[prof_kind] += clock64() - prof_t_item;)
            slot = (slot + 1) % kNumSlots;
            slot_phase^=(slot == 0);
        }
        SUPERMEOW_PROF(if (prof_print && epi_tid == 0)
            printf("DGPROF r=%u sm=%u role=epi total=%lld item_z=%llu item_dz=%llu item_dx=%llu item_dw=%llu tfull_z=%llu tfull_dz=%llu tfull_dx=%llu tfull_dw=%llu slot_wait=%llu\n",
                   prof_rank, sm_idx, clock64() - prof_t_main, p_item[kKindZ], p_item[kKindDz], p_item[kKindDx], p_item[kKindDw],
                   p_tfull[kKindZ], p_tfull[kKindDz], p_tfull[kKindDx], p_tfull[kKindDw], p_slotw);
            if (prof_print && epi_tid == 0)
            printf("DGPROF r=%u sm=%u role=episeg z_tmem=%llu z_zst=%llu z_hw=%llu z_tail=%llu dz_tmem=%llu dz_math=%llu dz_dtopk=%llu dz_tail=%llu dx_store=%llu dx_dtopk=%llu dx_tail=%llu\n",
                   prof_rank, sm_idx, p_seg[0][0], p_seg[0][1], p_seg[0][2], p_seg[0][3], p_seg[1][0], p_seg[1][1], p_seg[1][2], p_seg[1][3],
                   p_seg[2][0], p_seg[2][1], p_seg[2][2]);)
    }

    SUPERMEOW_PROF(const long long prof_t_roles_end = clock64();)
    __threadfence_system();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 199>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
    SUPERMEOW_PROF(const long long prof_t_barrier_end = clock64();
            if (sm_idx == 0 && tid == 0) g_dg_mega_bwd_prof_launch = prof_launch + 1;)
    if (warp == 0)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);

    for (uint32_t i=global_tid; i < kNumExperts; i += kNumGlobalThreads) {
        *workspace.get_expert_send_count_ptr(i) = 0;
        *workspace.get_expert_recv_count_ptr(i / kNumExpertsPerRank, i % kNumExpertsPerRank) = 0;
    }
    for (uint32_t i=global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;

    constexpr uint32_t kHiddenVec = kHidden / 8;
    for (uint64_t linear = static_cast<uint64_t>(sm_idx)*kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens)*kHiddenVec;
         linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
        uint32_t token = linear / kHiddenVec, k8 = linear % kHiddenVec;
        float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        auto accumulate = [&](const nv_bfloat16* slot) {
            const uint4 raw = *reinterpret_cast<const uint4*>(slot + k8*8);
            auto* h = reinterpret_cast<const nv_bfloat162*>(&raw);
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(h[i]);
                sum[i*2] += f.x, sum[i*2 + 1] += f.y;
            }
        };
        #pragma unroll
        for (uint32_t topk = 0; topk < kNumTopk; ++topk) {
            auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token*kNumTopk + topk];
            if (e >= 0)
                accumulate(bw.dx_slot_buffer.get_rank_buffer(topk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>());
        }
        if constexpr (kHasShared)
            accumulate(bw.dx_slot_buffer.get_rank_buffer(kNumTopk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>());
        uint4 out;
        auto* out_h = reinterpret_cast<nv_bfloat162*>(&out);
        #pragma unroll
        for (uint32_t i=0; i < 4; ++i)
            out_h[i] = __floats2bfloat162_rn(sum[i*2], sum[i*2 + 1]);
        reinterpret_cast<uint4*>(static_cast<nv_bfloat16*>(dx))[linear] = out;
    }

    for (uint64_t linear = static_cast<uint64_t>(sm_idx)*kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens)*kNumTopk;
         linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
        uint32_t token = linear / kNumTopk, topk = linear % kNumTopk;
        auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[linear];
        dtopk_weights[linear] = e < 0 ? 0.0f : *bw.dtopk_weight_slot_buffer.get_rank_buffer(topk)
            .get_data_buffer(token).template get_base_ptr<float>();
    }
    SUPERMEOW_PROF(if (prof_print && tid == 0)
        printf("DGPROF r=%u sm=%u role=phase prologue=%lld main=%lld final_barrier=%lld tail=%lld total=%lld\n",
               prof_rank, sm_idx, prof_t_main - prof_t_start, prof_t_roles_end - prof_t_main,
               prof_t_barrier_end - prof_t_roles_end, clock64() - prof_t_barrier_end, clock64() - prof_t_start);)
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only support sm_100f");
#endif
}
}
