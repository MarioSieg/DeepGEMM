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
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumPasses = kHasShared ? kNumSharedExperts : 1,
    uint32_t kGran = 8
>
CUTLASS_GLOBAL __launch_bounds__(256, 1) void
sm100_bf16_mega_moe_backward_impl(
    void* __restrict__ dx,
    float* __restrict__ dw1_weights,
    float* __restrict__ dw2_weights,
    float* __restrict__ dtopk_weights,
    float* __restrict__ shared_dw1_weights,
    float* __restrict__ shared_dw2_weights,
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using bf16_t = cutlass::bfloat16_t;
    using BlockDesc = layout::MegaMoEBackwardBuffer::BlockDesc;

    constexpr uint32_t kNumThreads = 256;
    constexpr uint32_t BLOCK_M = layout::MegaMoEBackwardBuffer::kBlockM;
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = 128;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t CHUNK = 32;
    constexpr uint32_t kNumChunks = UMMA_N / CHUNK;
    constexpr uint32_t kNumEpilogueThreads = 128;
    constexpr uint32_t kEpilogueBarrierIdx = 1;
    constexpr uint32_t kGatherBarrierIdx = 2;
    constexpr uint32_t kNumAccumStages = 4;
    constexpr uint32_t kNumTmemCols = kNumAccumStages*UMMA_N;
    constexpr uint32_t I2 = kIntermediateHidden<<1;
    constexpr uint32_t kNumG1Tiles = I2 / UMMA_M;
    constexpr uint32_t kNumG2Tiles = kIntermediateHidden / UMMA_M;
    constexpr uint32_t kNumHTiles = kHidden / UMMA_M;
    constexpr uint32_t kNumKBlocksH = kHidden / BLOCK_K;
    constexpr uint32_t kNumKBlocksI2 = I2 / BLOCK_K;
    constexpr uint32_t kNumDW2Tiles = kNumHTiles*kNumG2Tiles;
    constexpr uint32_t kNumDW1Tiles = kNumG1Tiles*kNumHTiles;
    constexpr uint32_t kNumTilesPerExpert = kNumDW2Tiles + kNumDW1Tiles;
    constexpr uint32_t kNumSharedSlots = kHasShared ? kNumPasses : 0;
    constexpr uint32_t kNumExpertSlots = kNumExpertsPerRank + kNumSharedSlots;
    constexpr uint32_t kDxStageStride = CHUNK + 8;
    constexpr uint32_t kNumVecPerRow = (kHidden<<1)>>4;
    constexpr uint32_t kVecPerLane = kNumVecPerRow>>5;
    constexpr uint32_t kVecUnroll = (7&kVecPerLane) == 0 ? 8 : (3&kVecPerLane) == 0 ? 4 : (1&kVecPerLane) == 0 ? 2 : 1;

    static_assert(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    static_assert(kHidden % UMMA_M == 0 && kIntermediateHidden % UMMA_M == 0, "Invalid hidden sizes");
    static_assert(kIntermediateHidden % kGran == 0, "Invalid intermediate hidden for gate/up interleaving");
    static_assert(kNumTopk <= 32, "Invalid number of topk");
    static_assert(kNumSMs > 1, "Invalid SM count");
    static_assert(BLOCK_M == UMMA_N && BLOCK_M % BLOCK_K == 0 && BLOCK_M % 4 == 0 && BLOCK_M <= kNumEpilogueThreads, "Invalid token block");
    static_assert((31&kNumVecPerRow) == 0, "Invalid hidden for the gather");
    static_assert(kNumTmemCols <= 512, "Invalid TMEM usage");
    static_assert(kNumExpertSlots <= layout::MegaMoEBackwardBuffer::kMaxExpertSlots, "Too many experts per rank");
    static_assert(kNumStages >= 2, "Invalid number of stages");

    const uint32_t tid = threadIdx.x;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t warp = tid>>5;
    const uint32_t lane = tid&31;
    constexpr uint32_t kNumGlobalThreads = kNumSMs*kNumThreads;
    const uint32_t global_tid = sm_idx*kNumThreads + tid;

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
    const uint64_t pool_stride = bw.num_pool_rows;

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads)
        *workspace.get_expert_send_count_ptr(i)=0;
    for (uint32_t i = global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i)=0;
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 101>( workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); }, false, true);

    for (uint32_t idx = global_tid; idx < num_tokens*kNumTopk; idx += kNumGlobalThreads) {
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
        ptx::atomic_add_sys( sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert), dst_rank), static_cast<uint64_t>(count));
    }
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 103>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });

    if (sm_idx == 0 && tid == 0) {
        uint32_t pool_base=0, meta_base=0, num_blocks=0;
        for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
            const auto valid_m = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            const auto blocks_e = math::ceil_div(valid_m, BLOCK_M);
            bw.expert_pool_base[e] = pool_base;
            bw.expert_num_blocks[e] = blocks_e;
            bw.expert_done[e]=0;
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
        if constexpr (kHasShared) {
            const auto blocks_s = math::ceil_div(num_tokens, BLOCK_M);
            for (uint32_t p=0; p < kNumPasses; ++p) {
                bw.expert_pool_base[kNumExpertsPerRank + p] = pool_base + p*bw.shared_region_stride;
                bw.expert_num_blocks[kNumExpertsPerRank + p] = blocks_s;
                bw.expert_done[kNumExpertsPerRank + p]=0;
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
        *bw.num_items = num_blocks + kNumExpertSlots*kNumTilesPerExpert;
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
        const auto valid_m_e = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
        const auto pool_begin_e = s_pool_begin[e];
        for (uint32_t local_pos = global_tid; local_pos < valid_m_e; local_pos += kNumGlobalThreads) {
            uint32_t r=0;
            #pragma unroll
            for (uint32_t rr=0; rr < kNumRanks; ++rr)
                if (s_rank_prefix[e][rr] <= local_pos) r = rr;
            const auto slot = local_pos - s_rank_prefix[e][r];
            const auto token_topk = *workspace.get_src_token_topk_idx_ptr(e, r, slot);
            const auto src_token = token_topk / kNumTopk;
            const auto src_topk = token_topk % kNumTopk;
            *workspace.get_token_src_metadata_ptr(pool_begin_e + local_pos) =
                layout::TokenSrcMetadata(r, src_token, src_topk);
        }
    }
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });

    struct Item {
        uint32_t kind;
        uint32_t expert;
        uint32_t pool_begin;
        uint32_t x_begin;
        uint32_t valid_m;
        uint32_t tile;
        uint32_t num_k_blocks;
    };
    struct SharedStorage {
        alignas(1024) bf16_t a[kNumStages][UMMA_M*BLOCK_K];
        alignas(1024) bf16_t b[kNumStages][UMMA_N*BLOCK_K];
        float route_weight[2][BLOCK_M];
        uint32_t src_rank[2][BLOCK_M];
        uint32_t src_token[2][BLOCK_M];
        uint32_t src_topk[2][BLOCK_M];
        Item item[2];
        uint32_t item_valid[2];
        float dtopk_acc[BLOCK_M];
        alignas(16) bf16_t dx_stage[4][CHUNK][kDxStageStride];
        Barrier slot_full[2];
        Barrier slot_empty[2];
        Barrier full[kNumStages];
        Barrier empty[kNumStages];
        Barrier tmem_full[kNumAccumStages];
        Barrier tmem_empty[kNumAccumStages];
        Barrier dz_ready;
        uint32_t tmem_ptr;
    };
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    constexpr uint32_t kStageABytes = sizeof(smem.a[0]);
    constexpr uint32_t kStageBBytes = sizeof(smem.b[0]);

    if (warp == 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i=0; i < 2; ++i) {
                smem.slot_full[i].init(1);
                smem.slot_empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i=0; i < kNumStages; ++i) {
                smem.full[i].init(1);
                smem.empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i=0; i < kNumAccumStages; ++i) {
                smem.tmem_full[i].init(1);
                smem.tmem_empty[i].init(kNumEpilogueThreads);
            }
            smem.dz_ready.init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp == 3) {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    __syncthreads();

    auto* z_scratch_sm = reinterpret_cast<float*>(math::advance_ptr(bw.z_scratch, sm_idx*bw.z_bytes_per_sm));

    const auto pg = [](const uint32_t& j) -> uint32_t {
        return (j / kGran)*2*kGran + (j % kGran);
    };
    const auto gather_sync = [&]() { cutlass::arch::NamedBarrier::sync(64, kGatherBarrierIdx); };
    const auto gather_rows = [&](const uint32_t& slot, const uint32_t& pool_begin, const uint32_t& valid_m, const uint32_t& parity) {
        for (uint32_t row = parity; row < BLOCK_M; row += 4) {
            const uint32_t rows[2] = {row, row + 2};
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
                const auto r_rank = smem.src_rank[slot][rows[h]];
                const auto r_token = smem.src_token[slot][rows[h]];
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

    if (warp == 0) {
        uint32_t slot=0, slot_phase=0;
        for (;;) {
            uint32_t item_idx=0;
            if (lane == 0)
                item_idx = ptx::atomic_add(bw.next_item, 1u);
            item_idx = __shfl_sync(0xffffffff, item_idx, 0);
            const uint32_t num_blocks = *bw.num_blocks;
            const bool done = item_idx >= *bw.num_items;
            Item cur{};
            if (!done && item_idx >= num_blocks) {
                const uint32_t t = item_idx - num_blocks;
                cur.kind = 1;
                cur.expert = t / kNumTilesPerExpert;
                cur.tile = t % kNumTilesPerExpert;
                cur.num_k_blocks = bw.expert_num_blocks[cur.expert]*(BLOCK_M / BLOCK_K);
                if (cur.num_k_blocks == 0)
                    continue;
                cur.pool_begin = bw.expert_pool_base[cur.expert];
                cur.x_begin = bw.expert_pool_base[cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank];
                const uint32_t done_slot = cur.expert < kNumExpertsPerRank ? cur.expert : kNumExpertsPerRank;
                const uint32_t target = bw.expert_num_blocks[done_slot];
                if (lane == 0) {
                    while (ptx::ld_acq(bw.expert_done + done_slot) < target)
                        __nanosleep(256);
                }
                __syncwarp();
                fence_proxy_async_global();
            }
            smem.slot_empty[slot].wait(slot_phase^1);
            if (!done && item_idx < num_blocks) {
                const auto bd = bw.block_desc[item_idx];
                const bool is_shared = bd.local_expert >= kNumExpertsPerRank;
                cur.kind=0;
                cur.expert = bd.local_expert;
                cur.pool_begin = bd.pool_begin;
                cur.x_begin = bd.pool_begin;
                cur.valid_m = bd.valid_m;
                for (uint32_t r = lane; r < BLOCK_M; r += 32) {
                    uint32_t src_rank = sym_buffer.rank_idx, src_token=0, src_topk = kNumTopk;
                    float weight=0.0f;
                    if (r < bd.valid_m) {
                        if (is_shared) {
                            src_token = bd.meta_begin + r;
                            weight = 1.0f;
                        } else {
                            const auto meta = *workspace.get_token_src_metadata_ptr(bd.meta_begin + r);
                            src_rank = meta.rank_idx;
                            src_token = meta.token_idx;
                            src_topk = meta.topk_idx;
                            weight = *sym_buffer.map(
                                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token*kNumTopk + src_topk, src_rank);
                        }
                    }
                    smem.route_weight[slot][r] = weight;
                    smem.src_rank[slot][r] = src_rank;
                    smem.src_token[slot][r] = src_token;
                    smem.src_topk[slot][r] = src_topk;
                }
            }
            if (lane == 0) {
                smem.item[slot] = cur;
                smem.item_valid[slot] = done ? 0u : 1u;
            }
            __syncwarp();
            gather_sync();
            if (done) {
                if (lane == 0)
                    smem.slot_full[slot].arrive();
                break;
            }
            if (cur.kind == 0)
                gather_rows(slot, cur.pool_begin, cur.valid_m, 0);
            gather_sync();
            if (lane == 0)
                smem.slot_full[slot].arrive();
            slot^=1;
            slot_phase^=(slot == 0);
        }
    } else if (warp == 3) {
        uint32_t slot=0;
        for (;;) {
            gather_sync();
            if (!smem.item_valid[slot])
                break;
            if (smem.item[slot].kind == 0)
                gather_rows(slot, smem.item[slot].pool_begin, smem.item[slot].valid_m, 1);
            gather_sync();
            slot^=1;
        }
    } else if (warp == 1) {
        uint32_t stage=0, phase=0;
        uint32_t slot=0, slot_phase=0;
        uint32_t dz_phase=0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase^=(stage == 0);
        };
        const auto issue = [&](const uint32_t& num_bytes) {
            smem.full[stage].arrive_and_expect_tx(num_bytes);
            advance();
        };
        constexpr uint32_t kStageBytes = UMMA_M*BLOCK_K*2 + UMMA_N*BLOCK_K*2;
        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.item_valid[slot])
                break;
            const auto item = smem.item[slot];
            if (item.kind == 0) {
                const bool is_shared = item.expert >= kNumExpertsPerRank;
                const auto* w1k = is_shared ? &tensor_map_shared_w1_k : &tensor_map_w1_k;
                const auto* w1mn = is_shared ? &tensor_map_shared_w1_mn : &tensor_map_w1_mn;
                const auto* w2mn = is_shared ? &tensor_map_shared_w2_mn : &tensor_map_w2_mn;
                const uint32_t num_passes = is_shared ? kNumPasses : 1u;
                for (uint32_t p=0; p < num_passes; ++p) {
                    const uint32_t w1_rows = is_shared ? p*I2 : item.expert*I2;
                    const uint32_t w2_rows = is_shared ? 0u : item.expert*kHidden;
                    const uint32_t w2_cols = is_shared ? p*kIntermediateHidden : 0u;
                    for (uint32_t mt=0; mt < kNumG1Tiles; ++mt) {
                        for (uint32_t kb=0; kb < kNumKBlocksH; ++kb) {
                            smem.empty[stage].wait(phase^1);
                            if (cute::elect_one_sync()) {
                                tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb*BLOCK_K, w1_rows + mt*UMMA_M);
                                tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_x_k, &smem.full[stage], smem.b[stage], kb*BLOCK_K, item.pool_begin);
                                issue(kStageBytes);
                            } else {
                                advance();
                            }
                            __syncwarp();
                        }
                    }
                    for (uint32_t mt=0; mt < kNumG2Tiles; ++mt) {
                        for (uint32_t kb=0; kb < kNumKBlocksH; ++kb) {
                            smem.empty[stage].wait(phase^1);
                            if (cute::elect_one_sync()) {
                                tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(w2mn, &smem.full[stage], smem.a[stage], w2_cols + mt*UMMA_M, w2_rows + kb*BLOCK_K);
                                tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_dy_k, &smem.full[stage], smem.b[stage], kb*BLOCK_K, item.pool_begin);
                                issue(kStageBytes);
                            } else {
                                advance();
                            }
                            __syncwarp();
                        }
                    }
                }
                smem.dz_ready.wait(dz_phase);
                dz_phase^=1;
                const uint32_t w1_rows_g4 = is_shared ? 0u : item.expert*I2;
                for (uint32_t mt=0; mt < kNumHTiles; ++mt) {
                    for (uint32_t kb=0; kb < num_passes*kNumKBlocksI2; ++kb) {
                        smem.empty[stage].wait(phase^1);
                        if (cute::elect_one_sync()) {
                            tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(w1mn, &smem.full[stage], smem.a[stage], mt*UMMA_M, w1_rows_g4 + kb*BLOCK_K);
                            tma::copy<UMMA_N, BLOCK_K, 128, bf16_t>(&tensor_map_dz_mn, &smem.full[stage], smem.b[stage],
                                                                    item.pool_begin + (kb / kNumKBlocksI2)*bw.shared_region_stride, (kb % kNumKBlocksI2)*BLOCK_K);
                            issue(kStageBytes);
                        } else {
                            advance();
                        }
                        __syncwarp();
                    }
                }
            } else {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                const uint32_t mt = is_dw2 ? item.tile / kNumG2Tiles : (item.tile - kNumDW2Tiles) / kNumHTiles;
                const uint32_t nt = is_dw2 ? item.tile % kNumG2Tiles : (item.tile - kNumDW2Tiles) % kNumHTiles;
                for (uint32_t kb=0; kb < item.num_k_blocks; ++kb) {
                    smem.empty[stage].wait(phase^1);
                    if (cute::elect_one_sync()) {
                        if (is_dw2) {
                            tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(&tensor_map_dy_mn, &smem.full[stage], smem.a[stage], mt*UMMA_M, item.x_begin + kb*BLOCK_K);
                            tma::copy<BLOCK_K, UMMA_N, 128, bf16_t>(&tensor_map_hw_k, &smem.full[stage], smem.b[stage], item.pool_begin + kb*BLOCK_K, nt*UMMA_N);
                        } else {
                            tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(&tensor_map_dz_k, &smem.full[stage], smem.a[stage], item.pool_begin + kb*BLOCK_K, mt*UMMA_M);
                            tma::copy<UMMA_N, BLOCK_K, 128, bf16_t>(&tensor_map_x_mn, &smem.full[stage], smem.b[stage], nt*UMMA_N, item.x_begin + kb*BLOCK_K);
                        }
                        issue(kStageBytes);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            }
            slot^=1;
            slot_phase^=(slot == 0);
        }
    } else if (warp == 2) {
        uint32_t stage=0, phase=0;
        uint32_t slot=0, slot_phase=0;
        uint32_t task_idx=0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase^=(stage == 0);
        };

        const auto idesc_kk = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        const auto idesc_mk = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::MN, cute::UMMA::Major::K>();
        const auto idesc_km = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::MN>();
        const auto idesc_mm = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::MN, cute::UMMA::Major::MN>();

        auto a_k = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto a_mn = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto b_k = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_N, BLOCK_K, 128>(smem.b[0], 0, 0);
        auto b_mn = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_N, BLOCK_K, 128>(smem.b[0], 0, 0);
        const uint32_t a_k_lo = a_k.lo, a_mn_lo = a_mn.lo, b_k_lo = b_k.lo, b_mn_lo = b_mn.lo;

        const auto run_task = [&](auto& a_desc, auto& b_desc, const uint32_t& a_lo, const uint32_t& b_lo, // exec comp pipeleune part
                                  auto advance_a, auto advance_b,
                                  const cute::UMMA::InstrDescriptor& idesc,
                                  const uint32_t& num_k_blocks) {
            const auto accum = task_idx % kNumAccumStages;
            const auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            smem.tmem_empty[accum].wait(accum_phase^1);
            ptx::tcgen05_after_thread_sync();
            const auto runtime_idesc = cute::UMMA::make_runtime_instr_desc(idesc);
            for (uint32_t kb=0; kb < num_k_blocks; ++kb) {
                smem.full[stage].wait(phase);
                ptx::tcgen05_after_thread_sync();
                const uint32_t a_base = a_lo + stage*(kStageABytes / 16);
                const uint32_t b_base = b_lo + stage*(kStageBBytes / 16);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k=0; k < BLOCK_K / UMMA_K; ++k) {
                        a_desc.lo = advance_a(a_base, k*UMMA_K);
                        b_desc.lo = advance_b(b_base, k*UMMA_K);
                        ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, accum*UMMA_N, kb > 0 || k > 0, runtime_idesc);
                    }
                }
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty[stage]));
                if (kb == num_k_blocks - 1)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full[accum]));
                __syncwarp();
                advance();
            }
        };
        const auto adv_a_k = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_M, 128, bf16_t>(base, 0, k); };
        const auto adv_a_mn = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_M, 128, bf16_t>(base, 0, k); };
        const auto adv_b_k = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_N, 128, bf16_t>(base, 0, k); };
        const auto adv_b_mn = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_N, 128, bf16_t>(base, 0, k); };

        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.item_valid[slot])
                break;
            const auto item = smem.item[slot];
            if (item.kind == 0) {
                const uint32_t num_passes = item.expert >= kNumExpertsPerRank ? kNumPasses : 1u;
                for (uint32_t p=0; p < num_passes; ++p) {
                    for (uint32_t mt=0; mt < kNumG1Tiles; ++mt)
                        run_task(a_k, b_k, a_k_lo, b_k_lo, adv_a_k, adv_b_k, idesc_kk, kNumKBlocksH);
                    for (uint32_t mt=0; mt < kNumG2Tiles; ++mt)
                        run_task(a_mn, b_k, a_mn_lo, b_k_lo, adv_a_mn, adv_b_k, idesc_mk, kNumKBlocksH);
                }
                for (uint32_t mt=0; mt < kNumHTiles; ++mt)
                    run_task(a_mn, b_mn, a_mn_lo, b_mn_lo, adv_a_mn, adv_b_mn, idesc_mm, num_passes*kNumKBlocksI2);
            } else if (item.tile < kNumDW2Tiles) {
                run_task(a_mn, b_k, a_mn_lo, b_k_lo, adv_a_mn, adv_b_k, idesc_mk, item.num_k_blocks);
            } else {
                run_task(a_k, b_mn, a_k_lo, b_mn_lo, adv_a_k, adv_b_mn, idesc_km, item.num_k_blocks);
            }
            slot^=1;
            slot_phase^=(slot == 0);
        }
    } else if (warp >= 4) {
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
        const uint32_t epi_warp = warp - 4;
        const uint32_t epi_tid = tid - 128;
        const uint32_t row = epi_warp*32 + lane;
        uint32_t slot=0, slot_phase=0;
        uint32_t task_idx=0;

        const auto epi_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, kEpilogueBarrierIdx); };
        const auto begin_task = [&](uint32_t& accum) {
            accum = task_idx % kNumAccumStages;
            const auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            smem.tmem_full[accum].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
        };
        const auto load_cols = [&](const uint32_t& accum, const uint32_t& col, float* v) {
            ptx::tmem_load_32dp32b<32>(accum*UMMA_N + col, reinterpret_cast<uint32_t*>(v));
            cutlass::arch::fence_view_async_tmem_load();
        };
        const auto release_tmem = [&](const uint32_t& accum) {
            ptx::tcgen05_before_thread_sync();
            smem.tmem_empty[accum].arrive();
        };
        const auto pack2 = [](const float& a, const float& b) {
            const auto h = __floats2bfloat162_rn(a, b);
            return *reinterpret_cast<const uint32_t*>(&h);
        };
        const auto store_row_bf16 = [&](nv_bfloat16* dst, const float* v) {
            auto* dst4 = reinterpret_cast<uint4*>(dst);
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i)
                dst4[i] = make_uint4(pack2(v[(i<<3) + 0], v[(i<<3) + 1]), pack2(v[(i<<3) + 2], v[(i<<3) + 3]), pack2(v[(i<<3) + 4], v[(i<<3) + 5]), pack2(v[(i<<3) + 6], v[(i<<3) + 7]));
        };
        const auto store_row_f32 = [](float* dst, const float* v) {
            auto* dst4 = reinterpret_cast<float4*>(dst);
            #pragma unroll
            for (uint32_t i=0; i < 8; ++i)
                dst4[i] = make_float4(v[(i<<2)], v[(i<<2) + 1], v[(i<<2) + 2], v[(i<<2) + 3]);
        };
        const auto load_row_f32 = [](const float* src, float* v) {
            const auto* src4 = reinterpret_cast<const float4*>(src);
            #pragma unroll
            for (uint32_t i=0; i < 8; ++i) {
                const float4 q = src4[i];
                v[(i<<2)] = q.x, v[(i<<2) + 1] = q.y, v[(i<<2) + 2] = q.z, v[(i<<2) + 3] = q.w;
            }
        };
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

        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.item_valid[slot]) break;
            const auto item = smem.item[slot];
            if (item.kind == 0) {
                const bool is_shared = item.expert >= kNumExpertsPerRank;
                const uint32_t valid_m = item.valid_m;
                if (epi_tid < BLOCK_M)
                    smem.dtopk_acc[epi_tid]=0.0f;
                epi_sync();

                const uint32_t num_passes = is_shared ? kNumPasses : 1u;
                for (uint32_t p=0; p < num_passes; ++p) {
                    const uint32_t pool_row = item.pool_begin + p*bw.shared_region_stride;
                    for (uint32_t mt=0; mt < kNumG1Tiles; ++mt) {
                        uint32_t accum;
                        begin_task(accum);
                        const uint32_t prow = mt*UMMA_M + row;
                        const bool is_gate = (prow % (2*kGran)) < kGran;
                        const uint32_t j = (prow / (2*kGran))*kGran + (prow % kGran);
                        #pragma unroll
                        for (uint32_t c=0; c < kNumChunks; ++c) {
                            float z[CHUNK];
                            load_cols(accum, c*CHUNK, z);
                            if (c == kNumChunks - 1)
                                release_tmem(accum);
                            store_row_f32(z_scratch_sm + static_cast<uint64_t>(prow)*BLOCK_M + c*CHUNK, z);
                            float hw[CHUNK];
                            #pragma unroll
                            for (uint32_t t=0; t < CHUNK; ++t) {
                                const float partner = __shfl_xor_sync(0xffffffff, z[t], kGran);
                                const float g = clamp_gate(z[t]);
                                const float u = clamp_up(partner);
                                hw[t] = g*sigmoid<kFastMath>(g)*u*smem.route_weight[slot][c*CHUNK + t];
                            }
                            if (is_gate)
                                store_row_bf16(hw_pool + static_cast<uint64_t>(j)*pool_stride + pool_row + c*CHUNK, hw);
                        }
                    }
                    for (uint32_t mt=0; mt < kNumG2Tiles; ++mt) {
                        uint32_t accum;
                        begin_task(accum);
                        const uint32_t i = mt*UMMA_M + row;
                        const uint32_t pgi = pg(i);
                        #pragma unroll
                        for (uint32_t c=0; c < kNumChunks; ++c) {
                            float dh[CHUNK];
                            load_cols(accum, c*CHUNK, dh);
                            if (c == kNumChunks - 1)
                                release_tmem(accum);
                            float g[CHUNK], u[CHUNK];
                            load_row_f32(z_scratch_sm + static_cast<uint64_t>(pgi)*BLOCK_M + c*CHUNK, g);
                            load_row_f32(z_scratch_sm + static_cast<uint64_t>(pgi + kGran)*BLOCK_M + c*CHUNK, u);
                            float partial[CHUNK], dz_gate[CHUNK], dz_up[CHUNK];
                            #pragma unroll
                            for (uint32_t t=0; t < CHUNK; ++t) {
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
                            store_row_bf16(dz_pool + static_cast<uint64_t>(pgi)*pool_stride + pool_row + c*CHUNK, dz_gate);
                            store_row_bf16(dz_pool + static_cast<uint64_t>(pgi + kGran)*pool_stride + pool_row + c*CHUNK, dz_up);
                            #pragma unroll
                            for (uint32_t t=0; t < CHUNK; ++t) {
                                #pragma unroll
                                for (uint32_t off = 16; off > 0; off >>= 1)
                                    partial[t] += __shfl_xor_sync(0xffffffff, partial[t], off);
                            }
                            float mine = partial[0];
                            #pragma unroll
                            for (uint32_t t = 1; t < CHUNK; ++t)
                                mine = lane == t ? partial[t] : mine;
                            atomicAdd(&smem.dtopk_acc[c*CHUNK + lane], mine);
                        }
                    }
                }
                fence_proxy_async_global();
                epi_sync();
                if (epi_tid == 0)
                    smem.dz_ready.arrive();
                if (!is_shared && epi_tid < valid_m) {
                    auto* remote_dw = sym_buffer.map(
                        bw.dtopk_weight_slot_buffer.get_rank_buffer(smem.src_topk[slot][epi_tid])
                            .get_data_buffer(smem.src_token[slot][epi_tid]).template get_base_ptr<float>(),
                        smem.src_rank[slot][epi_tid]);
                    *remote_dw = smem.dtopk_acc[epi_tid];
                }

                for (uint32_t mt=0; mt < kNumHTiles; ++mt) {
                    uint32_t accum;
                    begin_task(accum);
                    auto* stage = &smem.dx_stage[epi_warp][0][0];
                    #pragma unroll
                    for (uint32_t c=0; c < kNumChunks; ++c) {
                        float v[CHUNK];
                        load_cols(accum, c*CHUNK, v);
                        if (c == kNumChunks - 1)
                            release_tmem(accum);
                        #pragma unroll
                        for (uint32_t t=0; t < CHUNK; ++t)
                            stage[t*kDxStageStride + lane] = static_cast<bf16_t>(v[t]);
                        __syncwarp();
                        const uint32_t tok = c*CHUNK + lane;
                        if (tok < valid_m) {
                            const auto* src4 = reinterpret_cast<const uint4*>(stage + lane*kDxStageStride);
                            auto* remote_dx = reinterpret_cast<uint4*>(sym_buffer.map(
                                bw.dx_slot_buffer.get_rank_buffer(smem.src_topk[slot][tok])
                                    .get_data_buffer(smem.src_token[slot][tok]).template get_base_ptr<nv_bfloat16>(),
                                smem.src_rank[slot][tok]) + mt*UMMA_M + epi_warp*CHUNK);
                            #pragma unroll
                            for (uint32_t q=0; q < CHUNK / 8; ++q)
                                remote_dx[q] = src4[q];
                        }
                        __syncwarp();
                    }
                }
                epi_sync();
                if (epi_tid == 0) {
                    __threadfence();
                    ptx::atomic_add_rel(bw.expert_done + item.expert, 1u);
                    smem.slot_empty[slot].arrive();
                }
            } else {
                const bool is_dw2 = item.tile < kNumDW2Tiles;
                const uint32_t mt = is_dw2 ? item.tile / kNumG2Tiles : (item.tile - kNumDW2Tiles) / kNumHTiles;
                const uint32_t nt = is_dw2 ? item.tile % kNumG2Tiles : (item.tile - kNumDW2Tiles) % kNumHTiles;
                const bool is_shared = item.expert >= kNumExpertsPerRank;
                const uint32_t p = is_shared ? item.expert - kNumExpertsPerRank : 0u;
                const uint32_t m = mt*UMMA_M + row;
                float* dst;
                if (is_dw2) {
                    dst = is_shared
                        ? shared_dw2_weights + static_cast<uint64_t>(m)*(kIntermediateHidden*kNumPasses) + p*kIntermediateHidden + nt*UMMA_N
                        : dw2_weights + (static_cast<uint64_t>(item.expert)*kHidden + m)*kIntermediateHidden + nt*UMMA_N;
                } else {
                    dst = is_shared
                        ? shared_dw1_weights + (static_cast<uint64_t>(p)*I2 + m)*kHidden + nt*UMMA_N
                        : dw1_weights + (static_cast<uint64_t>(item.expert)*I2 + m)*kHidden + nt*UMMA_N;
                }
                uint32_t accum;
                begin_task(accum);
                #pragma unroll
                for (uint32_t c=0; c < kNumChunks; ++c) {
                    float v[CHUNK];
                    load_cols(accum, c*CHUNK, v);
                    if (c == kNumChunks - 1)
                        release_tmem(accum);
                    store_row_f32(dst + c*CHUNK, v);
                }
                epi_sync();
                if (epi_tid == 0)
                    smem.slot_empty[slot].arrive();
            }
            slot^=1;
            slot_phase^=(slot == 0);
        }
    }

    __threadfence_system();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 199>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
    if (warp == 0)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads) {
        *workspace.get_expert_send_count_ptr(i)=0;
        *workspace.get_expert_recv_count_ptr(i / kNumExpertsPerRank, i % kNumExpertsPerRank)=0;
    }
    for (uint32_t i = global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i)=0;

    constexpr uint32_t kHiddenVec = kHidden / 8;
    for (uint64_t linear = static_cast<uint64_t>(sm_idx)*kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens)*kHiddenVec;
         linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
        const uint32_t token = linear / kHiddenVec, k8 = linear % kHiddenVec;
        float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const auto accumulate = [&](const nv_bfloat16* slot) {
            const uint4 raw = *reinterpret_cast<const uint4*>(slot + k8*8);
            const auto* h = reinterpret_cast<const nv_bfloat162*>(&raw);
            #pragma unroll
            for (uint32_t i=0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(h[i]);
                sum[i*2] += f.x, sum[i*2 + 1] += f.y;
            }
        };
        #pragma unroll
        for (uint32_t topk=0; topk < kNumTopk; ++topk) {
            const auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token*kNumTopk + topk];
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
        const uint32_t token = linear / kNumTopk, topk = linear % kNumTopk;
        const auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[linear];
        dtopk_weights[linear] = e < 0 ? 0.0f : *bw.dtopk_weight_slot_buffer.get_rank_buffer(topk)
            .get_data_buffer(token).template get_base_ptr<float>();
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only support sm_100f");
#endif
}
}
