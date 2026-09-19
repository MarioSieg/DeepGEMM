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
    const __grid_constant__ cute::TmaDescriptor tensor_map_stage_x,
    const __grid_constant__ cute::TmaDescriptor tensor_map_stage_dy,
    const __grid_constant__ cute::TmaDescriptor tensor_map_hw,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dz_k,
    const __grid_constant__ cute::TmaDescriptor tensor_map_dz_mn
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using bf16_t = cutlass::bfloat16_t;
    using BlockDesc = layout::MegaMoEBackwardBuffer::BlockDesc;

    constexpr uint32_t kNumThreads = 256;
    constexpr uint32_t BLOCK_M = layout::MegaMoEBackwardBuffer::kBlockM;
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t TOK_K = BLOCK_M;
    constexpr uint32_t UMMA_N_W = 128;
    constexpr uint32_t kNumEpilogueThreads = 128;
    constexpr uint32_t kEpilogueBarrierIdx = 1;
    constexpr uint32_t kNumAccumStages = 2;
    constexpr uint32_t kNumTmemCols = kNumAccumStages*UMMA_N_W;
    constexpr uint32_t I2 = 2*kIntermediateHidden;
    constexpr uint32_t kNumG1Tiles = I2 / UMMA_M;
    constexpr uint32_t kNumG2Tiles = kIntermediateHidden / UMMA_M;
    constexpr uint32_t kNumHTiles = kHidden / UMMA_M;
    constexpr uint32_t kNumKBlocksH = kHidden / BLOCK_K;
    constexpr uint32_t kNumKBlocksI2 = I2 / BLOCK_K;

    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kHidden % UMMA_M == 0 && kIntermediateHidden % UMMA_M == 0, "Invalid hidden sizes");
    DG_STATIC_ASSERT(kIntermediateHidden % kGran == 0, "Invalid intermediate hidden for gate/up interleaving");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
    DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
    DG_STATIC_ASSERT(BLOCK_M == 32 && TOK_K == 32, "Token block must match the warp size");
    DG_STATIC_ASSERT(kNumStages >= 2, "Invalid number of stages");

    const uint32_t tid = threadIdx.x;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31;
    constexpr uint32_t kNumGlobalThreads = kNumSMs*kNumThreads;
    const uint32_t global_tid = sm_idx*kNumThreads + tid;

    if (warp == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_w1_k);
        cute::prefetch_tma_descriptor(&tensor_map_w1_mn);
        cute::prefetch_tma_descriptor(&tensor_map_w2_mn);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w1_k);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w1_mn);
        cute::prefetch_tma_descriptor(&tensor_map_shared_w2_mn);
        cute::prefetch_tma_descriptor(&tensor_map_stage_x);
        cute::prefetch_tma_descriptor(&tensor_map_stage_dy);
        cute::prefetch_tma_descriptor(&tensor_map_hw);
        cute::prefetch_tma_descriptor(&tensor_map_dz_k);
        cute::prefetch_tma_descriptor(&tensor_map_dz_mn);
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

    for (uint32_t i = global_tid; i < kNumExperts; i += kNumGlobalThreads)
        *workspace.get_expert_send_count_ptr(i) = 0;
    for (uint32_t i = global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
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
        uint32_t pool_begin = 0, num_blocks = 0;
        for (uint32_t e = 0; e < kNumExpertsPerRank; ++e) {
            const auto valid_m = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            for (uint32_t off = 0; off < valid_m; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = e;
                bw.block_desc[num_blocks].pool_begin = pool_begin + off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, valid_m - off);
                ++num_blocks;
            }
            pool_begin += valid_m;
        }
        if constexpr (kHasShared) {
            for (uint32_t off = 0; off < num_tokens; off += BLOCK_M) {
                bw.block_desc[num_blocks].local_expert = kNumExpertsPerRank;
                bw.block_desc[num_blocks].pool_begin = off;
                bw.block_desc[num_blocks].valid_m = cute::min(BLOCK_M, num_tokens - off);
                ++num_blocks;
            }
        }
        DG_DEVICE_ASSERT(num_blocks <= bw.max_num_blocks);
        *bw.num_blocks = num_blocks;
        *bw.next_block = 0;
    }
    comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
    __shared__ uint32_t s_pool_begin[kNumExpertsPerRank];
    __shared__ uint32_t s_rank_prefix[kNumExpertsPerRank][kNumRanks];
    if (tid == 0) {
        uint32_t pool_begin = 0;
        for (uint32_t e = 0; e < kNumExpertsPerRank; ++e) {
            s_pool_begin[e] = pool_begin;
            uint32_t rank_prefix = 0;
            for (uint32_t r = 0; r < kNumRanks; ++r) {
                s_rank_prefix[e][r] = rank_prefix;
                rank_prefix += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, e));
            }
            pool_begin += rank_prefix;
        }
    }
    __syncthreads();
    for (uint32_t e = 0; e < kNumExpertsPerRank; ++e) {
        const auto valid_m_e = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
        const auto pool_begin_e = s_pool_begin[e];
        for (uint32_t local_pos = global_tid; local_pos < valid_m_e; local_pos += kNumGlobalThreads) {
            uint32_t r = 0;
            #pragma unroll
            for (uint32_t rr = 0; rr < kNumRanks; ++rr)
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

    struct SharedStorage {
        alignas(1024) bf16_t a[kNumStages][UMMA_M*BLOCK_K];
        alignas(1024) bf16_t b[kNumStages][UMMA_N_W*TOK_K];
        float route_weight[2][BLOCK_M];
        uint32_t src_rank[2][BLOCK_M];
        uint32_t src_token[2][BLOCK_M];
        uint32_t src_topk[2][BLOCK_M];
        BlockDesc bd[2];
        uint32_t bd_valid[2];
        float dtopk_acc[BLOCK_M];
        Barrier slot_full[2];
        Barrier slot_empty[2];
        Barrier full[kNumStages];
        Barrier empty[kNumStages];
        Barrier tmem_full[kNumAccumStages];
        Barrier tmem_empty[kNumAccumStages];
        Barrier hw_ready;
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
            for (uint32_t i = 0; i < 2; ++i) {
                smem.slot_full[i].init(1);
                smem.slot_empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                smem.full[i].init(1);
                smem.empty[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccumStages; ++i) {
                smem.tmem_full[i].init(1);
                smem.tmem_empty[i].init(kNumEpilogueThreads);
            }
            smem.hw_ready.init(1);
            smem.dz_ready.init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp == 3) {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    __syncthreads();

    auto* stage_x_sm = reinterpret_cast<nv_bfloat16*>(math::advance_ptr(bw.stage_x, sm_idx*bw.stage_bytes_per_sm));
    auto* stage_dy_sm = reinterpret_cast<nv_bfloat16*>(math::advance_ptr(bw.stage_dy, sm_idx*bw.stage_bytes_per_sm));
    auto* z_scratch_sm = reinterpret_cast<float*>(math::advance_ptr(bw.z_scratch, sm_idx*bw.z_bytes_per_sm));
    auto* hw_scratch_sm = reinterpret_cast<nv_bfloat16*>(math::advance_ptr(bw.hw_scratch, sm_idx*bw.hw_bytes_per_sm));
    auto* dz_scratch_sm = reinterpret_cast<nv_bfloat16*>(math::advance_ptr(bw.dz_scratch, sm_idx*bw.dz_bytes_per_sm));
    const uint32_t stage_row_base = sm_idx*2*BLOCK_M;
    const uint32_t hw_row_base = sm_idx*kIntermediateHidden;
    const uint32_t dz_row_base = sm_idx*kNumPasses*I2;

    const auto pg = [](const uint32_t& j) -> uint32_t {
        return (j / kGran)*2*kGran + (j % kGran);
    };

    if (warp == 0) {
        uint32_t slot = 0, slot_phase = 0;
        for (;;) {
            uint32_t block_idx = 0;
            if (lane == 0)
                block_idx = ptx::atomic_add(bw.next_block, 1u);
            block_idx = __shfl_sync(0xffffffff, block_idx, 0);
            smem.slot_empty[slot].wait(slot_phase ^ 1);

            const bool done = block_idx >= *bw.num_blocks;
            if (!done) {
                const auto cur = bw.block_desc[block_idx];
                const bool is_shared = cur.local_expert >= kNumExpertsPerRank;
                uint32_t src_rank = sym_buffer.rank_idx, src_token = 0, src_topk = kNumTopk;
                float weight = 0.0f;
                if (lane < cur.valid_m) {
                    if (is_shared) {
                        src_token = cur.pool_begin + lane;
                        weight = 1.0f;
                    } else {
                        const auto meta = *workspace.get_token_src_metadata_ptr(cur.pool_begin + lane);
                        src_rank = meta.rank_idx;
                        src_token = meta.token_idx;
                        src_topk = meta.topk_idx;
                        weight = *sym_buffer.map(
                            buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token*kNumTopk + src_topk, src_rank);
                    }
                }
                if (lane == 0) {
                    smem.bd[slot] = cur;
                    smem.bd_valid[slot] = 1;
                }
                smem.route_weight[slot][lane] = weight;
                smem.src_rank[slot][lane] = src_rank;
                smem.src_token[slot][lane] = src_token;
                smem.src_topk[slot][lane] = src_topk;

                constexpr uint32_t kNumVecPerRow = kHidden*2 / 16;
                for (uint32_t row = 0; row < BLOCK_M; ++row) {
                    const auto r_rank = __shfl_sync(0xffffffff, src_rank, row);
                    const auto r_token = __shfl_sync(0xffffffff, src_token, row);
                    auto* dst_x = reinterpret_cast<uint4*>(stage_x_sm + (slot*BLOCK_M + row)*kHidden);
                    auto* dst_dy = reinterpret_cast<uint4*>(stage_dy_sm + (slot*BLOCK_M + row)*kHidden);
                    if (row < cur.valid_m) {
                        const auto* src_x = sym_buffer.map(reinterpret_cast<const uint4*>(
                            buffer.input_token_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
                        const auto* src_dy = sym_buffer.map(reinterpret_cast<const uint4*>(
                            bw.input_dy_buffer.get_data_buffer(r_token).template get_base_ptr<nv_bfloat16>()), r_rank);
                        for (uint32_t i = lane; i < kNumVecPerRow; i += 32) {
                            dst_x[i] = src_x[i];
                            dst_dy[i] = src_dy[i];
                        }
                    } else {
                        for (uint32_t i = lane; i < kNumVecPerRow; i += 32) {
                            dst_x[i] = make_uint4(0, 0, 0, 0);
                            dst_dy[i] = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
                fence_proxy_async_global();
            } else if (lane == 0) {
                smem.bd_valid[slot] = 0;
            }
            __syncwarp();
            if (lane == 0)
                smem.slot_full[slot].arrive();
            if (done)
                break;
            slot ^= 1;
            slot_phase ^= (slot == 0);
        }
    } else if (warp == 1) {
        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t hw_phase = 0, dz_phase = 0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase ^= (stage == 0);
        };
        const auto issue = [&](const uint32_t& num_bytes) {
            smem.full[stage].arrive_and_expect_tx(num_bytes);
            advance();
        };
        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.bd_valid[slot])
                break;
            const auto bd = smem.bd[slot];
            const bool is_shared = bd.local_expert >= kNumExpertsPerRank;
            const auto* w1k = is_shared ? &tensor_map_shared_w1_k : &tensor_map_w1_k;
            const auto* w1mn = is_shared ? &tensor_map_shared_w1_mn : &tensor_map_w1_mn;
            const auto* w2mn = is_shared ? &tensor_map_shared_w2_mn : &tensor_map_w2_mn;
            const uint32_t stage_row = stage_row_base + slot*BLOCK_M;
            const uint32_t num_passes = is_shared ? kNumPasses : 1u;
            for (uint32_t p = 0; p < num_passes; ++p) {
                const uint32_t w1_rows = is_shared ? p*I2 : bd.local_expert*I2;
                const uint32_t w2_rows = is_shared ? 0u : bd.local_expert*kHidden;
                const uint32_t w2_cols = is_shared ? p*kIntermediateHidden : 0u;
                for (uint32_t mt = 0; mt < kNumG1Tiles; ++mt) {
                    for (uint32_t kb = 0; kb < kNumKBlocksH; ++kb) {
                        smem.empty[stage].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            tma::copy<BLOCK_K, UMMA_M, 128, bf16_t>(w1k, &smem.full[stage], smem.a[stage], kb*BLOCK_K, w1_rows + mt*UMMA_M);
                            tma::copy<BLOCK_K, BLOCK_M, 128, bf16_t>(&tensor_map_stage_x, &smem.full[stage], smem.b[stage], kb*BLOCK_K, stage_row);
                            issue(UMMA_M*BLOCK_K*2 + BLOCK_M*BLOCK_K*2);
                        } else {
                            advance();
                        }
                        __syncwarp();
                    }
                }
                for (uint32_t mt = 0; mt < kNumG2Tiles; ++mt) {
                    for (uint32_t kb = 0; kb < kNumKBlocksH; ++kb) {
                        smem.empty[stage].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(w2mn, &smem.full[stage], smem.a[stage], w2_cols + mt*UMMA_M, w2_rows + kb*BLOCK_K);
                            tma::copy<BLOCK_K, BLOCK_M, 128, bf16_t>(&tensor_map_stage_dy, &smem.full[stage], smem.b[stage], kb*BLOCK_K, stage_row);
                            issue(UMMA_M*BLOCK_K*2 + BLOCK_M*BLOCK_K*2);
                        } else {
                            advance();
                        }
                        __syncwarp();
                    }
                }
                smem.hw_ready.wait(hw_phase);
                hw_phase ^= 1;
                for (uint32_t mt = 0; mt < kNumHTiles; ++mt) {
                    for (uint32_t nt = 0; nt < kNumG2Tiles; ++nt) {
                        smem.empty[stage].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            tma::copy<UMMA_M, TOK_K, 128, bf16_t>(&tensor_map_stage_dy, &smem.full[stage], smem.a[stage], mt*UMMA_M, stage_row);
                            tma::copy<TOK_K, UMMA_N_W, 64, bf16_t>(&tensor_map_hw, &smem.full[stage], smem.b[stage], 0, hw_row_base + nt*UMMA_N_W);
                            issue(UMMA_M*TOK_K*2 + UMMA_N_W*TOK_K*2);
                        } else {
                            advance();
                        }
                        __syncwarp();
                    }
                }
                smem.dz_ready.wait(dz_phase);
                dz_phase ^= 1;
                for (uint32_t mt = 0; mt < kNumG1Tiles; ++mt) {
                    for (uint32_t nt = 0; nt < kNumHTiles; ++nt) {
                        smem.empty[stage].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            tma::copy<TOK_K, UMMA_M, 64, bf16_t>(&tensor_map_dz_k, &smem.full[stage], smem.a[stage], 0, dz_row_base + p*I2 + mt*UMMA_M);
                            tma::copy<UMMA_N_W, TOK_K, 128, bf16_t>(&tensor_map_stage_x, &smem.full[stage], smem.b[stage], nt*UMMA_N_W, stage_row);
                            issue(UMMA_M*TOK_K*2 + UMMA_N_W*TOK_K*2);
                        } else {
                            advance();
                        }
                        __syncwarp();
                    }
                }
            }
            const uint32_t w1_rows_g4 = is_shared ? 0u : bd.local_expert*I2;
            for (uint32_t mt = 0; mt < kNumHTiles; ++mt) {
                for (uint32_t kb = 0; kb < num_passes*kNumKBlocksI2; ++kb) {
                    smem.empty[stage].wait(phase ^ 1);
                    if (cute::elect_one_sync()) {
                        tma::copy<UMMA_M, BLOCK_K, 128, bf16_t>(w1mn, &smem.full[stage], smem.a[stage], mt*UMMA_M, w1_rows_g4 + kb*BLOCK_K);
                        tma::copy<BLOCK_M, BLOCK_K, 64, bf16_t>(&tensor_map_dz_mn, &smem.full[stage], smem.b[stage], 0, dz_row_base + kb*BLOCK_K);
                        issue(UMMA_M*BLOCK_K*2 + BLOCK_M*BLOCK_K*2);
                    } else {
                        advance();
                    }
                    __syncwarp();
                }
            }
            slot ^= 1;
            slot_phase ^= (slot == 0);
        }
    } else if (warp == 2) {
        uint32_t stage = 0, phase = 0;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t task_idx = 0;
        const auto advance = [&]() {
            stage = (stage + 1) % kNumStages;
            phase ^= (stage == 0);
        };

        const auto idesc_g1 = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, BLOCK_M, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        const auto idesc_g2 = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, BLOCK_M, cute::UMMA::Major::MN, cute::UMMA::Major::K>();
        const auto idesc_g3 = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N_W, cute::UMMA::Major::MN, cute::UMMA::Major::K>();
        const auto idesc_g5 = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N_W, cute::UMMA::Major::K, cute::UMMA::Major::MN>();
        const auto idesc_g4 = cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, BLOCK_M, cute::UMMA::Major::MN, cute::UMMA::Major::MN>();

        auto a_g1 = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto b_g1 = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128>(smem.b[0], 0, 0);
        auto a_g2 = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto a_g3 = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_M, TOK_K, 128>(smem.a[0], 0, 0);
        auto b_g3 = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_N_W, TOK_K, 64>(smem.b[0], 0, 0);
        auto a_g5 = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_M, TOK_K, 64>(smem.a[0], 0, 0);
        auto b_g5 = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, UMMA_N_W, TOK_K, 128>(smem.b[0], 0, 0);
        auto b_g4 = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, BLOCK_M, BLOCK_K, 64>(smem.b[0], 0, 0);
        const uint32_t a_g1_lo = a_g1.lo, b_g1_lo = b_g1.lo, a_g2_lo = a_g2.lo;
        const uint32_t a_g3_lo = a_g3.lo, b_g3_lo = b_g3.lo, a_g5_lo = a_g5.lo, b_g5_lo = b_g5.lo, b_g4_lo = b_g4.lo;

        const auto run_task = [&](auto& a_desc, auto& b_desc, const uint32_t& a_lo, const uint32_t& b_lo,
                                  auto advance_a, auto advance_b,
                                  const cute::UMMA::InstrDescriptor& idesc,
                                  const uint32_t& num_k_blocks, const uint32_t& block_k) {
            const auto accum = task_idx % kNumAccumStages;
            const auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            smem.tmem_empty[accum].wait(accum_phase ^ 1);
            ptx::tcgen05_after_thread_sync();
            const auto runtime_idesc = cute::UMMA::make_runtime_instr_desc(idesc);
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                smem.full[stage].wait(phase);
                ptx::tcgen05_after_thread_sync();
                const uint32_t a_base = a_lo + stage*(kStageABytes / 16);
                const uint32_t b_base = b_lo + stage*(kStageBBytes / 16);
                if (cute::elect_one_sync()) {
                    for (uint32_t k = 0; k < block_k / UMMA_K; ++k) {
                        a_desc.lo = advance_a(a_base, k*UMMA_K);
                        b_desc.lo = advance_b(b_base, k*UMMA_K);
                        ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, accum*UMMA_N_W, kb > 0 || k > 0, runtime_idesc);
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
        const auto adv_a_g1 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_M, 128, bf16_t>(base, 0, k); };
        const auto adv_b_g1 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_M, 128, bf16_t>(base, 0, k); };
        const auto adv_a_g2 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_M, 128, bf16_t>(base, 0, k); };
        const auto adv_a_g3 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_M, 128, bf16_t>(base, 0, k); };
        const auto adv_b_g3 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_N_W, 64, bf16_t>(base, 0, k); };
        const auto adv_a_g5 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, UMMA_M, 64, bf16_t>(base, 0, k); };
        const auto adv_b_g5 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, UMMA_N_W, 128, bf16_t>(base, 0, k); };
        const auto adv_b_g4 = [](const uint32_t& base, const uint32_t& k) -> uint32_t { return mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::MN, BLOCK_M, 64, bf16_t>(base, 0, k); };

        for (;;) {
            smem.slot_full[slot].wait(slot_phase);
            if (!smem.bd_valid[slot])
                break;
            const uint32_t num_passes = smem.bd[slot].local_expert >= kNumExpertsPerRank ? kNumPasses : 1u;
            for (uint32_t p = 0; p < num_passes; ++p) {
                for (uint32_t mt = 0; mt < kNumG1Tiles; ++mt)
                    run_task(a_g1, b_g1, a_g1_lo, b_g1_lo, adv_a_g1, adv_b_g1, idesc_g1, kNumKBlocksH, BLOCK_K);
                for (uint32_t mt = 0; mt < kNumG2Tiles; ++mt)
                    run_task(a_g2, b_g1, a_g2_lo, b_g1_lo, adv_a_g2, adv_b_g1, idesc_g2, kNumKBlocksH, BLOCK_K);
                for (uint32_t t = 0; t < kNumHTiles*kNumG2Tiles; ++t)
                    run_task(a_g3, b_g3, a_g3_lo, b_g3_lo, adv_a_g3, adv_b_g3, idesc_g3, 1, TOK_K);
                for (uint32_t t = 0; t < kNumG1Tiles*kNumHTiles; ++t)
                    run_task(a_g5, b_g5, a_g5_lo, b_g5_lo, adv_a_g5, adv_b_g5, idesc_g5, 1, TOK_K);
            }
            for (uint32_t mt = 0; mt < kNumHTiles; ++mt)
                run_task(a_g2, b_g4, a_g2_lo, b_g4_lo, adv_a_g2, adv_b_g4, idesc_g4, num_passes*kNumKBlocksI2, BLOCK_K);
            slot ^= 1;
            slot_phase ^= (slot == 0);
        }
    } else if (warp >= 4) {
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
        const uint32_t epi_warp = warp - 4;
        const uint32_t epi_tid = tid - 128;
        const uint32_t row = epi_warp*32 + lane;
        uint32_t slot = 0, slot_phase = 0;
        uint32_t task_idx = 0;

        const auto epi_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, kEpilogueBarrierIdx); };
        const auto begin_task = [&](uint32_t& accum) {
            accum = task_idx % kNumAccumStages;
            const auto accum_phase = (task_idx / kNumAccumStages) & 1;
            ++task_idx;
            smem.tmem_full[accum].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
        };
        const auto load_cols = [&](const uint32_t& accum, const uint32_t& col, float* v) {
            ptx::tmem_load_32dp32b<32>(accum*UMMA_N_W + col, reinterpret_cast<uint32_t*>(v));
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
            for (uint32_t i = 0; i < 4; ++i)
                dst4[i] = make_uint4(pack2(v[i*8 + 0], v[i*8 + 1]), pack2(v[i*8 + 2], v[i*8 + 3]),
                                     pack2(v[i*8 + 4], v[i*8 + 5]), pack2(v[i*8 + 6], v[i*8 + 7]));
        };
        const auto store_row_f32 = [](float* dst, const float* v) {
            auto* dst4 = reinterpret_cast<float4*>(dst);
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                dst4[i] = make_float4(v[i*4], v[i*4 + 1], v[i*4 + 2], v[i*4 + 3]);
        };
        const auto load_row_f32 = [](const float* src, float* v) {
            const auto* src4 = reinterpret_cast<const float4*>(src);
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const float4 q = src4[i];
                v[i*4] = q.x, v[i*4 + 1] = q.y, v[i*4 + 2] = q.z, v[i*4 + 3] = q.w;
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
            if (!smem.bd_valid[slot]) break;
            const auto bd = smem.bd[slot];
            const bool is_shared = bd.local_expert >= kNumExpertsPerRank;
            const uint32_t valid_m = bd.valid_m;
            if (epi_tid < BLOCK_M)
                smem.dtopk_acc[epi_tid] = 0.0f;
            epi_sync();

            const uint32_t num_passes = is_shared ? kNumPasses : 1u;
            for (uint32_t p = 0; p < num_passes; ++p) {
                for (uint32_t mt = 0; mt < kNumG1Tiles; ++mt) {
                    uint32_t accum;
                    begin_task(accum);
                    float z[BLOCK_M];
                    load_cols(accum, 0, z);
                    release_tmem(accum);
                    const uint32_t prow = mt*UMMA_M + row;
                    const bool is_gate = (prow % (2*kGran)) < kGran;
                    const uint32_t j = (prow / (2*kGran))*kGran + (prow % kGran);
                    store_row_f32(z_scratch_sm + static_cast<uint64_t>(prow)*BLOCK_M, z);
                    float hw[BLOCK_M];
                    #pragma unroll
                    for (uint32_t t = 0; t < BLOCK_M; ++t) {
                        const float partner = __shfl_xor_sync(0xffffffff, z[t], kGran);
                        const float g = clamp_gate(z[t]);
                        const float u = clamp_up(partner);
                        hw[t] = g*sigmoid<kFastMath>(g)*u*smem.route_weight[slot][t];
                    }
                    if (is_gate)
                        store_row_bf16(hw_scratch_sm + static_cast<uint64_t>(j)*BLOCK_M, hw);
                }
                fence_proxy_async_global();
                epi_sync();
                if (epi_tid == 0)
                    smem.hw_ready.arrive();

                for (uint32_t mt = 0; mt < kNumG2Tiles; ++mt) {
                    uint32_t accum;
                    begin_task(accum);
                    float dh[BLOCK_M];
                    load_cols(accum, 0, dh);
                    release_tmem(accum);
                    const uint32_t i = mt*UMMA_M + row;
                    const uint32_t pgi = pg(i);
                    float g[BLOCK_M], u[BLOCK_M];
                    load_row_f32(z_scratch_sm + static_cast<uint64_t>(pgi)*BLOCK_M, g);
                    load_row_f32(z_scratch_sm + static_cast<uint64_t>(pgi + kGran)*BLOCK_M, u);
                    float dz_gate[BLOCK_M], dz_up[BLOCK_M], partial[BLOCK_M];
                    #pragma unroll
                    for (uint32_t t = 0; t < BLOCK_M; ++t) {
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
                        const float dhw = dh[t]*smem.route_weight[slot][t];
                        dz_gate[t] = gate_active ? dhw*uc*dsilu : 0.0f;
                        dz_up[t] = up_active ? dhw*silu : 0.0f;
                    }
                    #pragma unroll
                    for (uint32_t t = 0; t < BLOCK_M; ++t) {
                        #pragma unroll
                        for (uint32_t off = 16; off > 0; off >>= 1)
                            partial[t] += __shfl_xor_sync(0xffffffff, partial[t], off);
                    }
                    if (lane < BLOCK_M) {
                        float mine = partial[0];
                        #pragma unroll
                        for (uint32_t t = 1; t < BLOCK_M; ++t)
                            mine = lane == t ? partial[t] : mine;
                        atomicAdd(&smem.dtopk_acc[lane], mine);
                    }
                    store_row_bf16(dz_scratch_sm + (static_cast<uint64_t>(p)*I2 + pgi)*BLOCK_M, dz_gate);
                    store_row_bf16(dz_scratch_sm + (static_cast<uint64_t>(p)*I2 + pgi + kGran)*BLOCK_M, dz_up);
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

                for (uint32_t mt = 0; mt < kNumHTiles; ++mt) {
                    for (uint32_t nt = 0; nt < kNumG2Tiles; ++nt) {
                        uint32_t accum;
                        begin_task(accum);
                        const uint32_t n = mt*UMMA_M + row;
                        float* dst = is_shared
                            ? shared_dw2_weights + static_cast<uint64_t>(n)*(kIntermediateHidden*kNumPasses) + p*kIntermediateHidden + nt*UMMA_N_W
                            : dw2_weights + (static_cast<uint64_t>(bd.local_expert)*kHidden + n)*kIntermediateHidden + nt*UMMA_N_W;
                        #pragma unroll
                        for (uint32_t c = 0; c < UMMA_N_W / 32; ++c) {
                            float v[32];
                            load_cols(accum, c*32, v);
                            if (c == UMMA_N_W / 32 - 1)
                                release_tmem(accum);
                            #pragma unroll
                            for (uint32_t q = 0; q < 8; ++q)
                                atomicAdd(reinterpret_cast<float4*>(dst + c*32 + q*4), make_float4(v[q*4], v[q*4 + 1], v[q*4 + 2], v[q*4 + 3]));
                        }
                    }
                }
                for (uint32_t mt = 0; mt < kNumG1Tiles; ++mt) {
                    for (uint32_t nt = 0; nt < kNumHTiles; ++nt) {
                        uint32_t accum;
                        begin_task(accum);
                        const uint32_t m = mt*UMMA_M + row;
                        float* dst = is_shared
                            ? shared_dw1_weights + (static_cast<uint64_t>(p)*I2 + m)*kHidden + nt*UMMA_N_W
                            : dw1_weights + (static_cast<uint64_t>(bd.local_expert)*I2 + m)*kHidden + nt*UMMA_N_W;
                        #pragma unroll
                        for (uint32_t c = 0; c < UMMA_N_W / 32; ++c) {
                            float v[32];
                            load_cols(accum, c*32, v);
                            if (c == UMMA_N_W / 32 - 1)
                                release_tmem(accum);
                            #pragma unroll
                            for (uint32_t q = 0; q < 8; ++q)
                                atomicAdd(reinterpret_cast<float4*>(dst + c*32 + q*4), make_float4(v[q*4], v[q*4 + 1], v[q*4 + 2], v[q*4 + 3]));
                        }
                    }
                }
            }
            for (uint32_t mt = 0; mt < kNumHTiles; ++mt) {
                uint32_t accum;
                begin_task(accum);
                float v[BLOCK_M];
                load_cols(accum, 0, v);
                release_tmem(accum);
                const uint32_t n = mt*UMMA_M + row;
                for (uint32_t t = 0; t < valid_m; ++t) {
                    auto* remote_dx = sym_buffer.map(
                        bw.dx_slot_buffer.get_rank_buffer(smem.src_topk[slot][t])
                            .get_data_buffer(smem.src_token[slot][t]).template get_base_ptr<nv_bfloat16>(),
                        smem.src_rank[slot][t]);
                    remote_dx[n] = __float2bfloat16_rn(v[t]);
                }
            }
            epi_sync();
            if (epi_tid == 0)
                smem.slot_empty[slot].arrive();
            slot ^= 1;
            slot_phase ^= (slot == 0);
        }
    }

    __threadfence_system();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 199>(
        workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
    if (warp == 0)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);

    for (uint64_t linear = static_cast<uint64_t>(sm_idx)*kNumThreads + tid;
         linear < static_cast<uint64_t>(num_tokens)*kHidden;
         linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
        const uint32_t token = linear / kHidden, k = linear % kHidden;
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t topk = 0; topk < kNumTopk; ++topk) {
            const auto e = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token*kNumTopk + topk];
            if (e >= 0) {
                const auto* slot = bw.dx_slot_buffer.get_rank_buffer(topk)
                    .get_data_buffer(token).template get_base_ptr<nv_bfloat16>();
                sum += __bfloat162float(slot[k]);
            }
        }
        if constexpr (kHasShared) {
            const auto* shared_slot = bw.dx_slot_buffer.get_rank_buffer(kNumTopk)
                .get_data_buffer(token).template get_base_ptr<nv_bfloat16>();
            sum += __bfloat162float(shared_slot[k]);
        }
        static_cast<nv_bfloat16*>(dx)[linear] = __float2bfloat16_rn(sum);
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
