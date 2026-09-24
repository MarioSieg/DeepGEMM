#pragma once

#undef DG_DEVICE_PRINTF
#define DG_DEVICE_PRINTF(...) do {} while (0)

#include <cstdint>
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
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

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
    uint32_t kNumDispatchThreads, uint32_t kNumTMAThreads,
    uint32_t kNumMathThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kL1Natural,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumTMAWarps = kNumTMAThreads / 32,
    uint32_t kNumMathWarps = kNumMathThreads / 32,
    uint32_t kNumMathWarpgroups = kNumMathWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumTMAThreads + kNumMathThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_bf16_mega_moe_impl(void* y,
                        int* cumulative_local_expert_recv_stats,
                        const uint32_t num_tokens,
                        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(kNumDispatchThreads == 128, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid number of TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
    }
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        /*with_sf=*/ false,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;
    using a_dtype_t = cutlass::bfloat16_t;
    using b_dtype_t = cutlass::bfloat16_t;
    using d_dtype_t = cutlass::bfloat16_t;
    constexpr uint32_t WGMMA_N_0 = BLOCK_M <= 128 ? BLOCK_M : 128;
    constexpr uint32_t WGMMA_N_1 = BLOCK_M - WGMMA_N_0;
    using WGMMA_0 = typename mma::sm90::BF16MMASelector<WGMMA_N_0, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using WGMMA_1 = typename mma::sm90::BF16MMASelector<(WGMMA_N_1 > 0 ? WGMMA_N_1 : 8), cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    constexpr uint32_t WGMMA_M = WGMMA_0::M;
    constexpr uint32_t WGMMA_K = WGMMA_0::K;
    constexpr uint32_t kNumAccum = BLOCK_M / 2;
    DG_STATIC_ASSERT(BLOCK_N == WGMMA_M * kNumMathWarpgroups, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0 and BLOCK_M <= 256, "Invalid block M");
    DG_STATIC_ASSERT(STORE_BLOCK_M % 16 == 0 and BLOCK_M % STORE_BLOCK_M == 0, "Invalid store block M");
    DG_STATIC_ASSERT(WGMMA_0::kNumAccum + (WGMMA_N_1 > 0 ? WGMMA_1::kNumAccum : 0) == kNumAccum, "Invalid WGMMA accumulator size");
    constexpr bool kPingPong = BLOCK_M <= 64;
    constexpr uint32_t kNumMHalves = kPingPong ? BLOCK_N / WGMMA_M : 1;
    constexpr uint32_t WG_BLOCK_N = kNumMHalves * WGMMA_M;   // Weight rows per math warpgroup and task
    constexpr uint32_t kNumConsumerWarpsPerStage = kPingPong ? 4 : kNumMathWarps;
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    constexpr uint32_t kSwizzleL1OutMode = WG_L1_OUT_BLOCK_N * sizeof(d_dtype_t);
    constexpr uint32_t kSwizzleL2OutMode = WGMMA_M * sizeof(d_dtype_t);
    DG_STATIC_ASSERT((kSwizzleL1OutMode == 64 or kSwizzleL1OutMode == 128) and kSwizzleL2OutMode == 128, "Invalid output swizzling");
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumMathThreads;
    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) d_dtype_t l1[kNumMathWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * WG_L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t l2[kNumMathWarpgroups][kNumMHalves][STORE_BLOCK_M * WGMMA_M];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][BLOCK_N * BLOCK_K];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier combine_barriers[kNumMathWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    DG_STATIC_ASSERT(sizeof(SharedStorage::smem_a[0]) % kSharedMemoryAlignment == 0, "Invalid stage size");
    DG_STATIC_ASSERT(sizeof(SharedStorage::smem_b[0]) % kSharedMemoryAlignment == 0, "Invalid stage size");
    DG_STATIC_ASSERT(sizeof(SharedStorage::smem_d.l1[0][0]) % (8 * kSwizzleL1OutMode) == 0, "Invalid L1 output tile alignment");
    DG_STATIC_ASSERT(sizeof(SharedStorage::smem_d.l2[0][0]) % (8 * kSwizzleL2OutMode) == 0, "Invalid L2 output tile alignment");
    DG_STATIC_ASSERT(sizeof(SharedStorage::smem_d.l1[0]) == sizeof(SharedStorage::smem_d.l2[0]), "Per-warpgroup output tiles must not overlap");
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));
    if (warp_idx == 0) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            shared_storage.expert_token_count[i] = 0;
    } else if (warp_idx == 1) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                shared_storage.full_barriers[i].init(2);
                shared_storage.empty_barriers[i].init(2 * kNumConsumerWarpsPerStage);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_sync();
    cudaGridDependencySynchronize();
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;
    constexpr bool kUseMoreMathRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumTMARegisters = kUseMoreMathRegisters ? 40 : 88;
    constexpr uint32_t kNumMathRegisters = kUseMoreMathRegisters ? 208 : 184;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumTMARegisters * kNumTMAThreads +
                     kNumMathRegisters * kNumMathThreads <= 65536,
                     "Too many registers");
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
        if (sm_idx == 0) {
            DG_STATIC_ASSERT(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
            if (thread_idx < kNumRanks)
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
        ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithEpilogueBarrierIdx);
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;
            constexpr uint32_t kHiddenBytes = kHidden * sizeof(nv_bfloat16);
            constexpr uint32_t kNumChunks = kHiddenBytes / kNumBytesPerPull;
            DG_STATIC_ASSERT(kHiddenBytes % kNumBytesPerPull == 0, "Invalid hidden");
            const auto pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }
        ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = SHARED_L2_SHAPE_K / (BLOCK_N / 2);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                const uint32_t m_idx = block_idx * BLOCK_M;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_a[0]));
                }
                __syncwarp();
            }
        }
        for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
            shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                const uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                if (cute::elect_one_sync()) {
                    if (kL1Natural and task_info.block_phase == sched::BlockPhase::Linear1) {
                        tma::copy_gate_up_natural<BLOCK_K, BLOCK_N, kSwizzleBMode, L1_SHAPE_N, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx);
                    } else if (kL1Natural and task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                        // All shared experts form one `[gate | up]` matrix of `L1_SHAPE_N * kNumSharedExperts` rows
                        tma::copy_gate_up_natural<BLOCK_K, BLOCK_N, kSwizzleBMode, L1_SHAPE_N * (kHasShared ? kNumSharedExperts : 1), b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx);
                    } else {
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 1);
                    }
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_b[0]));
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (is_leader_cta)
            scheduler.mainloop(num_tokens);
    } else if (warp_idx < kNumDispatchWarps + kNumTMAWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const auto math_warp_idx = warp_idx - (kNumDispatchWarps + kNumTMAWarps);
        const auto math_wg_idx = __shfl_sync(0xffffffff, math_warp_idx / 4, 0);
        const auto math_thread_idx = math_warp_idx * 32 + lane_idx;
        const auto math_thread_idx_in_wg = math_thread_idx % 128;
        const auto warp_idx_in_wg = math_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumTMAWarps) % 4 == 0 and kNumMathWarps % 4 == 0, "Invalid math warps");
        const uint32_t wg_row_base = kPingPong ? 0u : math_wg_idx * WGMMA_M;
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(
            shared_storage.smem_b[0], 0, 0);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(
            shared_storage.smem_a[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumStoreAtomsPerBlock = STORE_BLOCK_M / 16;
        constexpr uint32_t kNumWeightRegs = math::constexpr_ceil_div(STORE_BLOCK_M, 32u);
        constexpr uint32_t kNumL2ChunksPerRow = kNumMHalves * (kSwizzleL2OutMode / kNumBankGroupBytes);
        constexpr uint32_t kNumL2RowsPerPass = 128 / kNumL2ChunksPerRow;
        constexpr uint32_t kNumL2HalfBytes = STORE_BLOCK_M * kSwizzleL2OutMode;
        DG_STATIC_ASSERT(STORE_BLOCK_M % kNumL2RowsPerPass == 0, "Invalid store block M");
        const uint32_t frag_col_offset = (lane_idx % 4) * 2;
        const auto owner_sync = [&]() {
            if constexpr (kPingPong) {
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);
            } else {
                ptx::sync_aligned(kNumMathThreads, kEpilogueFullBarrierIdx);
            }
        };
        const bool is_owner_leader = kPingPong ? (math_thread_idx_in_wg == 0) : (math_thread_idx == 0);
        ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithEpilogueBarrierIdx);
        uint32_t task_seq = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto num_k_blocks = task_info.shape_k / BLOCK_K;

            if constexpr (kPingPong) {
                if ((task_seq ++ % kNumMathWarpgroups) != math_wg_idx) {
                    scheduler.release_task_info();
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx));
                    continue;
                }
            }
            float accum[kNumMHalves][kNumAccum] = {};
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (sizeof(SharedStorage::smem_b[0]) / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (sizeof(SharedStorage::smem_a[0]) / 16);
                shared_storage.full_barriers[stage_idx].wait(phase);
                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[mh][i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++ k) {
                    #pragma unroll
                    for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, nv_bfloat16>(
                            a_desc_base_lo, wg_row_base + mh * WGMMA_M, k * WGMMA_K);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, nv_bfloat16>(
                            b_desc_base_lo, 0, k * WGMMA_K);
                        WGMMA_0::wgmma(a_desc, b_desc, accum[mh], 1);
                        if constexpr (WGMMA_N_1 > 0) {
                            b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, nv_bfloat16>(
                                b_desc_base_lo, WGMMA_N_0, k * WGMMA_K);
                            WGMMA_1::wgmma(a_desc, b_desc, accum[mh] + WGMMA_0::kNumAccum, 1);
                        }
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[mh][i]);
                }
                ptx::warpgroup_wait<0>();
                if (lane_idx < 2)
                    shared_storage.empty_barriers[stage_idx].arrive(lane_idx);
                __syncwarp();
            }
            scheduler.release_task_info();
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }
                const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_row_base / 2;

                #pragma unroll
                for (uint32_t s = 0; s < BLOCK_M / STORE_BLOCK_M; ++ s) {
                    if (s * STORE_BLOCK_M >= valid_m)
                        break;
                    float stored_cached_weight[kNumWeightRegs];
                    #pragma unroll
                    for (uint32_t r = 0; r < kNumWeightRegs; ++ r) {
                        const uint32_t token_idx_in_store = r * 32 + lane_idx;
                        stored_cached_weight[r] = 1.0f;
                        if (not task_info.is_shared() and (STORE_BLOCK_M % 32 == 0 or token_idx_in_store < STORE_BLOCK_M)) {
                            stored_cached_weight[r] = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + s * STORE_BLOCK_M + token_idx_in_store)
                                .template get_base_ptr<float>();
                        }
                    }
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);

                    const auto smem_tile = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[math_wg_idx][tma_stage_idx]);
                    #pragma unroll
                    for (uint32_t a = 0; a < kNumStoreAtomsPerBlock; ++ a) {
                        #pragma unroll
                        for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                            uint32_t packed[2];
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++ h) {
                                const uint32_t i = (s * STORE_BLOCK_M + a * 16) / 8 + h;
                                const uint32_t token_group_base = a * 16 + h * 8;
                                const uint32_t weight_lane = (token_group_base % 32) + frag_col_offset;
                                const float2 weights = {
                                    ptx::exchange(stored_cached_weight[token_group_base / 32], weight_lane + 0),
                                    ptx::exchange(stored_cached_weight[token_group_base / 32], weight_lane + 1)
                                };

                                auto bf16_gate = __float22bfloat162_rn(make_float2(accum[mh][i * 4 + 0], accum[mh][i * 4 + 1]));
                                auto bf16_up = __float22bfloat162_rn(make_float2(accum[mh][i * 4 + 2], accum[mh][i * 4 + 3]));
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                    bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                    bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                    bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                                }
                                auto gate = __bfloat1622float2(bf16_gate);
                                const auto neg_gate_exp = make_float2(
                                    kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                    kFastMath ? __expf(-gate.y) : expf(-gate.y));
                                const auto denom = make_float2(1.0f + neg_gate_exp.x, 1.0f + neg_gate_exp.y);
                                if constexpr (kFastMath) {
                                    gate = make_float2(gate.x * math::fast_rcp(denom.x), gate.y * math::fast_rcp(denom.y));
                                } else {
                                    gate = make_float2(gate.x / denom.x, gate.y / denom.y);
                                }
                                const auto up = __bfloat1622float2(bf16_up);
                                const auto bf16_output = __float22bfloat162_rn(make_float2(
                                    gate.x * up.x * weights.x, gate.y * up.y * weights.y));
                                packed[h] = *reinterpret_cast<const uint32_t*>(&bf16_output);
                            }
                            const uint32_t row = a * 16 + (lane_idx % 16);
                            const uint32_t col_chunk = mh * 4 + warp_idx_in_wg;
                            const uint32_t byte_offset = math::swizzle_byte_offset<kSwizzleL1OutMode>(
                                row * kSwizzleL1OutMode + col_chunk * kNumBankGroupBytes);
                            ptx::SM90_U32x2_STSM_T<uint32_t>::copy(packed[0], packed[1], smem_tile + byte_offset);
                        }
                    }
                    cute::tma_store_fence();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr, smem_tile,
                            out_n_idx, m_idx + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
                ptx::tma_store_wait<0>();
                owner_sync();
                if (is_owner_leader) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                if (not task_info.is_shared()) {
                    if (is_owner_leader)
                        ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    __syncwarp();
                }
                const auto smem_tile = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[math_wg_idx]);
                const uint32_t wg_n_byte_offset = (n_idx + wg_row_base) * static_cast<uint32_t>(sizeof(d_dtype_t));

                #pragma unroll
                for (uint32_t s = 0; s < BLOCK_M / STORE_BLOCK_M; ++ s) {
                    if (s * STORE_BLOCK_M >= valid_m)
                        break;
                    if (s > 0)
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);
                    #pragma unroll
                    for (uint32_t a = 0; a < kNumStoreAtomsPerBlock; ++ a) {
                        const uint32_t i = (s * STORE_BLOCK_M + a * 16) / 8;
                        const uint32_t m = lane_idx / 8;
                        const uint32_t row = a * 16 + (m / 2) * 8 + (lane_idx % 8);
                        const uint32_t col_chunk = warp_idx_in_wg * 2 + (m % 2);
                        #pragma unroll
                        for (uint32_t mh = 0; mh < kNumMHalves; ++ mh) {
                            const uint32_t byte_offset = mh * kNumL2HalfBytes + math::swizzle_byte_offset<kSwizzleL2OutMode>(
                                row * kSwizzleL2OutMode + col_chunk * kNumBankGroupBytes);
                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                math::cast_into_bf16_and_pack(accum[mh][i * 4 + 0], accum[mh][i * 4 + 1]),
                                math::cast_into_bf16_and_pack(accum[mh][i * 4 + 2], accum[mh][i * 4 + 3]),
                                math::cast_into_bf16_and_pack(accum[mh][i * 4 + 4], accum[mh][i * 4 + 5]),
                                math::cast_into_bf16_and_pack(accum[mh][i * 4 + 6], accum[mh][i * 4 + 7]),
                                smem_tile + byte_offset
                            );
                        }
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);
                    #pragma unroll
                    for (uint32_t p = 0; p < STORE_BLOCK_M / kNumL2RowsPerPass; ++ p) {
                        const uint32_t idx = p * 128 + math_thread_idx_in_wg;
                        const uint32_t row = idx / kNumL2ChunksPerRow;
                        const uint32_t chunk = idx % kNumL2ChunksPerRow;
                        const uint32_t m_idx_in_block = s * STORE_BLOCK_M + row;
                        if (m_idx_in_block >= valid_m)
                            continue;

                        const auto [dst_rank_idx, dst_token_idx, dst_topk_idx] = task_info.is_shared() ?
                            layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                            *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                        const uint32_t mh = chunk / (kSwizzleL2OutMode / kNumBankGroupBytes);
                        const uint32_t col_chunk = chunk % (kSwizzleL2OutMode / kNumBankGroupBytes);
                        const uint32_t byte_offset = mh * kNumL2HalfBytes + math::swizzle_byte_offset<kSwizzleL2OutMode>(
                            row * kSwizzleL2OutMode + col_chunk * kNumBankGroupBytes);
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_tile + byte_offset));
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(), wg_n_byte_offset + chunk * kNumBankGroupBytes);
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + math_wg_idx);
            }
        }
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumMathWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumMathWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (math_warp_idx + i * kNumMathWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (math_warp_idx + kNumMathWarps * 2) * kNumChunkBytes);
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + math_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        DG_STATIC_ASSERT(kNumRanks <= kNumMathThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and math_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(math_thread_idx);
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, math_thread_idx,
            [&]() { ptx::sync_aligned(kNumMathThreads, kEpilogueFullBarrierIdx); }
        );
        if (sm_idx == 0 and math_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), math_thread_idx), peer_grid_idx);
        ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithEpilogueBarrierIdx);
        const auto grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t token_chunk_idx = math_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumMathWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { DG_DEVICE_PRINTF("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };
            bool do_reduce = move_mask_and_load(load_stage_idx);
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
            while (do_reduce) {
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                cutlass::arch::fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                    combine_store_buffer, kNumChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm
