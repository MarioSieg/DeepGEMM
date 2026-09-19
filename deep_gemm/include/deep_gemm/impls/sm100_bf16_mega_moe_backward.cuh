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
    template<const bool kFastMath>
    [[nodiscard]] __device__ __forceinline__ float sigmoid(float x) {
        if constexpr (kFastMath) return 1.0f/(1.0f+__expf(-x));
        else return 1.0f/(1.0f+expf(-x));
    }

    template <
        uint32_t kNumMaxTokensPerRank,
        uint32_t kHidden, uint32_t kIntermediateHidden,
        uint32_t kNumExperts, uint32_t kNumSharedExperts,
        uint32_t kNumTopk,
        uint32_t BLOCK_M,
        uint32_t kNumRingTokens,
        uint32_t kNumSMs, uint32_t kNumRanks,
        float kActivationClamp,
        bool kFastMath,
        uint32_t kNumThreads,
        bool kHasShared=(kNumSharedExperts > 0),
        uint32_t kNumExpertsPerRank=kNumExperts / kNumRanks,
        uint32_t kNumWarps=(kNumThreads>>5),
        uint32_t kGran=8
    >
    CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
    sm100_bf16_mega_moe_backward_impl(
        void *__restrict__ dx,
        float *__restrict__ dw1_weights,
        float *__restrict__ dw2_weights,
        float *__restrict__ dtopk_weights,
        const void *__restrict__ w1_weights,
        const void *__restrict__ w2_weights,
        const void *__restrict__ shared_w1_weights,
        const void *__restrict__ shared_w2_weights,
        float *__restrict__ shared_dw1_weights,
        float *__restrict__ shared_dw2_weights,
        uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer
    ) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
        DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid number of threads");
        DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts || ranks");
        DG_STATIC_ASSERT(BLOCK_M <= 32, "Reference implementation keeps one logical token per warp lane");
        DG_STATIC_ASSERT(kIntermediateHidden % kGran == 0, "Invalid intermediate hidden for gate/up interleaving");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");

        uint32_t tid=threadIdx.x;
        uint32_t sm_idx=blockIdx.x;
        uint32_t warp=tid>>5;
        uint32_t lane=31&tid;
        constexpr uint32_t kNumGlobalThreads=kNumSMs*kNumThreads;
        uint32_t global_tid=sm_idx*kNumThreads + tid;
        const auto buffer=layout::MegaMoEBuffer(
            sym_buffer.get_base_ptr(),
            kHidden, kIntermediateHidden,
            kNumRanks, kNumExperts,
            kNumMaxTokensPerRank, kNumTopk,
            kNumRingTokens, 0,
            false, // NO SF
            kNumSharedExperts
        );
        const auto workspace=buffer.workspace;
        const auto bw=layout::MegaMoEBackwardBuffer(
            buffer.get_end_ptr(),
            kHidden, kNumMaxTokensPerRank, kNumTopk, kNumSharedExperts,
            workspace.num_max_pool_tokens
        );
        using BlockDesc=layout::MegaMoEBackwardBuffer::BlockDesc;
        for (uint32_t i=global_tid; i < kNumExperts; i += kNumGlobalThreads)
            *workspace.get_expert_send_count_ptr(i)=0;
        for (uint32_t i=global_tid; i < kNumExpertsPerRank; i += kNumGlobalThreads)
            *workspace.get_expert_recv_count_sum_ptr(i)=0;
        comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() -> void { __syncthreads(); });
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 101>(workspace, sym_buffer, sm_idx, tid, [&]() -> void { __syncthreads(); }, false, true);
        for (uint32_t idx=global_tid; idx < num_tokens*kNumTopk; idx += kNumGlobalThreads) {
            const int64_t expert_idx=buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[idx];
            if (expert_idx < 0) continue;
            auto dst_rank=static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
            auto dst_local_expert=static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
            auto slot=static_cast<uint32_t>(ptx::atomic_add(workspace.get_expert_send_count_ptr(expert_idx), 1));
            auto *dst_ptr=workspace.get_src_token_topk_idx_ptr(dst_local_expert, sym_buffer.rank_idx, slot);
            *sym_buffer.map(dst_ptr, dst_rank)=idx;
        }
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 102>(workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
        for (uint32_t i=global_tid; i < kNumExperts; i += kNumGlobalThreads) {
            auto dst_rank=i / kNumExpertsPerRank;
            auto dst_local_expert=i % kNumExpertsPerRank;
            auto count=static_cast<uint32_t>(*workspace.get_expert_send_count_ptr(i));
            *sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert), dst_rank)=count;
            ptx::atomic_add_sys(sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert), dst_rank),static_cast<uint64_t>(count));
        }
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 103>(workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
        if (sm_idx == 0 && tid == 0) {
            uint32_t pool_begin=0, num_blocks=0;
            for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
                const auto valid_m=static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
                for (uint32_t off=0; off < valid_m; off += BLOCK_M) {
                    bw.block_desc[num_blocks].local_expert=e;
                    bw.block_desc[num_blocks].pool_begin=pool_begin + off;
                    bw.block_desc[num_blocks].valid_m=cute::min(BLOCK_M, valid_m - off);
                    ++num_blocks;
                }
                pool_begin += valid_m;
            }
            if constexpr (kHasShared) {
                for (uint32_t off=0; off < num_tokens; off += BLOCK_M) {
                    bw.block_desc[num_blocks].local_expert=kNumExpertsPerRank;
                    bw.block_desc[num_blocks].pool_begin=off;
                    bw.block_desc[num_blocks].valid_m=cute::min(BLOCK_M, num_tokens - off);
                    ++num_blocks;
                }
            }
            DG_DEVICE_ASSERT(num_blocks <= bw.max_num_blocks);
            *bw.num_blocks=num_blocks;
            *bw.next_block=0;
        }
        comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
        __shared__ uint32_t s_pool_begin[kNumExpertsPerRank];
        __shared__ uint32_t s_rank_prefix[kNumExpertsPerRank][kNumRanks];
        if (tid == 0) {
            uint32_t pool_begin=0;
            for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
                s_pool_begin[e]=pool_begin;
                uint32_t rank_prefix=0;
                for (uint32_t r=0; r < kNumRanks; ++r) {
                    s_rank_prefix[e][r]=rank_prefix;
                    rank_prefix += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, e));
                }
                pool_begin += rank_prefix;
            }
        }
        __syncthreads();
        for (uint32_t e=0; e < kNumExpertsPerRank; ++e) {
            auto valid_m_e=static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            auto pool_begin_e=s_pool_begin[e];
            for (uint32_t local_pos=global_tid; local_pos < valid_m_e; local_pos += kNumGlobalThreads) {
                uint32_t r=0;
                #pragma unroll
                for (uint32_t rr=0; rr < kNumRanks; ++rr)
                    if (s_rank_prefix[e][rr] <= local_pos)
                        r=rr;
                auto slot=local_pos - s_rank_prefix[e][r];
                auto token_topk=*workspace.get_src_token_topk_idx_ptr(e, r, slot);
                auto src_token=token_topk / kNumTopk;
                auto src_topk=token_topk % kNumTopk;
                *workspace.get_token_src_metadata_ptr(pool_begin_e + local_pos) = layout::TokenSrcMetadata(r, src_token, src_topk);
            }
        }
        comm::grid_sync<kNumSMs, 0>(workspace, sm_idx, tid, [&]() { __syncthreads(); });
        extern __shared__ __align__(1024) uint8_t smem_raw[];
        struct smem_shared final {
            alignas(1024) nv_bfloat16 x[BLOCK_M][kHidden];
            alignas(1024) nv_bfloat16 dy[BLOCK_M][kHidden];
            alignas(1024) float z[BLOCK_M][2*kIntermediateHidden];
            alignas(1024) float h[BLOCK_M][kIntermediateHidden];
            alignas(1024) float dh[BLOCK_M][kIntermediateHidden];
            alignas(1024) float dz[BLOCK_M][2*kIntermediateHidden];
            alignas(1024) float dx_local[BLOCK_M][kHidden];
            float route_weight[BLOCK_M];
            uint32_t src_rank[BLOCK_M];
            uint32_t src_token[BLOCK_M];
            uint32_t src_topk[BLOCK_M];
        };
        static_assert(std::is_trivial_v<smem_shared>);
        auto& smem=*reinterpret_cast<smem_shared*>(smem_raw);

        const auto load_block=[&](const BlockDesc& bd) {
            const bool is_shared=bd.local_expert >= kNumExpertsPerRank;
            for (uint32_t row=warp; row < bd.valid_m; row += kNumWarps) {
                uint32_t src_rank, src_token, src_topk;
                if (is_shared) {
                    src_rank=sym_buffer.rank_idx;
                    src_token=bd.pool_begin + row;
                    src_topk=kNumTopk;
                } else {
                    const auto meta=*workspace.get_token_src_metadata_ptr(bd.pool_begin + row);
                    src_rank=meta.rank_idx;
                    src_token=meta.token_idx;
                    src_topk=meta.topk_idx;
                }
                if (lane == 0) {
                    smem.src_rank[row]=src_rank;
                    smem.src_token[row]=src_token;
                    smem.src_topk[row]=src_topk;
                    smem.route_weight[row]=is_shared ? 1.0f : *sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token*kNumTopk + src_topk,
                        src_rank);
                }
                const auto *remote_x=sym_buffer.map(
                    buffer.input_token_buffer.get_data_buffer(src_token).template get_base_ptr<nv_bfloat16>(), src_rank);
                const auto *remote_dy=sym_buffer.map(
                    bw.input_dy_buffer.get_data_buffer(src_token).template get_base_ptr<nv_bfloat16>(), src_rank);
                for (uint32_t k=lane; k < kHidden; k += 32) {
                    smem.x[row][k]=remote_x[k];
                    smem.dy[row][k]=remote_dy[k];
                }
            }
        };
        const auto pg=[](uint32_t j) -> uint32_t {
            return ((j / kGran)<<1)*kGran + j%kGran;
        };
        const auto compute_block=[&](const BlockDesc& bd) {
            const bool is_shared=bd.local_expert >= kNumExpertsPerRank;
            uint32_t valid_m=bd.valid_m;
            for (uint32_t idx=tid; idx < valid_m*kHidden; idx += kNumThreads)
                smem.dx_local[idx / kHidden][idx % kHidden]=0.0f;
            __syncthreads();
            uint32_t num_passes=is_shared ? kNumSharedExperts : 1;
            for (uint32_t pass=0; pass < num_passes; ++pass) {
                const auto *w1=static_cast<const nv_bfloat16*>(is_shared ? shared_w1_weights : w1_weights) +
                    static_cast<uint64_t>(is_shared ? pass : bd.local_expert)*(2ull*kIntermediateHidden)*kHidden;
                const auto dw1=(is_shared ? shared_dw1_weights : dw1_weights) +
                    static_cast<uint64_t>(is_shared ? pass : bd.local_expert)*(2ull*kIntermediateHidden)*kHidden;
                uint32_t w2_stride=is_shared ? kIntermediateHidden*kNumSharedExperts : kIntermediateHidden;
                const uint64_t w2_col=is_shared ? static_cast<uint64_t>(pass)*kIntermediateHidden : 0;
                const auto *w2=static_cast<const nv_bfloat16*>(is_shared ? shared_w2_weights : w2_weights) + (is_shared ? 0ull : static_cast<uint64_t>(bd.local_expert)*kHidden*kIntermediateHidden);
                const auto dw2=(is_shared ? shared_dw2_weights : dw2_weights) + (is_shared ? 0ull : static_cast<uint64_t>(bd.local_expert)*kHidden*kIntermediateHidden);
                for (uint32_t row=warp; row < valid_m; row += kNumWarps) {
                    const auto *xrow=smem.x[row];
                    for (uint32_t n=lane; n < 2*kIntermediateHidden; n += 32) {
                        float acc=0.0f;
                        const auto *wrow=w1 + static_cast<uint64_t>(n)*kHidden;
                        for (uint32_t k=0; k < kHidden; ++k)
                            acc=fmaf(__bfloat162float(xrow[k]), __bfloat162float(wrow[k]), acc);
                        smem.z[row][n]=acc;
                    }
                }
                __syncthreads();
                for (uint32_t idx=tid; idx < valid_m*kIntermediateHidden; idx += kNumThreads) {
                    uint32_t row=idx / kIntermediateHidden, j=idx % kIntermediateHidden;
                    auto p=pg(j);
                    float gate=smem.z[row][p], up=smem.z[row][p + kGran];
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                        gate=cute::min(gate, kActivationClamp);
                        up=cute::max(cute::min(up, kActivationClamp), -kActivationClamp);
                    }
                    smem.h[row][j]=gate*sigmoid<kFastMath>(gate)*up;
                }
                __syncthreads();
                for (uint32_t row=warp; row < valid_m; row += kNumWarps) {
                    for (uint32_t j=lane; j < kIntermediateHidden; j += 32) {
                        float acc=0.0f;
                        for (uint32_t n=0; n < kHidden; ++n)
                            acc=fmaf(__bfloat162float(smem.dy[row][n]),
                                       __bfloat162float(w2[static_cast<uint64_t>(n)*w2_stride + w2_col + j]), acc);
                        smem.dh[row][j]=acc;
                    }
                }
                __syncthreads();
                for (uint32_t row=warp; row < valid_m; row += kNumWarps) {
                    float partial=0.0f;
                    for (uint32_t j=lane; j < kIntermediateHidden; j += 32)
                        partial=fmaf(smem.h[row][j], smem.dh[row][j], partial);
                    #pragma unroll
                    for (uint32_t offset=16; offset > 0; offset >>= 1)
                        partial += __shfl_xor_sync(0xffffffff, partial, offset);
                    if (lane == 0 && not is_shared) {
                        auto *remote_dw=sym_buffer.map(
                            bw.dtopk_weight_slot_buffer.get_rank_buffer(smem.src_topk[row])
                                .get_data_buffer(smem.src_token[row]).template get_base_ptr<float>(),
                            smem.src_rank[row]);
                        *remote_dw=partial;
                    }
                    auto w=smem.route_weight[row];
                    for (uint32_t j=lane; j < kIntermediateHidden; j += 32)
                        smem.dh[row][j] *= w;
                }
                __syncthreads();
                for (uint64_t idx=tid; idx < static_cast<uint64_t>(kHidden)*kIntermediateHidden; idx += kNumThreads) {
                    uint32_t n=idx / kIntermediateHidden, j=idx % kIntermediateHidden;
                    float acc=0.0f;
                    for (uint32_t row=0; row < valid_m; ++row) {
                        float dO=smem.route_weight[row]*__bfloat162float(smem.dy[row][n]);
                        acc=fmaf(smem.h[row][j], dO, acc);
                    }
                    atomicAdd(dw2 + n*w2_stride + w2_col + j, acc);
                }
                for (uint32_t idx=tid; idx < valid_m*kIntermediateHidden; idx += kNumThreads) {
                    uint32_t row=idx / kIntermediateHidden, j=idx % kIntermediateHidden;
                    auto p=pg(j);
                    float gate=smem.z[row][p], up=smem.z[row][p + kGran];
                    bool gate_active=true, up_active=true;
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                        gate_active=gate < kActivationClamp;
                        up_active=up > -kActivationClamp && up < kActivationClamp;
                        gate=cute::min(gate, kActivationClamp);
                        up=cute::max(cute::min(up, kActivationClamp), -kActivationClamp);
                    }
                    float sig=sigmoid<kFastMath>(gate);
                    float silu=gate*sig;
                    float dsilu=sig*(1.0f + gate*(1.0f - sig));
                    float dh=smem.dh[row][j];
                    smem.dz[row][p]=gate_active ? dh*up*dsilu : 0.0f;
                    smem.dz[row][p + kGran]=up_active ? dh*silu : 0.0f;
                }
                __syncthreads();
                for (uint64_t idx=tid; idx < static_cast<uint64_t>(2*kIntermediateHidden)*kHidden; idx += kNumThreads) {
                    uint32_t n=idx / kHidden, k=idx % kHidden;
                    float acc=0.0f;
                    for (uint32_t row=0; row < valid_m; ++row)
                        acc=fmaf(smem.dz[row][n], __bfloat162float(smem.x[row][k]), acc);
                    atomicAdd(dw1 + idx, acc);
                }
                for (uint32_t row=warp; row < valid_m; row += kNumWarps) {
                    for (uint32_t k=lane; k < kHidden; k += 32) {
                        float acc=0.0f;
                        for (uint32_t n=0; n < 2*kIntermediateHidden; ++n)
                            acc=fmaf(smem.dz[row][n], __bfloat162float(w1[static_cast<uint64_t>(n)*kHidden + k]), acc);
                        smem.dx_local[row][k] += acc;
                    }
                }
                __syncthreads();
            }
            for (uint32_t row=warp; row < valid_m; row += kNumWarps) {
                auto *remote_dx=sym_buffer.map(bw.dx_slot_buffer.get_rank_buffer(smem.src_topk[row]).get_data_buffer(smem.src_token[row]).template get_base_ptr<nv_bfloat16>(),
                    smem.src_rank[row]);
                for (uint32_t k=lane; k < kHidden; k += 32)
                    remote_dx[k]=__float2bfloat16_rn(smem.dx_local[row][k]);
            }
        };
        __shared__ uint32_t s_block_idx;
        for (;;) {
            if (tid == 0) s_block_idx=ptx::atomic_add(bw.next_block, 1u);
            __syncthreads();
            if (s_block_idx >= *bw.num_blocks) break;
            BlockDesc bd=bw.block_desc[s_block_idx];
            load_block(bd);
            __syncthreads();
            compute_block(bd);
            __syncthreads();
        }
        __threadfence_system();
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 199>(
            workspace, sym_buffer, sm_idx, tid, [&]() { __syncthreads(); });
        for (uint64_t linear=static_cast<uint64_t>(sm_idx)*kNumThreads + tid; linear < static_cast<uint64_t>(num_tokens)*kHidden; linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
            uint32_t token=linear / kHidden, k=linear % kHidden;
            float sum=0.0f;
            #pragma unroll
            for (uint32_t topk=0; topk < kNumTopk; ++topk) {
                const auto e=buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token*kNumTopk + topk];
                if (e >= 0) {
                    const auto *slot=bw.dx_slot_buffer.get_rank_buffer(topk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>();
                    sum += __bfloat162float(slot[k]);
                }
            }
            if constexpr (kHasShared) {
                const auto *shared_slot=bw.dx_slot_buffer.get_rank_buffer(kNumTopk).get_data_buffer(token).template get_base_ptr<nv_bfloat16>();
                sum += __bfloat162float(shared_slot[k]);
            }
            static_cast<nv_bfloat16*>(dx)[linear]=__float2bfloat16_rn(sum);
        }
        for (uint64_t linear=static_cast<uint64_t>(sm_idx)*kNumThreads + tid;
             linear < static_cast<uint64_t>(num_tokens)*kNumTopk;
             linear += static_cast<uint64_t>(kNumSMs)*kNumThreads) {
            uint32_t token=linear / kNumTopk, topk=linear % kNumTopk;
            const auto e=buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[linear];
            dtopk_weights[linear]=e < 0 ? 0.0f : *bw.dtopk_weight_slot_buffer.get_rank_buffer(topk).get_data_buffer(token).template get_base_ptr<float>();
        }
    #else
        if (blockIdx.x == 0 && threadIdx.x == 0)
            DG_DEVICE_ASSERT(false && "This kernel only support sm_100f");
    #endif
    }
}
