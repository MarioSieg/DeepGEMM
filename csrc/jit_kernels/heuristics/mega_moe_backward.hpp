#pragma once

#include <ostream>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEBackwardConfig {
    int block_m;
    int num_threads;
    int num_stages;
    int smem_size;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEBackwardConfig& config) {
        os << "MegaMoEBackwardConfig("
           << "block_m=" << config.block_m
           << ", num_threads=" << config.num_threads
           << ", num_stages=" << config.num_stages
           << ", smem_size=" << config.smem_size << ")";
        return os;
    }
};

static constexpr int kMegaMoEBackwardBlockM = layout::MegaMoEBackwardBuffer::kBlockM;
static constexpr int kMegaMoEBackwardNumThreads = 256;
static constexpr int kMegaMoEBackwardStageABytes = 128 * 64 * 2;
static constexpr int kMegaMoEBackwardStageBBytes = 128 * 32 * 2;

static int get_mega_moe_backward_smem_size(const int& num_stages) {
    return num_stages * (kMegaMoEBackwardStageABytes + kMegaMoEBackwardStageBBytes) + 8192;
}

static MegaMoEBackwardConfig get_mega_moe_backward_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {
    DG_HOST_ASSERT(mma_kind == MmaKind::BF16);
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    int num_stages = 8;
    while (get_mega_moe_backward_smem_size(num_stages) > SM100ArchSpec::smem_capacity)
        --num_stages;
    DG_HOST_ASSERT(num_stages >= 2);
    return MegaMoEBackwardConfig{kMegaMoEBackwardBlockM, kMegaMoEBackwardNumThreads, num_stages,
                                 get_mega_moe_backward_smem_size(num_stages)};
}

} // namespace deep_gemm
