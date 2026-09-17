#pragma once

#include <ostream>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

#include "sm100.hpp"

namespace deep_gemm {


struct MegaMoEBackwardConfig {
    int block_m;
    int num_threads;
    int smem_size;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEBackwardConfig& config) {
        os << "MegaMoEBackwardConfig("
           << "block_m=" << config.block_m
           << ", num_threads=" << config.num_threads
           << ", smem_size=" << config.smem_size << ")";
        return os;
    }
};

static int get_mega_moe_backward_smem_size(const int& block_m, const int& hidden, const int& intermediate_hidden) {
    constexpr int kAlign = 1024;
    const auto a = [&](const int& n) { return math::align(n, kAlign); };
    const int smem_size =
        a(block_m * hidden * 2) +                     // x
        a(block_m * hidden * 2) +                      // dy
        a(block_m * 2 * intermediate_hidden * 4) +     // z
        a(block_m * intermediate_hidden * 4) +          // h
        a(block_m * intermediate_hidden * 4) +          // dh
        a(block_m * 2 * intermediate_hidden * 4) +      // dz
        a(block_m * hidden * 4) +                        // dx_local
        4096;                                              // route_weight + src rank/token/topk + slack
    return smem_size;
}

static MegaMoEBackwardConfig get_mega_moe_backward_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {
    DG_HOST_ASSERT(mma_kind == MmaKind::BF16);
    DG_HOST_ASSERT(hidden % 8 == 0 and intermediate_hidden % 8 == 0);
    int block_m = 8;
    for (const int& candidate: {32, 16, 8}) {
        if (get_mega_moe_backward_smem_size(candidate, hidden, intermediate_hidden) <= SM100ArchSpec::smem_capacity) {
            block_m = candidate;
            break;
        }
    }
    const auto smem_size = get_mega_moe_backward_smem_size(block_m, hidden, intermediate_hidden);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    return MegaMoEBackwardConfig{block_m, 256, smem_size};
}

} // namespace deep_gemm
