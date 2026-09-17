#pragma once

#include "mega_moe.hpp"

namespace deep_gemm {

struct MegaMoEBackwardConfig : MegaMoEConfig {};

static MegaMoEBackwardConfig get_mega_moe_backward_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {
    return MegaMoEBackwardConfig{get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens, mma_kind)};
}

}