#pragma once

#include <format>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe_backward.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_bf16_mega_moe_backward(
    const torch::Tensor& dx,
    const torch::Tensor& dw1_weights, const torch::Tensor& dw2_weights, const torch::Tensor& dtopk_weights,
    const torch::Tensor& w1_weights, const torch::Tensor& w2_weights,
    const torch::Tensor* shared_w1_weights, const torch::Tensor* shared_w2_weights,
    torch::Tensor* shared_dw1_weights, torch::Tensor* shared_dw2_weights,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto config = get_mega_moe_backward_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    const auto num_sms = runtime->get_num_sms();

    // Shared-expert args are optional
    void* shared_w1_ptr = nullptr;
    void* shared_w2_ptr = nullptr;
    float* shared_dw1_ptr = nullptr;
    float* shared_dw2_ptr = nullptr;
    if (shared_w1_weights != nullptr) {
        shared_w1_ptr = shared_w1_weights->data_ptr();
        shared_w2_ptr = shared_w2_weights->data_ptr();
        shared_dw1_ptr = shared_dw1_weights->data_ptr<float>();
        shared_dw2_ptr = shared_dw2_weights->data_ptr<float>();
    }

    const auto kernel = jit->compile("sm100_bf16_mega_moe_backward", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_backward_impl<
        {},
        {}, {},
        {}, {},
        {},
        {},
        {},
        {}, {},
        {}, {},
        {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m,
        num_ring_tokens,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        config.num_threads));
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_threads, 1, 1),
        },
        dx.data_ptr(),
        dw1_weights.data_ptr<float>(), dw2_weights.data_ptr<float>(), dtopk_weights.data_ptr<float>(),
        w1_weights.data_ptr(), w2_weights.data_ptr(),
        shared_w1_ptr, shared_w2_ptr, shared_dw1_ptr, shared_dw2_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx));
}

}
