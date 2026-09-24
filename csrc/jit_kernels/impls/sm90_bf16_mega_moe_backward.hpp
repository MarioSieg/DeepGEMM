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

static void sm90_bf16_mega_moe_backward(
    const torch::Tensor& dx,
    const torch::Tensor& dw1_weights, const torch::Tensor& dw2_weights, const torch::Tensor& dtopk_weights,
    const torch::Tensor& w1_weights, const torch::Tensor& w2_weights,
    const torch::Tensor* shared_w1_weights, const torch::Tensor* shared_w2_weights,
    torch::Tensor* shared_dw1_weights, torch::Tensor* shared_dw2_weights,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& dw_natural_layout,
    const bool& l1_natural_layout
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_sms = runtime->get_num_sms();
    const auto config = get_sm90_mega_moe_backward_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    constexpr int kBlockM = kMegaMoEBackwardBlockM;

    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, 0, false,
        num_shared_experts);
    const auto bwd_buffer = layout::MegaMoEBackwardBuffer(
        mega_buffer.get_end_ptr(), hidden, intermediate_hidden, num_max_tokens_per_rank, num_topk, num_shared_experts,
        mega_buffer.workspace.num_max_pool_tokens, num_sms);
    const auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(sym_buffer.device());
    const auto view = [&](void* offset, const int64_t& rows, const int64_t& cols) {
        return torch::from_blob(math::advance_ptr(sym_buffer.data_ptr(), reinterpret_cast<int64_t>(offset)), {rows, cols}, bf16_opts);
    };
    const auto num_pool_rows = static_cast<int64_t>(bwd_buffer.num_pool_rows);
    const auto x_pool = view(bwd_buffer.x_pool, num_pool_rows, hidden);
    const auto dy_pool = view(bwd_buffer.dy_pool, num_pool_rows, hidden);
    const auto dz_pool = view(bwd_buffer.dz_pool, 2 * intermediate_hidden, num_pool_rows);
    const auto hw_pool = view(bwd_buffer.hw_pool, intermediate_hidden, num_pool_rows);

    const auto tensor_map_w1_k = l1_natural_layout ?
        make_tma_gate_up_natural_desc(w1_weights, hidden, intermediate_hidden, num_experts_per_rank, 128, 128) :
        make_tma_2d_desc(w1_weights, hidden, num_experts_per_rank * 2 * intermediate_hidden, 64, 128, hidden, 128);
    const auto tensor_map_w1_mn = l1_natural_layout ?
        make_tma_gate_up_natural_desc(w1_weights, hidden, intermediate_hidden, num_experts_per_rank, 64, 128) :
        make_tma_2d_desc(w1_weights, hidden, num_experts_per_rank * 2 * intermediate_hidden, 64, 64, hidden, 128);
    const auto tensor_map_w2_mn = make_tma_2d_desc(w2_weights, intermediate_hidden, num_experts_per_rank * hidden, 64, 64, intermediate_hidden, 128);
    const auto tensor_map_shared_w1_k = num_shared_experts == 0 ? tensor_map_w1_k :
        l1_natural_layout ?
        make_tma_gate_up_natural_desc(*shared_w1_weights, hidden, num_shared_experts * intermediate_hidden, 1, 128, 128) :
        make_tma_2d_desc(*shared_w1_weights, hidden, num_shared_experts * 2 * intermediate_hidden, 64, 128, hidden, 128);
    const auto tensor_map_shared_w1_mn = num_shared_experts == 0 ? tensor_map_w1_mn :
        l1_natural_layout ?
        make_tma_gate_up_natural_desc(*shared_w1_weights, hidden, num_shared_experts * intermediate_hidden, 1, 64, 128) :
        make_tma_2d_desc(*shared_w1_weights, hidden, num_shared_experts * 2 * intermediate_hidden, 64, 64, hidden, 128);
    const auto tensor_map_shared_w2_mn = num_shared_experts > 0 ?
        make_tma_2d_desc(*shared_w2_weights, num_shared_experts * intermediate_hidden, hidden, 64, 64, num_shared_experts * intermediate_hidden, 128) : tensor_map_w2_mn;
    const auto tensor_map_x_k = make_tma_2d_desc(x_pool, hidden, static_cast<int>(num_pool_rows), 64, kBlockM, hidden, 128);
    const auto tensor_map_x_mn = make_tma_2d_desc(x_pool, hidden, static_cast<int>(num_pool_rows), 64, 64, hidden, 128);
    const auto tensor_map_dy_k = make_tma_2d_desc(dy_pool, hidden, static_cast<int>(num_pool_rows), 64, kBlockM, hidden, 128);
    const auto tensor_map_dy_mn = make_tma_2d_desc(dy_pool, hidden, static_cast<int>(num_pool_rows), 64, 64, hidden, 128);
    const auto tensor_map_dz_k = make_tma_2d_desc(dz_pool, static_cast<int>(num_pool_rows), 2 * intermediate_hidden, 64, 128, static_cast<int>(num_pool_rows), 128);
    const auto tensor_map_dz_mn = make_tma_2d_desc(dz_pool, static_cast<int>(num_pool_rows), 2 * intermediate_hidden, 64, 64, static_cast<int>(num_pool_rows), 128);
    const auto tensor_map_hw_k = make_tma_2d_desc(hw_pool, static_cast<int>(num_pool_rows), intermediate_hidden, 64, 128, static_cast<int>(num_pool_rows), 128);

    void* shared_dw1_ptr = nullptr;
    void* shared_dw2_ptr = nullptr;
    if (shared_w1_weights != nullptr) {
        shared_dw1_ptr = shared_dw1_weights->data_ptr();
        shared_dw2_ptr = shared_dw2_weights->data_ptr();
    }
    const std::string dw_type = dw1_weights.scalar_type() == torch::kBFloat16 ? "nv_bfloat16" : "float";

    const auto kernel = jit->compile("sm90_bf16_mega_moe_backward", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_moe_backward_impl<
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        num_ring_tokens,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        config.num_stages,
        dw_type, dw_natural_layout ? "true" : "false",
        l1_natural_layout ? "true" : "false"));
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_threads, 1, 1),
        },
        dx.data_ptr(),
        dw1_weights.data_ptr(), dw2_weights.data_ptr(), dtopk_weights.data_ptr<float>(),
        shared_dw1_ptr, shared_dw2_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_w1_k, tensor_map_w1_mn, tensor_map_w2_mn,
        tensor_map_shared_w1_k, tensor_map_shared_w1_mn, tensor_map_shared_w2_mn,
        tensor_map_x_k, tensor_map_x_mn, tensor_map_dy_k, tensor_map_dy_mn,
        tensor_map_dz_k, tensor_map_dz_mn, tensor_map_hw_k);
}

}
