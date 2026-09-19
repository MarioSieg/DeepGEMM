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
    const torch::Tensor& sym_buffer,
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
    const auto num_sms = runtime->get_num_sms();
    const auto config = get_mega_moe_backward_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    constexpr int kBlockM = kMegaMoEBackwardBlockM;
    const int num_passes = num_shared_experts > 0 ? num_shared_experts : 1;

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
    const auto stage_x = view(bwd_buffer.stage_x, static_cast<int64_t>(num_sms) * layout::MegaMoEBackwardBuffer::kNumStageSlots * kBlockM, hidden);
    const auto stage_dy = view(bwd_buffer.stage_dy, static_cast<int64_t>(num_sms) * layout::MegaMoEBackwardBuffer::kNumStageSlots * kBlockM, hidden);
    const auto hw_scratch = view(bwd_buffer.hw_scratch, static_cast<int64_t>(num_sms) * intermediate_hidden, kBlockM);
    const auto dz_scratch = view(bwd_buffer.dz_scratch, static_cast<int64_t>(num_sms) * num_passes * 2 * intermediate_hidden, kBlockM);

    const auto tensor_map_w1_k = make_tma_2d_desc(w1_weights, hidden, num_experts_per_rank * 2 * intermediate_hidden, 64, 128, hidden, 128);
    const auto tensor_map_w1_mn = make_tma_2d_desc(w1_weights, hidden, num_experts_per_rank * 2 * intermediate_hidden, 64, 64, hidden, 128);
    const auto tensor_map_w2_mn = make_tma_2d_desc(w2_weights, intermediate_hidden, num_experts_per_rank * hidden, 64, 64, intermediate_hidden, 128);
    const auto tensor_map_shared_w1_k = num_shared_experts > 0 ?
        make_tma_2d_desc(*shared_w1_weights, hidden, num_shared_experts * 2 * intermediate_hidden, 64, 128, hidden, 128) : tensor_map_w1_k;
    const auto tensor_map_shared_w1_mn = num_shared_experts > 0 ?
        make_tma_2d_desc(*shared_w1_weights, hidden, num_shared_experts * 2 * intermediate_hidden, 64, 64, hidden, 128) : tensor_map_w1_mn;
    const auto tensor_map_shared_w2_mn = num_shared_experts > 0 ?
        make_tma_2d_desc(*shared_w2_weights, num_shared_experts * intermediate_hidden, hidden, 64, 64, num_shared_experts * intermediate_hidden, 128) : tensor_map_w2_mn;
    const auto tensor_map_stage_x = make_tma_2d_desc(stage_x, hidden, static_cast<int>(stage_x.size(0)), 64, kBlockM, hidden, 128);
    const auto tensor_map_stage_dy = make_tma_2d_desc(stage_dy, hidden, static_cast<int>(stage_dy.size(0)), 64, kBlockM, hidden, 128);
    const auto tensor_map_hw = make_tma_2d_desc(hw_scratch, kBlockM, static_cast<int>(hw_scratch.size(0)), kBlockM, 128, kBlockM, 64);
    const auto tensor_map_dz_k = make_tma_2d_desc(dz_scratch, kBlockM, static_cast<int>(dz_scratch.size(0)), kBlockM, 128, kBlockM, 64);
    const auto tensor_map_dz_mn = make_tma_2d_desc(dz_scratch, kBlockM, static_cast<int>(dz_scratch.size(0)), kBlockM, 64, kBlockM, 64);

    float* shared_dw1_ptr = nullptr;
    float* shared_dw2_ptr = nullptr;
    if (shared_w1_weights != nullptr) {
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
        {}, {},
        {}, {},
        {}
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
        config.num_stages));
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_threads, 1, 1),
        },
        dx.data_ptr(),
        dw1_weights.data_ptr<float>(), dw2_weights.data_ptr<float>(), dtopk_weights.data_ptr<float>(),
        shared_dw1_ptr, shared_dw2_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_w1_k, tensor_map_w1_mn, tensor_map_w2_mn,
        tensor_map_shared_w1_k, tensor_map_shared_w1_mn, tensor_map_shared_w2_mn,
        tensor_map_stage_x, tensor_map_stage_dy,
        tensor_map_hw, tensor_map_dz_k, tensor_map_dz_mn);
}

}
