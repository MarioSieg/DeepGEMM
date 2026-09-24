import argparse
import random
from typing import List, Optional

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.mega import _interleave_weights
from deep_gemm.testing import calc_diff
from deep_gemm.utils.dist import init_dist


def swiglu_mlp_forward(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                        clamp: Optional[float]) -> torch.Tensor:
    z = x @ w1.t()
    intermediate = w1.shape[0] // 2
    gate, up = z[:, :intermediate], z[:, intermediate:]
    if clamp is not None:
        gate = torch.clamp(gate, max=clamp)
        up = torch.clamp(up, -clamp, clamp)
    h = gate * torch.sigmoid(gate) * up
    return h @ w2.t()


def moe_reference_forward(x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                          l1_weights: torch.Tensor, l2_weights: torch.Tensor,
                          shared_l1_weights: List[Optional[torch.Tensor]],
                          shared_l2_weights: List[Optional[torch.Tensor]],
                          tokens_per_rank: int, num_ranks: int,
                          clamp: Optional[float]) -> torch.Tensor:
    num_experts = l1_weights.shape[0]
    y = torch.zeros_like(x)
    for e in range(num_experts):
        mask = topk_idx == e
        if not mask.any():
            continue
        token_rows, topk_cols = mask.nonzero(as_tuple=True)
        oe = swiglu_mlp_forward(x[token_rows], l1_weights[e], l2_weights[e], clamp)
        we = topk_weights[token_rows, topk_cols].unsqueeze(-1)
        y = y.index_add(0, token_rows, oe * we)
    if shared_l1_weights[0] is not None:
        shared_chunks = []
        for r in range(num_ranks):
            lo, hi = r * tokens_per_rank, (r + 1) * tokens_per_rank
            shared_chunks.append(swiglu_mlp_forward(x[lo:hi], shared_l1_weights[r], shared_l2_weights[r], clamp))
        y = y + torch.cat(shared_chunks, dim=0)
    return y


# noinspection PyUnboundLocalVariable,PyShadowingNames
def _test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_tokens = args.num_tokens
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    num_shared_experts = args.num_shared_experts
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    clamp = args.activation_clamp

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_tokens, num_topk, hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts, mma_type='bf16xbf16')

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    dy = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden),
                             dtype=torch.bfloat16, device='cuda')
    l2_weights = torch.randn((num_experts_per_rank, hidden, intermediate_hidden),
                             dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx = topk_idx.masked_fill(rand_mask < args.masked_ratio, -1)
        topk_weights = topk_weights.masked_fill(topk_idx < 0, 0)

    has_shared = num_shared_experts > 0
    if has_shared:
        shared_l1_weights = torch.randn((intermediate_hidden * 2 * num_shared_experts, hidden),
                                        dtype=torch.bfloat16, device='cuda')
        shared_l2_weights = torch.randn((hidden, intermediate_hidden * num_shared_experts),
                                        dtype=torch.bfloat16, device='cuda')
    else:
        shared_l1_weights = shared_l2_weights = None

    if args.l1_natural:
        assert not has_shared, 'Natural L1 weights do not support shared experts'
        transformed_l1_weights, transformed_l2_weights = l1_weights, l2_weights
    else:
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            l1_weights, l2_weights)
    if has_shared:
        transformed_shared_l1_weights, transformed_shared_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            shared_l1_weights, shared_l2_weights)
    else:
        transformed_shared_l1_weights = transformed_shared_l2_weights = None

    buffer.x[:num_tokens].copy_(x)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    dist.barrier()
    torch.cuda.synchronize()

    def run_forward():
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.bf16_mega_moe(
            y, transformed_l1_weights, transformed_l2_weights, buffer,
            shared_l1_weights=transformed_shared_l1_weights, shared_l2_weights=transformed_shared_l2_weights,
            activation_clamp=clamp, fast_math=bool(args.fast_math), l1_natural_layout=bool(args.l1_natural))
        return y

    dw_dtype = torch.bfloat16 if args.natural_bf16 else torch.float32

    def run_backward():
        dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        dw1_weights = torch.empty_like(l1_weights, dtype=dw_dtype)
        dw2_weights = torch.empty_like(l2_weights, dtype=dw_dtype)
        dtopk_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')
        shared_dw1_weights = torch.empty_like(shared_l1_weights, dtype=dw_dtype) if has_shared else None
        shared_dw2_weights = torch.empty_like(shared_l2_weights, dtype=dw_dtype) if has_shared else None
        deep_gemm.bf16_mega_moe_backward(
            dx, dw1_weights, dw2_weights, dtopk_weights, dy,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            shared_l1_weights=transformed_shared_l1_weights, shared_l2_weights=transformed_shared_l2_weights,
            shared_dw1_weights=shared_dw1_weights, shared_dw2_weights=shared_dw2_weights,
            activation_clamp=clamp, fast_math=bool(args.fast_math), dw_natural_layout=bool(args.natural_bf16),
            l1_natural_layout=bool(args.l1_natural))
        return dx, dw1_weights, dw2_weights, dtopk_weights, shared_dw1_weights, shared_dw2_weights

    y_first = run_forward()
    first = run_backward()
    y_second = run_forward()
    dx, dw1_weights, dw2_weights, dtopk_weights, shared_dw1_weights, shared_dw2_weights = run_backward()
    torch.cuda.synchronize()
    dist.barrier()
    assert torch.equal(y_first, y_second), f'[rank {rank_idx}] forward output changed after a backward on the same buffer'
    for name, a, b in zip(('dx', 'dw1', 'dw2', 'dtopk'), first, (dx, dw1_weights, dw2_weights, dtopk_weights)):
        assert calc_diff(a, b) < 1e-6, f'[rank {rank_idx}] {name} differs between two backward launches'

    def gather(t: torch.Tensor) -> torch.Tensor:
        out = [torch.empty_like(t) for _ in range(num_ranks)]
        dist.all_gather(out, t, group=group)
        return torch.cat(out, dim=0)

    x_all = gather(x).float().requires_grad_(True)
    dy_all = gather(dy).float()
    topk_idx_all = gather(topk_idx)
    topk_weights_all = gather(topk_weights).float().requires_grad_(True)
    l1_all = gather(l1_weights).float().requires_grad_(True)  # [num_experts, 2I, H]
    l2_all = gather(l2_weights).float().requires_grad_(True)  # [num_experts, H, I]

    shared_l1_list, shared_l2_list = [], []
    if has_shared:
        shared_l1_all = gather(shared_l1_weights).float().view(num_ranks, -1, hidden).requires_grad_(True)
        shared_l2_all = gather(shared_l2_weights).float().view(num_ranks, hidden, -1).requires_grad_(True)
        for r in range(num_ranks):
            shared_l1_list.append(shared_l1_all[r])
            shared_l2_list.append(shared_l2_all[r])
    else:
        shared_l1_all = shared_l2_all = None
        shared_l1_list = [None] * num_ranks
        shared_l2_list = [None] * num_ranks

    y_all = moe_reference_forward(
        x_all, topk_idx_all, topk_weights_all, l1_all, l2_all,
        shared_l1_list, shared_l2_list, num_tokens, num_ranks, clamp)
    y_all.backward(dy_all)

    lo, hi = rank_idx * num_tokens, (rank_idx + 1) * num_tokens
    elo, ehi = rank_idx * num_experts_per_rank, (rank_idx + 1) * num_experts_per_rank

    dx_ref = x_all.grad[lo:hi]
    dw1_ref = l1_all.grad[elo:ehi] if args.natural_bf16 else _interleave_weights(l1_all.grad[elo:ehi])
    dw2_ref = l2_all.grad[elo:ehi]
    dtopk_ref = topk_weights_all.grad[lo:hi]

    errs = {
        'dx': calc_diff(dx.float(), dx_ref),
        'dw1': calc_diff(dw1_weights.float(), dw1_ref),
        'dw2': calc_diff(dw2_weights.float(), dw2_ref),
        'dtopk_weights': calc_diff(dtopk_weights, dtopk_ref),
    }
    if has_shared:
        dshared_w1_ref = shared_l1_all.grad[rank_idx] if args.natural_bf16 else _interleave_weights(shared_l1_all.grad[rank_idx])
        dshared_w2_ref = shared_l2_all.grad[rank_idx]
        errs['shared_dw1'] = calc_diff(shared_dw1_weights.float(), dshared_w1_ref)
        errs['shared_dw2'] = calc_diff(shared_dw2_weights.float(), dshared_w2_ref)

    print(f'[rank {rank_idx}] calc_diff: ' +
          ', '.join(f'{k}={v:.6f}' for k, v in errs.items()), flush=True)
    for k, v in errs.items():
        assert v < args.tolerance, f'[rank {rank_idx}] {k} diff {v} exceeds tolerance {args.tolerance}'

    dist.barrier()
    if rank_idx == 0:
        print('All ranks passed the Mega MoE backward correctness check.', flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mega MoE backward correctness test')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of ranks to spawn')
    parser.add_argument('--num-tokens', type=int, default=32, help='Tokens per rank')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=128, help='Intermediate hidden size')
    parser.add_argument('--num-experts', type=int, default=8, help='Total number of routed experts')
    parser.add_argument('--num-topk', type=int, default=2, help='Number of expert selections')
    parser.add_argument('--num-shared-experts', type=int, default=1, help='Number of shared experts (0 to disable)')
    parser.add_argument('--activation-clamp', type=float, default=8.0, help='SwiGLU activation clamp')
    parser.add_argument('--masked-ratio', type=float, default=0.1, help='Fraction of topk slots to mask out')
    parser.add_argument('--fast-math', type=int, default=0, help='Enable fast math (0 or 1); 0 for a tight check')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Max allowed `calc_diff` (global relative error)')
    parser.add_argument('--natural-bf16', type=int, default=0, help='Write dW in bf16 and the untransformed [gate; up] layout')
    parser.add_argument('--l1-natural', type=int, default=0, help='Pass L1 weights in the untransformed [gate; up] layout (requires --num-shared-experts 0)')
    args = parser.parse_args()

    torch.multiprocessing.spawn(_test, args=(args.num_processes, args), nprocs=args.num_processes)
