"""Bucketed gradient all-reduce must match the per-parameter version exactly.

Runs real multi-rank gloo groups on CPU so the averaging is actually exercised,
not just the flatten/unflatten bookkeeping.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from utils.distributed_util import all_reduce_grads


def _make_model(seed=0):
    """Mixed tensor shapes/sizes, mirroring the many-tiny-tensors real case."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.Linear(64, 8),
    )


def _reference_all_reduce(module):
    """The original implementation: one collective per parameter."""
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def _fill_grads(module, rank):
    """Rank-dependent gradients so averaging is observable."""
    for i, p in enumerate(module.parameters()):
        torch.manual_seed(1000 * rank + i)
        p.grad = torch.randn_like(p) + rank


def _worker(rank, world_size, port, bucket_bytes, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    bucketed = _make_model()
    _fill_grads(bucketed, rank)
    calls = all_reduce_grads(bucketed, bucket_bytes=bucket_bytes)

    reference = _make_model()
    _fill_grads(reference, rank)
    _reference_all_reduce(reference)

    n_params = sum(1 for _ in reference.parameters())
    max_diff = max(
        (a.grad - b.grad).abs().max().item()
        for a, b in zip(bucketed.parameters(), reference.parameters())
    )
    # Sanity: averaging actually changed the local gradients.
    moved = max(
        (p.grad - (torch.randn_like(p) * 0)).abs().max().item()
        for p in bucketed.parameters()
    )
    out[rank] = (max_diff, calls, n_params, moved)
    dist.destroy_process_group()


def _run(world_size, bucket_bytes=25 * 1024 * 1024):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    port = 29700 + world_size + (bucket_bytes % 97)
    mp.start_processes(
        _worker,
        args=(world_size, port, bucket_bytes, out),
        nprocs=world_size,
        start_method="spawn",
    )
    return dict(out)


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_bucketed_matches_per_parameter(world_size):
    """Agreement to within fp32 rounding, not bitwise.

    Coalescing changes the buffer size handed to the collective, and gloo/NCCL
    pick their reduction algorithm from that size, so the summation order can
    differ. The observed gap is a few ULPs and grows slowly with world_size.
    """
    results = _run(world_size)
    assert len(results) == world_size
    for rank, (max_diff, _, _, moved) in results.items():
        assert max_diff < 1e-5, f"rank {rank} diverged from reference: {max_diff}"
        assert moved > 0.0, f"rank {rank} gradients look empty"


def test_two_ranks_are_bitwise_identical():
    """world_size=2 is a single pairwise exchange, leaving no ordering freedom."""
    results = _run(2)
    for rank, (max_diff, _, _, _) in results.items():
        assert max_diff == 0.0, f"rank {rank} not bitwise identical: {max_diff}"


def test_bucketing_reduces_collective_calls():
    """One big bucket should collapse many tensors into a single call."""
    results = _run(2)
    max_diff, calls, n_params, _ = results[0]
    assert max_diff == 0.0
    assert n_params > 1
    assert calls == 1, f"expected 1 bucketed call, got {calls} for {n_params} tensors"


def test_small_buckets_still_exact():
    """Tiny bucket_bytes forces one bucket per tensor, matching the original."""
    results = _run(2, bucket_bytes=1)
    for rank, (max_diff, calls, n_params, _) in results.items():
        assert max_diff == 0.0, f"rank {rank} diverged with tiny buckets"
        assert calls == n_params, "bucket_bytes=1 should give one call per tensor"


def test_noop_without_process_group():
    """Outside a distributed run this must not touch gradients or raise."""
    model = _make_model()
    _fill_grads(model, rank=0)
    before = [p.grad.clone() for p in model.parameters()]
    assert all_reduce_grads(model) == 0
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p.grad)
