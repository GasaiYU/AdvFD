"""DDP via the _TrainForward wrapper must match manual gradient all-reduce.

main_fd.py cannot be imported here (it pulls in wandb), so the wrapper is
reproduced below. The property under test is the one that matters: DDP only
instruments ``forward()``, so a generator driven through a custom method such as
``sample_images_with_grad`` needs this indirection or DDP silently does nothing.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from utils.distributed_util import all_reduce_grads


class _Generator(nn.Module):
    """Stands in for the pMF/JiT denoisers: training runs off a custom method."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 16)
        )

    def forward(self, x):  # deliberately not the training entry point
        return self.net(x)

    def sample_images_with_grad(self, z, y, sampling_args=None):
        num_steps = (sampling_args or {}).get("num_steps", 1)
        for _ in range(num_steps):
            z = z - 0.1 * self.net(z)
        return z


class _TrainForward(nn.Module):
    """Mirror of main_fd._TrainForward."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, z, y, sampling_args):
        return self.module.sample_images_with_grad(z, y, sampling_args=sampling_args)


def _batch(rank):
    torch.manual_seed(100 + rank)
    return torch.randn(4, 16), torch.zeros(4, dtype=torch.long)


def _worker(rank, world_size, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    z, y = _batch(rank)
    sampling_args = {"num_steps": 1}

    # Path A: DDP through the wrapper.
    ddp = nn.parallel.DistributedDataParallel(_TrainForward(_Generator()))
    ddp(z, y, sampling_args).square().mean().backward()
    ddp_grads = [p.grad.clone() for p in ddp.module.module.parameters()]

    # Path B: raw module plus explicit bucketed all-reduce.
    manual = _Generator()
    manual.sample_images_with_grad(z, y, sampling_args).square().mean().backward()
    all_reduce_grads(manual)
    manual_grads = [p.grad.clone() for p in manual.parameters()]

    # Path C: bypassing DDP via .module, the mistake this wrapper prevents.
    bypass_ddp = nn.parallel.DistributedDataParallel(_TrainForward(_Generator()))
    inner = bypass_ddp.module.module
    inner.sample_images_with_grad(z, y, sampling_args).square().mean().backward()
    bypass_grads = [p.grad.clone() for p in inner.parameters()]

    out[rank] = (
        max((a - b).abs().max().item() for a, b in zip(ddp_grads, manual_grads)),
        max((a - b).abs().max().item() for a, b in zip(bypass_grads, manual_grads)),
        sum(g.abs().sum().item() for g in ddp_grads),
    )
    dist.destroy_process_group()


def _run(world_size=2, port=29811):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    mp.start_processes(
        _worker, args=(world_size, port, out), nprocs=world_size, start_method="spawn"
    )
    return dict(out)


def test_ddp_matches_manual_all_reduce():
    for rank, (ddp_vs_manual, _, grad_mass) in _run().items():
        assert grad_mass > 0, f"rank {rank} produced no gradients"
        assert ddp_vs_manual < 1e-6, (
            f"rank {rank}: DDP disagrees with manual all-reduce by {ddp_vs_manual}"
        )


def test_calling_through_module_skips_ddp_sync():
    """Guards the reason the wrapper exists.

    Reaching past DDP to the inner module leaves gradients unsynced: each rank
    keeps its local values. If this ever stops differing, DDP is no longer doing
    the sync and the wrapper indirection has been defeated.
    """
    results = _run(port=29813)
    assert any(
        bypass_vs_manual > 1e-6 for _, bypass_vs_manual, _ in results.values()
    ), "bypassing DDP should leave gradients unsynced across ranks"
