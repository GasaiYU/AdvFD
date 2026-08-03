"""Compare DDP, bucketed manual, and per-parameter reduction across many buckets.

The other tests use a model small enough to fit one bucket, which hides any
difference between the three paths. This one forces many buckets, matching the
real case (pMF_L is 443 tensors across ~55 buckets).
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from utils.distributed_util import all_reduce_grads

BUCKET_BYTES = 64 * 1024


class _Gen(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            *[nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256)) for _ in range(12)]
        )

    def sample_images_with_grad(self, z, y, sampling_args=None):
        return self.net(z)


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, z, y, sampling_args):
        return self.module.sample_images_with_grad(z, y, sampling_args=sampling_args)


def _worker(rank, world_size, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(50 + rank)
    z = torch.randn(4, 256)

    ddp = nn.parallel.DistributedDataParallel(
        _Wrap(_Gen()), bucket_cap_mb=BUCKET_BYTES / (1024 * 1024)
    )
    ddp(z, None, {}).square().mean().backward()
    ddp_g = [p.grad.clone() for p in ddp.module.module.parameters()]

    bucketed = _Gen()
    bucketed.sample_images_with_grad(z, None).square().mean().backward()
    all_reduce_grads(bucketed, bucket_bytes=BUCKET_BYTES)
    buck_g = [p.grad.clone() for p in bucketed.parameters()]

    per_param = _Gen()
    per_param.sample_images_with_grad(z, None).square().mean().backward()
    for p in per_param.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    ref_g = [p.grad.clone() for p in per_param.parameters()]

    scale = max(g.abs().max().item() for g in ref_g)
    out[rank] = {
        "ddp_vs_bucketed": max((a - b).abs().max().item() for a, b in zip(ddp_g, buck_g)),
        "ddp_vs_per_param": max((a - b).abs().max().item() for a, b in zip(ddp_g, ref_g)),
        "bucketed_vs_per_param": max((a - b).abs().max().item() for a, b in zip(buck_g, ref_g)),
        "n_tensors": len(ddp_g),
        "grad_scale": scale,
    }
    dist.destroy_process_group()


def _run(world_size, port):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    mp.start_processes(
        _worker, args=(world_size, port, out), nprocs=world_size, start_method="spawn"
    )
    return dict(out)


@pytest.mark.parametrize("world_size,port", [(2, 29962), (4, 29964)])
def test_all_three_paths_agree_to_fp32_rounding(world_size, port):
    results = _run(world_size, port)
    for rank, r in results.items():
        assert r["n_tensors"] > 40, "model should span many buckets"
        rel = r["grad_scale"]
        for key in ("ddp_vs_bucketed", "ddp_vs_per_param", "bucketed_vs_per_param"):
            assert r[key] <= 1e-5 * max(rel, 1.0), f"rank {rank} {key}={r[key]:.3e}"


def test_report_differences(capsys):
    """Not an assertion, just records the measured gaps."""
    for world_size, port in [(2, 29966), (4, 29968)]:
        r = _run(world_size, port)[0]
        with capsys.disabled():
            print(
                f"\nws={world_size} tensors={r['n_tensors']} "
                f"grad_scale={r['grad_scale']:.3e}\n"
                f"  DDP vs bucketed      : {r['ddp_vs_bucketed']:.3e}\n"
                f"  DDP vs per-param     : {r['ddp_vs_per_param']:.3e}\n"
                f"  bucketed vs per-param: {r['bucketed_vs_per_param']:.3e}"
            )
