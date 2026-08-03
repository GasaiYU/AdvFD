"""Run the real pMF and JiT denoisers under DDP, 2 ranks, gloo/CPU.

This is the closest local check to the cluster path: the actual generator, the
actual ``sample_images_with_grad`` entry point, and the actual _TrainForward
indirection. Catches the failure modes that matter before a multi-node launch:

* DDP raising on parameters that never receive a gradient
  (``find_unused_parameters=False`` is the default)
* the wrapper failing to route a custom method through ``DDP.forward``
* gradients disagreeing with the manual all-reduce path

Small img_size keeps it runnable on CPU; the parameter *structure* that DDP
cares about is unchanged.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from utils.distributed_util import all_reduce_grads


class _TrainForward(nn.Module):
    """Mirror of main_fd._TrainForward."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, z, y, sampling_args):
        return self.module.sample_images_with_grad(z, y, sampling_args=sampling_args)


def _build(kind, img_size=32):
    torch.manual_seed(0)
    if kind == "jit":
        from models.denoiser_jit import JiTDenoiser

        return JiTDenoiser(
            model_size="base", img_size=img_size, in_channels=3, num_classes=100,
            rope_2d=True, learned_pe=True, legacy_time_convention=True,
        )
    from models.denoiser_pmf import pMFDenoiser_models

    return pMFDenoiser_models["pMF_B"](
        img_size=img_size, patch_size=2, in_channels=3, num_classes=100,
        rope_2d=True, learned_pe=True, disable_v_head=True,
    )


def _loss(out):
    x = out[0] if isinstance(out, tuple) else out
    return x.square().mean()


def _worker(rank, world_size, port, kind, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(7 + rank)

    model = _build(kind)
    model.train()
    img = model.input_size if hasattr(model, "input_size") else 32
    z = torch.randn(2, 3, img, img)
    y = torch.randint(0, 100, (2,))
    sampling_args = {"num_steps": 1, "cfg": 1.0}

    error = None
    try:
        ddp = nn.parallel.DistributedDataParallel(_TrainForward(model))
        _loss(ddp(z, y, sampling_args)).backward()
        ddp_g = {n: p.grad.clone() for n, p in ddp.module.module.named_parameters()
                 if p.grad is not None}
        missing = [n for n, p in ddp.module.module.named_parameters()
                   if p.requires_grad and p.grad is None]
    except Exception as exc:  # noqa: BLE001 - reported to the parent process
        out[rank] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.destroy_process_group()
        return

    manual = _build(kind)
    manual.train()
    _loss(manual.sample_images_with_grad(z, y, sampling_args=sampling_args)).backward()
    all_reduce_grads(manual)
    manual_g = {n: p.grad for n, p in manual.named_parameters() if p.grad is not None}

    max_diff = max(
        (ddp_g[n] - manual_g[n]).abs().max().item() for n in ddp_g if n in manual_g
    )
    out[rank] = {
        "error": error,
        "n_with_grad": len(ddp_g),
        "n_missing": len(missing),
        "missing": missing[:5],
        "max_diff": max_diff,
        "grad_scale": max(g.abs().max().item() for g in manual_g.values()),
    }
    dist.destroy_process_group()


def _run(kind, port, world_size=2):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    mp.start_processes(
        _worker, args=(world_size, port, kind, out), nprocs=world_size,
        start_method="spawn",
    )
    return dict(out)


@pytest.mark.parametrize("kind,port", [("jit", 30011), ("pmf", 30013)])
def test_ddp_runs_and_matches_manual(kind, port):
    for rank, r in _run(kind, port).items():
        assert r.get("error") is None, f"{kind} rank {rank} raised: {r['error']}"
        assert r["n_missing"] == 0, (
            f"{kind} rank {rank}: {r['n_missing']} params without gradient "
            f"(DDP would raise): {r['missing']}"
        )
        assert r["n_with_grad"] > 50
        tol = 1e-5 * max(r["grad_scale"], 1.0)
        assert r["max_diff"] <= tol, (
            f"{kind} rank {rank}: DDP vs manual {r['max_diff']:.3e} > {tol:.3e}"
        )
