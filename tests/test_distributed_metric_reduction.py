import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from amfd.integration import _accum_log
from utils.distributed_util import all_reduce_mean_many, all_ranks_true


def test_all_reduce_mean_many_local_mixed_scalars():
    tensor_value = torch.tensor(3.5, requires_grad=True)

    reduced = all_reduce_mean_many([1, 2.25, True, tensor_value])

    assert reduced == pytest.approx([1.0, 2.25, 1.0, 3.5])
    assert all(isinstance(value, float) for value in reduced)
    assert tensor_value.grad is None


def test_all_reduce_mean_many_validates_inputs():
    with pytest.raises(ValueError, match="scalar tensors"):
        all_reduce_mean_many([torch.ones(2)])
    with pytest.raises(TypeError, match="complex metrics"):
        all_reduce_mean_many([torch.tensor(1.0 + 2.0j)])
    with pytest.raises(TypeError, match="real numbers"):
        all_reduce_mean_many([None])


def test_amfd_metric_accumulation_stays_detached():
    logs = {}
    value = torch.tensor(2.0, requires_grad=True)

    _accum_log(logs, "metric", value, scale=0.5)
    _accum_log(logs, "metric", torch.tensor(4.0), scale=0.5)

    assert isinstance(logs["metric"], torch.Tensor)
    assert not logs["metric"].requires_grad
    assert logs["metric"].item() == pytest.approx(3.0)


def _gloo_metric_worker(rank, world_size, init_file):
    # macOS CI/sandboxes may not resolve the loopback hostname automatically.
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME",
        "lo0" if sys.platform == "darwin" else "lo",
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    original_all_reduce = dist.all_reduce
    calls = 0

    def counted_all_reduce(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_all_reduce(*args, **kwargs)

    dist.all_reduce = counted_all_reduce
    try:
        reduced = all_reduce_mean_many([
            float(rank + 1),
            torch.tensor(float(10 * (rank + 1))),
            rank == 0,
        ])
        assert calls == 1
        assert reduced == pytest.approx([1.5, 15.0, 0.5])

        calls = 0
        assert not all_ranks_true(torch.tensor(rank == 0))
        assert calls == 1
    finally:
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_all_reduce_mean_many_uses_one_gloo_collective():
    fd, init_file = tempfile.mkstemp(prefix="advfd-metrics-")
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(
            _gloo_metric_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
