"""ZeRO-1 sharding of the AMFD amortizer optimizer.

The claim this change rests on is that sharding the AdamW moments over the
process group is a memory-layout change and nothing else: the update arithmetic,
and therefore the trained weights, must be identical to the replicated
optimizer.  Single-process tests cannot show that, so everything here spawns
real multi-rank gloo groups on CPU and compares against a locally computed
replicated reference.

The gradients handed to the optimizer are deliberately made rank-independent,
because that is the invariant the training loop already guarantees:
``update_amortizers`` all-reduces amortizer gradients before stepping, so every
rank enters ``step()`` with the same gradients and the same clip scale.
"""
import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import amfd.integration as I  # noqa: E402

LR, BETA1, BETA2, WD = 1e-4, 0.9, 0.95, 0.0


def _opt_args(**kw):
    """Minimal stand-in for the argparse namespace the builder reads."""
    args = types.SimpleNamespace(
        amort_lr=LR, amort_beta1=BETA1, amort_beta2=BETA2, amort_weight_decay=WD,
    )
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _make_model(seed=0):
    """Mixed shapes and sizes so the partition is not a trivial even split."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.Linear(64, 8),
    )


def _fill_grads(module, step):
    """Rank-independent gradients: the post-all-reduce state of a real step."""
    for i, p in enumerate(module.parameters()):
        torch.manual_seed(7717 * step + i)
        p.grad = torch.randn_like(p)


def _flat_params(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _local_shard_numel(optimizer):
    """Number of moment elements this rank actually holds."""
    inner = optimizer.optim
    return sum(
        v.numel()
        for state in inner.state.values()
        for k, v in state.items()
        if torch.is_tensor(v) and k in ("exp_avg", "exp_avg_sq")
    )


def _run(worker, world_size, *args, port_salt=0):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    port = 29800 + world_size * 7 + (port_salt % 53)
    mp.start_processes(
        worker,
        args=(world_size, port, *args, out),
        nprocs=world_size,
        start_method="spawn",
    )
    return dict(out)


def _init(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# ---------------------------------------------------------------------------
# The optimizer the builder hands back
# ---------------------------------------------------------------------------

def _worker_builder_picks_zero(rank, world_size, port, out):
    _init(rank, world_size, port)
    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())

    group = opt.param_groups[0]
    out[rank] = (
        type(opt).__name__,
        I._is_zero_optimizer(opt),
        group["lr"],
        tuple(group["betas"]),
        group["weight_decay"],
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_builder_shards_when_distributed_and_keeps_hyperparameters(world_size):
    results = _run(_worker_builder_picks_zero, world_size, port_salt=1)
    assert len(results) == world_size
    for rank, (name, is_zero, lr, betas, wd) in results.items():
        assert name == "ZeroRedundancyOptimizer", f"rank {rank} got {name}"
        assert is_zero
        # The whole point of the change is that it costs no hyperparameter drift.
        assert lr == LR
        assert betas == (BETA1, BETA2)
        assert wd == WD


def test_builder_falls_back_to_adamw_on_single_process():
    """No process group means nothing to shard over."""
    assert not dist.is_initialized()
    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())

    assert isinstance(opt, torch.optim.AdamW)
    assert not I._is_zero_optimizer(opt)
    group = opt.param_groups[0]
    assert (group["lr"], tuple(group["betas"]), group["weight_decay"]) == (
        LR, (BETA1, BETA2), WD,
    )


def test_builder_accepts_a_generator():
    """``.parameters()`` is a generator and must not be consumed before use."""
    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    assert sum(len(g["params"]) for g in opt.param_groups) == len(
        list(model.parameters())
    )


# ---------------------------------------------------------------------------
# Sharding is real, and it does not change the arithmetic
# ---------------------------------------------------------------------------

STEPS = 6


def _worker_matches_replicated(rank, world_size, port, out):
    _init(rank, world_size, port)

    sharded = _make_model()
    opt = I._build_amort_optimizer(sharded.parameters(), _opt_args())

    # Same parameters, same gradients, replicated AdamW: the reference every
    # rank must reproduce.
    replicated = _make_model()
    ref_opt = torch.optim.AdamW(
        replicated.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WD,
    )

    for step in range(STEPS):
        for module, optimizer in ((sharded, opt), (replicated, ref_opt)):
            optimizer.zero_grad(set_to_none=True)
            _fill_grads(module, step)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()

    total_numel = sum(p.numel() for p in sharded.parameters())
    out[rank] = (
        (_flat_params(sharded) - _flat_params(replicated)).abs().max().item(),
        _local_shard_numel(opt),
        total_numel,
        # Plain floats, not a tensor: a Manager dict proxy pickles tensors
        # through fd-passing shared memory, which deadlocks under pytest's
        # output capture.  The list still compares exactly.
        _flat_params(sharded).tolist(),
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_step_is_bitwise_identical_to_replicated_adamw(world_size):
    """The load-bearing claim: sharding moves memory, not numbers.

    Bitwise, not approximate.  ZeRO-1 leaves gradients replicated and full, so
    unlike a bucketed all-reduce there is no reduction whose summation order
    could change; each parameter sees the same AdamW arithmetic on the same
    inputs, just computed on a different rank.
    """
    results = _run(_worker_matches_replicated, world_size, port_salt=2)
    assert len(results) == world_size
    for rank, (max_diff, _, _, _) in results.items():
        assert max_diff == 0.0, (
            f"rank {rank} drifted from replicated AdamW after {STEPS} steps: {max_diff}"
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_every_rank_holds_only_its_own_slice_of_the_moments(world_size):
    """Guards against ZeRO silently degrading to a replicated optimizer.

    Without this, the equivalence test above would still pass while the memory
    saving -- the entire reason for the change -- had quietly gone away.
    """
    results = _run(_worker_matches_replicated, world_size, port_salt=2)
    total = results[0][2]
    shards = {rank: shard for rank, (_, shard, _, _) in results.items()}

    # Two fp32 moments per parameter, partitioned exactly once over the group.
    assert sum(shards.values()) == 2 * total, (
        f"moment elements {sum(shards.values())} != 2 * {total}; "
        "the partition either overlaps or drops parameters"
    )
    for rank, shard in shards.items():
        assert 0 < shard < 2 * total, (
            f"rank {rank} holds {shard} of {2 * total} moment elements -- "
            "that is a full replica, not a shard"
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_step_leaves_every_rank_with_the_same_full_parameters(world_size):
    """``step()`` must broadcast before returning.

    The training loop updates the amortizer EMA immediately after ``step()`` and
    reads the live parameters to do it, so a rank still holding pre-step values
    for parameters owned by another rank would corrupt the EMA on that rank.
    """
    results = _run(_worker_matches_replicated, world_size, port_salt=2)
    rank0 = results[0][3]
    for rank, (_, _, _, params) in results.items():
        assert params == rank0, f"rank {rank} parameters diverged from rank 0"


# ---------------------------------------------------------------------------
# Checkpoint round trip
# ---------------------------------------------------------------------------

def _snapshot(state):
    """An independent copy of *state*, the way a checkpoint write/read gives one.

    Optimizer state dicts hand back live references to the moment tensors, and
    ``load_state_dict`` keeps those references when dtype and device already
    match instead of copying.  Feeding one dict to two optimizers therefore
    silently makes them share moments.  ZeRO goes further and *edits* the dict
    it is given, replacing entries this rank does not own with None.

    Neither bites in production: ``save_checkpoint`` serializes the dict and
    every rank reads its own fresh copy back from disk.  Tests that skip the
    disk hop have to reproduce that isolation explicitly.
    """
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _save_load_via_disk(state, tmpdir, rank, tag="ckpt"):
    """Round-trip *state* the way the training loop really does.

    ``save_checkpoint`` writes from rank 0 only and ``ckpt_resume`` then loads
    that one file on every rank, so a faithful test cannot hand each rank its
    own in-memory dict.
    """
    path = os.path.join(tmpdir, f"{tag}.pth")
    if rank == 0:
        torch.save(state, path)
    dist.barrier()
    return torch.load(path, map_location="cpu", weights_only=False)


def _worker_optimizer_state_round_trip(rank, world_size, port, tmpdir, out):
    _init(rank, world_size, port)

    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _fill_grads(model, step)
        opt.step()

    saved = I._amort_optimizer_state_dict(opt)
    has_state = saved is not None
    n_saved_entries = len(saved["state"]) if has_state else 0

    payload = _save_load_via_disk({"amort_optimizer": saved}, tmpdir, rank)

    # A fresh rank, fresh optimizer, restored from the one file on disk.
    resumed_model = _make_model()
    resumed_model.load_state_dict(model.state_dict())
    resumed_opt = I._build_amort_optimizer(resumed_model.parameters(), _opt_args())
    restored = I._load_amort_optimizer_state(resumed_opt, payload)

    # Stepping both from here diverges immediately unless the moments came back:
    # fresh moments give a bias-corrected first step, restored ones do not.
    for step in range(STEPS, STEPS + 3):
        for module, optimizer in ((model, opt), (resumed_model, resumed_opt)):
            optimizer.zero_grad(set_to_none=True)
            _fill_grads(module, step)
            optimizer.step()

    out[rank] = (
        has_state,
        n_saved_entries,
        restored,
        (_flat_params(model) - _flat_params(resumed_model)).abs().max().item(),
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_zero_resume_restores_the_moments_on_every_rank(world_size):
    """The user-facing behaviour: resume must continue, not restart.

    Comparing parameters after further steps rather than comparing state dicts
    is deliberate -- it fails if the moments come back empty, stale, or assigned
    to the wrong rank's shard.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = _run(
            _worker_optimizer_state_round_trip, world_size, tmpdir, port_salt=3,
        )
    assert len(results) == world_size
    for rank, (_, _, restored, max_diff) in results.items():
        assert restored, f"rank {rank} reported no optimizer state restored"
        assert max_diff == 0.0, (
            f"rank {rank} diverged after resume: {max_diff} -- moments did not survive"
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_only_rank_zero_returns_optimizer_state(world_size):
    """Non-target ranks must return None rather than raising.

    ``state_dict()`` on a rank that was not the ``consolidate_state_dict(to=)``
    target raises RuntimeError.  Returning None is safe because
    ``save_checkpoint`` returns early on every rank but the main one.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = _run(
            _worker_optimizer_state_round_trip, world_size, tmpdir, port_salt=3,
        )
    assert results[0][0] is True, "rank 0 must hold the consolidated state"
    assert results[0][1] > 0, "rank 0 consolidated an empty state"
    for rank in range(1, world_size):
        assert results[rank][0] is False, f"rank {rank} should not carry the state"


def test_single_process_round_trip_needs_no_consolidation():
    """The fallback path keeps working, and keeps the same dict layout."""
    assert not dist.is_initialized()
    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _fill_grads(model, step)
        opt.step()

    saved = _snapshot(I._amort_optimizer_state_dict(opt))
    assert saved is not None
    assert set(saved) == {"state", "param_groups"}

    resumed = _make_model()
    resumed.load_state_dict(model.state_dict())
    resumed_opt = I._build_amort_optimizer(resumed.parameters(), _opt_args())
    assert I._load_amort_optimizer_state(resumed_opt, {"amort_optimizer": saved})

    for step in range(STEPS, STEPS + 3):
        for module, optimizer in ((model, opt), (resumed, resumed_opt)):
            optimizer.zero_grad(set_to_none=True)
            _fill_grads(module, step)
            optimizer.step()
    assert torch.equal(_flat_params(model), _flat_params(resumed))


def test_missing_optimizer_state_is_reported_not_fatal():
    """Pre-AMFD-optimizer checkpoints must still resume the module weights."""
    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    assert I._load_amort_optimizer_state(opt, {}) is False
    assert I._load_amort_optimizer_state(opt, {"amort_optimizer": None}) is False


# ---------------------------------------------------------------------------
# Interoperability across launch shapes
# ---------------------------------------------------------------------------

def _reference_adamw_state(steps=STEPS):
    """A replicated AdamW checkpoint, as a single-process run would write it."""
    model = _make_model()
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WD,
    )
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        _fill_grads(model, step)
        opt.step()
    return _snapshot(opt.state_dict())


def _worker_loads_replicated_checkpoint(rank, world_size, port, out):
    _init(rank, world_size, port)

    ref_state = _reference_adamw_state()

    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    assert I._is_zero_optimizer(opt)
    assert I._load_amort_optimizer_state(opt, {"amort_optimizer": _snapshot(ref_state)})

    # Same checkpoint into a replicated optimizer, from its own copy.
    # ``_make_model`` is seeded, so both models start from identical weights.
    ref_model = _make_model()
    ref_opt = torch.optim.AdamW(
        ref_model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WD,
    )
    ref_opt.load_state_dict(_snapshot(ref_state))

    for step in range(STEPS, STEPS + 3):
        for module, optimizer in ((model, opt), (ref_model, ref_opt)):
            optimizer.zero_grad(set_to_none=True)
            _fill_grads(module, step)
            optimizer.step()

    out[rank] = (_flat_params(model) - _flat_params(ref_model)).abs().max().item()
    dist.destroy_process_group()


def _worker_load_edits_the_input_dict(rank, world_size, port, out):
    """ZeRO's load nulls out entries this rank does not own, in place."""
    _init(rank, world_size, port)

    given = _reference_adamw_state()
    n_before = len(given["state"])

    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    I._load_amort_optimizer_state(opt, {"amort_optimizer": given})

    out[rank] = (
        n_before,
        sum(1 for v in given["state"].values() if v is None),
        len(given["state"]),
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_load_mutates_the_dict_it_is_given(world_size):
    """Documents a sharp edge, so a future caller does not reuse the dict.

    Harmless in the training loop -- every rank reads its own copy off disk and
    nothing looks at ``amort_optimizer`` again afterwards -- but it means a
    single state dict cannot be fed to two optimizers.
    """
    results = _run(_worker_load_edits_the_input_dict, world_size, port_salt=7)
    for rank, (n_before, n_nulled, n_after) in results.items():
        assert n_after == n_before, "entry count should be preserved"
        assert n_nulled > 0, (
            f"rank {rank} saw no nulled entries; if ZeRO stopped editing its "
            "input, the _snapshot guards in this file can be relaxed"
        )
        assert n_nulled < n_before, f"rank {rank} kept nothing at all"


@pytest.mark.parametrize("world_size", [2, 4])
def test_replicated_checkpoint_loads_into_sharded_optimizer(world_size):
    """A run started before ZeRO landed must resume onto a ZeRO launch.

    Both layouts index parameters globally in the same order, so the dicts are
    interchangeable -- this pins that down rather than assuming it.
    """
    results = _run(_worker_loads_replicated_checkpoint, world_size, port_salt=4)
    assert len(results) == world_size
    for rank, max_diff in results.items():
        assert max_diff == 0.0, f"rank {rank} diverged from the replicated reference"


def _worker_saves_for_a_smaller_world(rank, world_size, port, tmpdir, out):
    """Save under ZeRO, then reload the same file at a different world size."""
    _init(rank, world_size, port)

    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _fill_grads(model, step)
        opt.step()

    saved = I._amort_optimizer_state_dict(opt)
    if rank == 0:
        torch.save({"amort_optimizer": saved}, os.path.join(tmpdir, "zero.pth"))
        # Rank 0 also records that the consolidated layout matches a replicated
        # run's, which is what makes the two interchangeable.
        out["global_state_matches_replicated_layout"] = (
            set(saved["state"]) == set(_reference_adamw_state()["state"])
        )
    dist.barrier()
    out[rank] = True
    dist.destroy_process_group()


def test_zero_checkpoint_resumes_at_a_different_world_size():
    """World size is not baked into the checkpoint.

    The consolidated dict is global, so a resume recomputes the partition for
    whatever group it finds -- including a single-process rerun, which is how
    these checkpoints get evaluated.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        four = _run(_worker_saves_for_a_smaller_world, 4, tmpdir, port_salt=5)
        assert four["global_state_matches_replicated_layout"], (
            "consolidated state does not use global parameter indexing"
        )

        # Resume that 4-rank checkpoint in this single process, then confirm it
        # continues along the same trajectory the 4-rank run was on.
        payload = torch.load(
            os.path.join(tmpdir, "zero.pth"), map_location="cpu", weights_only=False,
        )

    model = _make_model()
    opt = I._build_amort_optimizer(model.parameters(), _opt_args())
    assert isinstance(opt, torch.optim.AdamW)
    assert I._load_amort_optimizer_state(opt, _snapshot(payload))

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _fill_grads(model, step)
        opt.step()

    # Same restored moments, same grads, replicated AdamW: the single-process
    # resume has to track it exactly.
    reference = _make_model()
    ref_opt = torch.optim.AdamW(
        reference.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WD,
    )
    ref_opt.load_state_dict(_snapshot(payload)["amort_optimizer"])
    for step in range(STEPS):
        ref_opt.zero_grad(set_to_none=True)
        _fill_grads(reference, step)
        ref_opt.step()
    assert torch.equal(_flat_params(model), _flat_params(reference))


# ---------------------------------------------------------------------------
# The real amortizer, through the public build/save/load entry points
# ---------------------------------------------------------------------------

FEAT_DIM, NUM_CLASSES = 16, 4


class _cpu_runtime:
    """Redirect the integration layer's CUDA-only calls onto CPU.

    The in-process suite gets this from a monkeypatch fixture, which does not
    survive into a spawned worker.
    """

    def __enter__(self):
        self._cuda = torch.nn.Module.cuda
        torch.nn.Module.cuda = lambda self, *a, **k: self
        return self

    def __exit__(self, *exc):
        torch.nn.Module.cuda = self._cuda
        return False


def _amfd_args(argv=()):
    import argparse

    parser = argparse.ArgumentParser()
    I.add_amfd_args(parser)
    args = parser.parse_args(["--amfd_static", *argv])
    args.num_classes = NUM_CLASSES
    args.grad_checkpointing = False
    args.current_step = 0
    args.fd_repr_models = ["fake_0", "fake_1"]
    args.amort_model_channels = 32
    args.amort_depth = 2
    args.amort_num_adaln_blocks = 2
    I.resolve_amfd_args(args)
    return args


def _amfd_judges(n=2):
    return [
        {
            "name": f"fake_{i}",
            "feat_dim": FEAT_DIM,
            "pool_type": "cls",
            "mu_ref": torch.zeros(FEAT_DIM, dtype=torch.float64),
            "sigma_ref": torch.eye(FEAT_DIM, dtype=torch.float64) * (i + 1.0),
            "weight": 1.0,
        }
        for i in range(n)
    ]


def _step_amortizers(modules, optimizer, step):
    """One amortizer update, minus the sampling: the loop's step, faithfully.

    Mirrors ``update_amortizers`` in the parts ZeRO can see -- the requires_grad
    toggling around the step, gradients only on trainable parameters, the clip,
    and ``zero_grad`` afterwards.
    """
    I._set_requires_grad(modules, True)
    optimizer.zero_grad(set_to_none=True)
    for i, p in enumerate(modules.parameters()):
        if p.requires_grad:
            torch.manual_seed(7717 * step + i)
            p.grad = torch.randn_like(p)
    torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    I._set_requires_grad(modules, False)


def _worker_amfd_round_trip(rank, world_size, port, tmpdir, argv, out):
    _init(rank, world_size, port)

    with _cpu_runtime():
        args = _amfd_args(argv)
        judges, modules, opt = I.build_amfd_amortizers(_amfd_judges(), args)
        is_zero = I._is_zero_optimizer(opt)
        frozen = sum(1 for p in modules.parameters() if not p.requires_grad)

        for step in range(STEPS):
            _step_amortizers(modules, opt, step)

        state = I.save_amfd_state(judges, opt, args)
        has_optimizer_key = "amort_optimizer" in state
        payload = _save_load_via_disk(state, tmpdir, rank, tag="amfd")

        args2 = _amfd_args(argv)
        judges2, modules2, opt2 = I.build_amfd_amortizers(_amfd_judges(), args2)
        restored = I.load_amfd_state(judges2, opt2, payload, args2)
        weights_match = torch.equal(_flat_params(modules), _flat_params(modules2))

        for step in range(STEPS, STEPS + 3):
            _step_amortizers(modules, opt, step)
            _step_amortizers(modules2, opt2, step)

        out[rank] = (
            is_zero,
            frozen,
            has_optimizer_key,
            restored,
            weights_match,
            (_flat_params(modules) - _flat_params(modules2)).abs().max().item(),
        )
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "argv, expect_frozen",
    [
        pytest.param([], False, id="defaults"),
        pytest.param(
            ["--no_amort_share_real_fake_mlp", "--no_amort_train_real_branch"],
            True,
            id="frozen-real-branch",
        ),
    ],
)
def test_amfd_checkpoint_round_trip_under_zero(argv, expect_frozen):
    """End to end on the real amortizer: build, step, save, resume, keep going.

    The frozen case matters on its own: with the real branch frozen, part of
    every shard is parameters that never receive a gradient, and ZeRO has to
    partition and consolidate around them without erroring or losing state.
    """
    import tempfile

    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        results = _run(
            _worker_amfd_round_trip, world_size, tmpdir, argv, port_salt=6,
        )

    assert len(results) == world_size
    for rank, row in results.items():
        is_zero, frozen, has_key, restored, weights_match, max_diff = row
        assert is_zero, f"rank {rank} did not get a sharded optimizer"
        assert (frozen > 0) == expect_frozen, (
            f"rank {rank}: {frozen} frozen parameters, expected "
            f"{'some' if expect_frozen else 'none'}"
        )
        assert has_key == (rank == 0), (
            f"rank {rank}: amort_optimizer key present={has_key}"
        )
        assert restored, f"rank {rank} did not restore AMFD state"
        assert weights_match, f"rank {rank} restored the wrong module weights"
        assert max_diff == 0.0, (
            f"rank {rank} diverged after resume: {max_diff} -- "
            "the amortizer moments did not survive the round trip"
        )
