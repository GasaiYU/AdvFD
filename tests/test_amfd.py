"""CPU tests for the AMFD static-branch integration.

Covers the vendored loss module (shapes, gradient flow, JVP implementations)
and the repository wiring in ``amfd.integration`` (amortizer construction, the
alternating update, the generator loss, and checkpoint round trips).

Runs on CPU with fake judges and a fake generator, so no GPU, dataset, or
pretrained encoder is needed.
"""

import argparse

import pytest
import torch

import amfd.integration as I
from amfd.amfd_loss import AmortizedFDLoss
from amfd.integration import add_amfd_args

FEAT_DIM = 32
BATCH = 6
NUM_CLASSES = 10
IMG = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_amortizer(**kw):
    cfg = dict(
        feat_dim=FEAT_DIM,
        model_channels=32,
        depth=4,
        num_classes=NUM_CLASSES,
        num_adaln_blocks=2,
        feature_mean=torch.zeros(FEAT_DIM),
        feature_std=torch.ones(FEAT_DIM),
        t=0.25,
        diff_batch_mul=4,
    )
    cfg.update(kw)
    return AmortizedFDLoss(**cfg)


def warm_amortizer(amort, steps=12, num_classes=NUM_CLASSES):
    """Run a few amortizer updates so its outputs leave the zero init.

    ``MlpEncoder.initialize_weights`` zeroes the final layer, so a fresh
    amortizer returns 0 for both the mean and the operator action.  That makes
    ``dmu`` and ``delta_au`` vanish and the generator loss identically zero --
    correct behaviour, but useless for testing gradient flow.  The real loop
    always runs ``update_amortizers`` before the generator step, so warming up
    here mirrors production rather than working around a bug.
    """
    for p in amort.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(amort.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = amort.amort_loss(
            h_real=torch.randn(BATCH, FEAT_DIM) + 2.0,
            h_fake=torch.randn(BATCH, FEAT_DIM) * 0.5,
            labels=torch.randint(0, num_classes, (BATCH,)),
        )
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)


class FakeReprModel(torch.nn.Module):
    """Returns ``(cls, avg)`` like ``TimmReprModel``."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * IMG * IMG, FEAT_DIM)

    def forward(self, x):
        h = self.lin(x.flatten(1))
        return h, h * 0.5


class FakeGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * IMG * IMG, 3 * IMG * IMG)

    def sample_images_with_grad(self, z, y, sampling_args=None):
        return torch.tanh(self.lin(z.flatten(1)).view_as(z))


def make_args(**kw):
    p = argparse.ArgumentParser()
    add_amfd_args(p)
    a = p.parse_args(["--amfd_static"])
    a.num_classes = NUM_CLASSES
    a.grad_checkpointing = False
    a.input_channels, a.input_size = 3, IMG
    a.noise_scale = 1.0
    a.vae_decode_bsz = 8
    a.current_step = 0
    a.fd_repr_models = ["fake_a", "fake_b"]
    # Keep the amortizer small so the tests stay fast.
    a.amort_model_channels = 32
    a.amort_depth = 4
    a.amort_num_adaln_blocks = 2
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def make_judges(n=2):
    judges = []
    for i in range(n):
        judges.append({
            "name": f"fake_{i}",
            "model": FakeReprModel(),
            "feat_dim": FEAT_DIM,
            "pool_type": "cls",
            "inception_layer": None,
            "mu_ref": torch.zeros(FEAT_DIM, dtype=torch.float64),
            # judge i has sigma = (i+1) * I, so per-judge stds differ.
            "sigma_ref": torch.eye(FEAT_DIM, dtype=torch.float64) * (i + 1.0),
            "weight": 1.0,
        })
    return judges


def real_batch_fn():
    return (
        torch.rand(BATCH, 3, IMG, IMG),
        torch.randint(0, NUM_CLASSES, (BATCH,)),
    )


SAMPLING_ARGS = {
    "t_min": 0.1, "t_max": 1.0, "cfg": 1.0, "num_steps": 1,
    "t_start_min": 0.1, "return_velocity": False,
}


@pytest.fixture
def cpu_runtime(monkeypatch):
    """Redirect the integration layer's CUDA-only calls onto CPU.

    ``torch.func.jvp`` lazily ``torch.jit.script``s its forward-AD
    decompositions on first use.  Doing that after ``torch.zeros`` is patched
    makes the scripter trip over the patched signature, so the load is forced
    before the patches go in.  ``monkeypatch`` reverts everything on teardown
    so no other test in the suite sees these.
    """
    torch.func.jvp(lambda x: x * 2, (torch.ones(1),), (torch.ones(1),))

    orig_zeros, orig_randn = torch.zeros, torch.randn

    def cpu_zeros(*a, **k):
        k.pop("device", None)
        return orig_zeros(*a, **k)

    def cpu_randn(*a, **k):
        k.pop("device", None)
        return orig_randn(*a, **k)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "randn", cpu_randn)
    yield


# ---------------------------------------------------------------------------
# Vendored loss module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("jacobi", [True, False])
@pytest.mark.parametrize("jvp_impl", ["torch_func", "manual"])
def test_amort_loss_backward_reaches_params(share, jacobi, jvp_impl):
    amort = build_amortizer(
        share_real_fake_mlp=share,
        jacobi_generator_loss=jacobi,
        jvp_impl=jvp_impl,
    )
    loss, logs = amort.amort_loss(
        h_real=torch.randn(BATCH, FEAT_DIM),
        h_fake=torch.randn(BATCH, FEAT_DIM),
        labels=torch.randint(0, NUM_CLASSES, (BATCH,)),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "amfd/amort_loss" in logs

    loss.backward()
    grads = [p.grad for p in amort.parameters() if p.grad is not None]
    assert grads, "amortizer loss produced no parameter gradients"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("jacobi", [True, False])
@pytest.mark.parametrize("jvp_impl", ["torch_func", "manual"])
def test_generator_loss_reaches_features_and_leaves_amortizer_frozen(
    share, jacobi, jvp_impl,
):
    amort = build_amortizer(
        share_real_fake_mlp=share,
        jacobi_generator_loss=jacobi,
        jvp_impl=jvp_impl,
    )
    warm_amortizer(amort)
    amort.zero_grad(set_to_none=True)
    for p in amort.parameters():
        p.requires_grad_(False)

    feats = torch.randn(BATCH, FEAT_DIM, requires_grad=True)
    gen_loss, logs = amort.generator_loss(
        h_fake=feats, labels=torch.randint(0, NUM_CLASSES, (BATCH,)),
    )
    assert gen_loss.ndim == 0
    assert torch.isfinite(gen_loss)
    assert "amfd/gen_loss" in logs

    gen_loss.backward()
    assert feats.grad is not None
    assert float(feats.grad.abs().sum()) > 0
    assert torch.isfinite(feats.grad).all()
    # The generator step must not accumulate into the amortizer.
    assert all(p.grad is None for p in amort.parameters())


def test_manual_and_torch_func_jvp_agree():
    a1 = build_amortizer(jvp_impl="torch_func", share_real_fake_mlp=False)
    a2 = build_amortizer(jvp_impl="manual", share_real_fake_mlp=False)
    a2.load_state_dict(a1.state_dict())

    v = torch.randn(BATCH, FEAT_DIM)
    t = torch.full((BATCH, 1), 0.25)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    with torch.no_grad():
        o1 = a1.A_apply("real", v, t, labels)
        o2 = a2.A_apply("real", v, t, labels)
    assert float((o1 - o2).abs().max()) < 1e-4


def test_per_encoder_normalized_generator_loss():
    amort = build_amortizer(
        normalize_generator_loss=True,
        generator_loss_norm_eps=0.01,
        generator_loss_norm_power=1.0,
    )
    warm_amortizer(amort)
    for p in amort.parameters():
        p.requires_grad_(False)

    feats = torch.randn(BATCH, FEAT_DIM, requires_grad=True)
    gen_loss, logs = amort.generator_loss(
        h_fake=feats, labels=torch.randint(0, NUM_CLASSES, (BATCH,)),
    )
    gen_loss.backward()
    assert torch.isfinite(gen_loss)
    assert float(feats.grad.abs().sum()) > 0
    assert "amfd/gen_loss_norm_denom" in logs


def test_x_prediction_target():
    amort = build_amortizer(prediction_target="x")
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss, _ = amort.amort_loss(
        h_real=torch.randn(BATCH, FEAT_DIM),
        h_fake=torch.randn(BATCH, FEAT_DIM),
        labels=labels,
    )
    assert torch.isfinite(loss)

    warm_amortizer(amort)
    for p in amort.parameters():
        p.requires_grad_(False)
    feats = torch.randn(BATCH, FEAT_DIM, requires_grad=True)
    gen_loss, _ = amort.generator_loss(h_fake=feats, labels=labels)
    gen_loss.backward()
    assert float(feats.grad.abs().sum()) > 0


def test_unconditional_single_label():
    amort = build_amortizer(num_classes=1)
    labels = torch.zeros(BATCH, dtype=torch.long)
    loss, _ = amort.amort_loss(
        h_real=torch.randn(BATCH, FEAT_DIM),
        h_fake=torch.randn(BATCH, FEAT_DIM),
        labels=labels,
    )
    assert torch.isfinite(loss)

    warm_amortizer(amort, num_classes=1)
    for p in amort.parameters():
        p.requires_grad_(False)
    feats = torch.randn(BATCH, FEAT_DIM, requires_grad=True)
    gen_loss, _ = amort.generator_loss(h_fake=feats, labels=labels)
    gen_loss.backward()
    assert float(feats.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# Upstream parity
# ---------------------------------------------------------------------------

def test_argparse_defaults_match_upstream():
    """Every default is copied from upstream ``main_amfd.py``.

    Upstream's own ImageNet launcher overrides some of these (c2048/d16/a4,
    manual JVP, per-encoder normalization); the shipped scripts do the same.
    These are the bare argparse values.
    """
    p = argparse.ArgumentParser()
    add_amfd_args(p)
    a = p.parse_args([])

    assert a.amort_lr == 1e-4
    assert a.amort_beta1 == 0.9
    assert a.amort_beta2 == 0.95
    assert a.amort_weight_decay == 0.0
    assert a.amort_grad_clip == 1.0
    assert a.amort_model_channels == 1024
    assert a.amort_depth == 8
    assert a.amort_num_adaln_blocks == 2
    assert a.amort_uncond is False
    assert a.amort_updates_per_gen_update == 1
    assert a.amort_ema_decay == 0.0
    assert a.amort_t == 0.25
    assert a.amort_prediction_target == "v"
    assert a.amort_diff_batch_mul == 4
    assert a.amort_jvp_impl == "torch_func"
    assert a.amort_jacobi_gen_loss is True
    assert a.amort_share_real_fake_mlp is True
    assert a.amort_train_real_branch is True
    assert a.amort_normalize_gen_loss_per_encoder is False
    assert a.amort_gen_loss_norm_eps == 0.01
    assert a.amort_gen_loss_norm_power == 1.0


def test_negated_flags_flip_upstream_defaults():
    p = argparse.ArgumentParser()
    add_amfd_args(p)
    a = p.parse_args([
        "--no_amort_jacobi_gen_loss",
        "--no_amort_share_real_fake_mlp",
        "--no_amort_train_real_branch",
    ])
    assert a.amort_jacobi_gen_loss is False
    assert a.amort_share_real_fake_mlp is False
    assert a.amort_train_real_branch is False


# ---------------------------------------------------------------------------
# Integration layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uncond", [True, False])
def test_build_amortizers_per_judge(cpu_runtime, uncond):
    args = make_args(amort_uncond=uncond)
    judges = make_judges()
    I.resolve_amfd_args(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(judges, args)

    assert len(amort_judges) == 2
    assert isinstance(amort_modules, torch.nn.ModuleList)
    assert opt is not None

    amort = amort_judges[0]["amort"]
    net = amort.shared_net if args.amort_share_real_fake_mlp else amort.real_net
    assert net.cond_embed.num_embeddings == (1 if uncond else NUM_CLASSES)

    # Normalizers are derived per judge from that judge's own FD reference
    # stats: judge 0 has sigma = I, judge 1 has sigma = 2I.
    assert torch.allclose(amort.feature_std, torch.ones(FEAT_DIM))
    assert torch.allclose(
        amort_judges[1]["amort"].feature_std,
        torch.full((FEAT_DIM,), 2.0 ** 0.5),
        atol=1e-6,
    )


@pytest.mark.parametrize("uncond", [True, False])
def test_update_amortizers_steps_and_freezes(cpu_runtime, uncond):
    args = make_args(amort_uncond=uncond)
    I.resolve_amfd_args(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(make_judges(), args)
    model = FakeGenerator()

    before = [p.detach().clone() for p in amort_modules.parameters()]
    labels = None
    logs = {}
    for step in range(3):
        args.current_step = step
        amort_loss, grad_norm, logs, labels = I.update_amortizers(
            model, amort_judges, amort_modules, opt,
            real_batch_fn, dict(SAMPLING_ARGS), args,
        )
        assert torch.isfinite(amort_loss)
        assert torch.isfinite(grad_norm)

    assert any(
        not torch.equal(b, p)
        for b, p in zip(before, amort_modules.parameters())
    ), "amortizer parameters did not move"
    assert labels.shape == (BATCH,)
    assert "amfd/amort_total" in logs and "amfd/grad_norm" in logs
    assert any(k.startswith("fake_0/") for k in logs)
    # update_amortizers must hand the generator step a frozen amortizer.
    assert all(not p.requires_grad for p in amort_modules.parameters())


@pytest.mark.parametrize("uncond", [True, False])
def test_generator_loss_reaches_generator(cpu_runtime, uncond):
    # A high amortizer lr and several updates are needed before the generator
    # loss is nonzero at all -- see test_amortizer_needs_warmup_before_it_
    # produces_generator_signal for why. At the default 1e-4 the real and fake
    # branches are still numerically identical after one step.
    args = make_args(amort_uncond=uncond, amort_lr=1e-2)
    I.resolve_amfd_args(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(make_judges(), args)
    model = FakeGenerator()

    labels = None
    for step in range(15):
        args.current_step = step
        _, _, _, labels = I.update_amortizers(
            model, amort_judges, amort_modules, opt,
            real_batch_fn, dict(SAMPLING_ARGS), args,
        )

    sampled = model.sample_images_with_grad(torch.randn(BATCH, 3, IMG, IMG), labels)
    sampled01 = sampled * 0.5 + 0.5

    total = torch.zeros(())
    for judge in amort_judges:
        feats, _ = judge["model"](sampled01)
        gen_loss, logs = I.amfd_generator_loss(judge, feats, labels, args)
        assert "amfd/gen_loss" in logs
        total = total + gen_loss
    total.backward()

    gparams = [p for p in model.parameters() if p.grad is not None]
    assert gparams, "AMFD generator loss did not reach the generator"
    assert sum(float(p.grad.abs().sum()) for p in gparams) > 0
    assert all(torch.isfinite(p.grad).all() for p in gparams)
    assert all(p.grad is None for p in amort_modules.parameters())


def test_amortizer_needs_warmup_before_it_produces_generator_signal(cpu_runtime):
    """A cold amortizer yields exactly zero generator loss. This is expected.

    ``MlpEncoder.initialize_weights`` zeroes the final layer, so a fresh
    amortizer returns 0 for both the mean and the operator action.  With the
    shared real/fake network the two branches are distinguished only by the
    sign of ``t``, so after a single step at the default ``--amort_lr 1e-4``
    the final layer is still ~1e-4 and the two branches differ by ~1e-8
    relative -- below float32 epsilon.  ``mu_g - mu_r`` and ``A_g - A_r`` round
    to exactly zero, and so does the generator loss.

    The consequence for training: the static branch contributes no gradient for
    the first stretch of steps while the amortizers converge.  A zero
    ``amfd/gen_loss`` early in a run is not a wiring failure.
    """
    args = make_args(amort_uncond=True)
    assert args.amort_lr == 1e-4, "this test is about the upstream default lr"
    I.resolve_amfd_args(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(make_judges(), args)
    model = FakeGenerator()

    amort_loss, grad_norm, _, labels = I.update_amortizers(
        model, amort_judges, amort_modules, opt,
        real_batch_fn, dict(SAMPLING_ARGS), args,
    )
    # The amortizer itself is learning: real loss and real gradients.
    assert float(amort_loss) > 0
    assert float(grad_norm) > 0

    sampled = model.sample_images_with_grad(torch.randn(BATCH, 3, IMG, IMG), labels)
    gen_loss, logs = I.amfd_generator_loss(
        amort_judges[0], amort_judges[0]["model"](sampled * 0.5 + 0.5)[0], labels, args,
    )
    assert float(gen_loss.detach()) == 0.0
    assert float(logs["amfd/delta_mu_norm"]) == 0.0
    assert float(logs["amfd/delta_au_norm"]) == 0.0

    # Once the amortizer has actually moved, the signal appears.
    for step in range(15):
        args.current_step = step
        I.update_amortizers(
            model, amort_judges, amort_modules, opt,
            real_batch_fn, dict(SAMPLING_ARGS), args,
        )
    for g in opt.param_groups:
        g["lr"] = 1e-2
    for step in range(15):
        args.current_step = step
        _, _, _, labels = I.update_amortizers(
            model, amort_judges, amort_modules, opt,
            real_batch_fn, dict(SAMPLING_ARGS), args,
        )
    sampled = model.sample_images_with_grad(torch.randn(BATCH, 3, IMG, IMG), labels)
    _, logs = I.amfd_generator_loss(
        amort_judges[0], amort_judges[0]["model"](sampled * 0.5 + 0.5)[0], labels, args,
    )
    assert float(logs["amfd/delta_au_norm"]) > 0


@pytest.mark.parametrize("uncond", [True, False])
def test_checkpoint_round_trip(cpu_runtime, uncond):
    args = make_args(amort_uncond=uncond)
    I.resolve_amfd_args(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(make_judges(), args)
    model = FakeGenerator()
    I.update_amortizers(
        model, amort_judges, amort_modules, opt,
        real_batch_fn, dict(SAMPLING_ARGS), args,
    )

    state = I.save_amfd_state(amort_judges, opt, args)
    assert {"amort_states", "amort_optimizer", "amort_metadata"} <= set(state)
    assert state["amort_metadata"]["uncond"] == uncond

    args2 = make_args(amort_uncond=uncond)
    I.resolve_amfd_args(args2)
    aj2, _, opt2 = I.build_amfd_amortizers(make_judges(), args2)
    assert I.load_amfd_state(aj2, opt2, state, args2)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            amort_judges[0]["amort"].parameters(),
            aj2[0]["amort"].parameters(),
        )
    )


def test_resume_refuses_uncond_flip(cpu_runtime):
    args = make_args(amort_uncond=True)
    I.resolve_amfd_args(args)
    amort_judges, _, opt = I.build_amfd_amortizers(make_judges(), args)
    state = I.save_amfd_state(amort_judges, opt, args)

    flipped = make_args(amort_uncond=False)
    I.resolve_amfd_args(flipped)
    aj, _, opt2 = I.build_amfd_amortizers(make_judges(), flipped)
    # The amortizer's label space changes, so the weights are meaningless.
    with pytest.raises(ValueError, match="amort_uncond"):
        I.load_amfd_state(aj, opt2, state, flipped)


def test_sampling_args_isolation():
    """AMFD samples from pure noise and must not inherit the loop's mutations."""
    dirty = {
        "t_min": 0.1, "cfg": 1.0, "num_steps": 1,
        "return_velocity": True, "t_start": torch.rand(BATCH),
    }
    clean = I._amfd_sampling_args(dirty)
    assert "return_velocity" not in clean
    assert "t_start" not in clean
    assert clean["t_min"] == 0.1 and clean["cfg"] == 1.0
    # The caller's dict is shared with the training loop; leave it alone.
    assert "return_velocity" in dirty and "t_start" in dirty


def test_disabled_path_allocates_nothing():
    p = argparse.ArgumentParser()
    add_amfd_args(p)
    args = p.parse_args([])
    args.fd_repr_models = ["fake_a"]
    args.num_classes = NUM_CLASSES

    assert not I.amfd_enabled(args)
    amort_judges, amort_modules, opt = I.build_amfd_amortizers(make_judges(1), args)
    assert amort_judges == []
    assert amort_modules is None
    assert opt is None
    assert I.save_amfd_state(amort_judges, opt, args) == {}
    assert I.update_amortizers(
        None, amort_judges, amort_modules, opt, None, {}, args,
    ) == (None, None, {}, None)


def test_resolve_rejects_mismatched_norm_stats_paths():
    args = make_args(amort_norm_stats_paths=["only_one.npz"])
    # Two repr models, one stats path.
    with pytest.raises(ValueError, match="amort_norm_stats_paths"):
        I.resolve_amfd_args(args)


def test_resolve_requires_both_scalar_norm_values():
    args = make_args(amort_norm_mu=[0.0])
    with pytest.raises(ValueError, match="amort_norm_sigma"):
        I.resolve_amfd_args(args)


def test_resolve_broadcasts_single_scalar_norm_value():
    args = make_args(amort_norm_mu=[0.0], amort_norm_sigma=[1.0])
    I.resolve_amfd_args(args)
    assert args.amort_norm_mu == [0.0, 0.0]
    assert args.amort_norm_sigma == [1.0, 1.0]
