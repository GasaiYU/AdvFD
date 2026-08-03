"""Quantify the numerical cost of bf16 gradient compression.

Not a pass/fail gate on training quality -- that needs a real run. This measures
the perturbation bf16 compression introduces, so the decision is made against
numbers instead of intuition. Compared against the bucketing change already in
utils.distributed_util.all_reduce_grads, which is ~1 fp32 ULP.
"""
import torch


def _bf16_roundtrip(x):
    return x.to(torch.bfloat16).to(torch.float32)


def _rel_err(approx, exact):
    scale = exact.abs().max().clamp_min(1e-30)
    return ((approx - exact).abs().max() / scale).item()


def test_report_bf16_vs_fp32_gradient_error(capsys):
    torch.manual_seed(0)
    world_size = 16
    # Match the observed run: grad_norm ~0.0062 over ~1e9 params, so per-element
    # magnitudes are small. bf16 keeps fp32's exponent range, so this is about
    # mantissa precision, not underflow.
    for label, scale in [("grad_norm~0.0062 (observed)", 6.2e-3 / (955e6 ** 0.5)),
                         ("unit-scale", 1.0)]:
        per_rank = [torch.randn(200_000) * scale for _ in range(world_size)]

        exact = torch.stack(per_rank).mean(dim=0)
        compressed = torch.stack([_bf16_roundtrip(g) for g in per_rank]).mean(dim=0)
        single = _bf16_roundtrip(per_rank[0])

        with capsys.disabled():
            print(f"\n{label}  (element scale {scale:.3e})")
            print(f"  bf16 round-trip, one tensor      : {_rel_err(single, per_rank[0]):.3e} relative")
            print(f"  after averaging {world_size} ranks        : {_rel_err(compressed, exact):.3e} relative")
            print(f"  fp32 eps for reference           : {torch.finfo(torch.float32).eps:.3e}")
            print(f"  bf16 eps for reference           : {torch.finfo(torch.bfloat16).eps:.3e}")

    # The property that actually matters: bf16 error is orders of magnitude
    # above the bucketing change, so it cannot be waved through the same way.
    g = torch.randn(200_000)
    bf16_err = _rel_err(_bf16_roundtrip(g), g)
    assert bf16_err > 1e-3, "bf16 mantissa is 7 bits; expect ~1e-3 relative"
    assert bf16_err > 1e4 * torch.finfo(torch.float32).eps


def test_bf16_has_no_underflow_at_observed_gradient_scale():
    """bf16 shares fp32's exponent range, so small gradients do not flush to zero.

    This is the one failure mode that would be fatal rather than merely lossy,
    and it is the reason bf16 is used for this instead of fp16.
    """
    tiny = torch.full((1000,), 6.2e-3 / (955e6 ** 0.5))
    assert (_bf16_roundtrip(tiny) != 0).all()
    # fp16 would be fine here too, but dies far earlier:
    assert (torch.full((1000,), 1e-8).to(torch.bfloat16) != 0).all()
    assert (torch.full((1000,), 1e-8).to(torch.float16) == 0).all()
