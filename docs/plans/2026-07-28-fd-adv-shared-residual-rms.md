# FD-Adv Shared Residual RMS Constraint

## Goal

Bound the distribution-level drift of a trainable FD-Adv representation from
its frozen pretrained reference without changing the existing FD, EMA, or
whitening objectives.

## Transform

For globally gathered real and fake features, define

\[
\Delta_r=\psi_\omega(x_r)-\psi_0(x_r),\qquad
\Delta_f=\psi_\omega(x_f)-\psi_0(x_f),
\]

and the 50:50 mixture second moment

\[
R^2=\tfrac12\left(\mathbb E\|\Delta_r\|_2^2+
\mathbb E\|\Delta_f\|_2^2\right).
\]

The fixed budget is derived from the reference statistics:

\[
S_0=\operatorname{tr}(\Sigma_0)+\|\mu_0\|_2^2,
\qquad \tau=\kappa\sqrt{S_0}.
\]

Real and fake share the differentiable scalar

\[
a=(1+R^2/\tau^2+\epsilon)^{-1/2},\qquad
\widehat\psi=\psi_0+a(\psi_\omega-\psi_0).
\]

Consequently, the projected mixture residual satisfies

\[
\widehat R^2=a^2R^2<\tau^2.
\]

The scale is not detached. Features are gathered before computing it, so all
distributed ranks use the same global scalar.

## Data flow

Both critic and generator paths use:

```text
raw trainable features + frozen reference features
  -> shared residual RMS projection
  -> existing EMA statistics
  -> existing whitening
  -> existing FD objective
```

The main frozen FD representation is reused as the reference when model and
pool type match. In the generator step its graph remains connected to the
generated image. Reference parameters remain frozen.

Hard per-sample norm capping and residual RMS projection are mutually
exclusive. Checkpoints record the feature transform, kappa, and tau so EMA
statistics cannot be restored into an incompatible feature space.

## Default ablation

Use `kappa=0.2` for JiT-B, with a small follow-up grid of `0.1, 0.2, 0.4` if
needed. Log the raw RMS-to-tau ratio, projected RMS-to-tau ratio, and shared
scale every 100 steps.
