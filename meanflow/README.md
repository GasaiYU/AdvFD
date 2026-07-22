# JiT-B sCT / MeanFlow training

`train_jit_b_meanflow.sh` now launches the pixel-space ImageNet sCT setting from
Lu & Song, *Simplifying, Stabilizing & Scaling Continuous-Time Consistency
Models* (arXiv:2410.11081). The legacy MeanFlow loss remains available through
`--objective meanflow` for ablations.

## Stable sCT launcher

```bash
bash meanflow/train_jit_b_meanflow.sh
```

The launcher uses `train_scm_jit_b.py`. The supplied checkpoint has an
`x_pred` raw head, so the first 10k iterations use the supervised TrigFlow
diffusion objective to adapt it into a real `F_theta` head. Adam and EMA are
then reset before the 400k-step sCT phase starts. This avoids silently treating
an `x_pred` tensor as `F_theta` and keeps the resulting checkpoint compatible
with the ordinary TrigFlow sCM sampler.

The remaining defaults are:

- TrigFlow with `sigma_data=0.5` and `sigma_max=80`;
- `tau ~ Normal(-1.0, 1.6^2)` and `t=atan(exp(tau)/sigma_data)`;
- identity time conditioning and positional time embeddings;
- adaptive double normalization;
- full-batch JVP rearrangement (no JVP dropout);
- tangent normalization `g / (||g|| + 0.1)`;
- 10k-iteration tangent warmup;
- learned adaptive timestep weighting;
- global batch 2048 on 8 GPUs (by default micro-batch 128 per process and 2
  gradient-accumulation steps), with AMP JVP, fused Adam, and foreach EMA;
- 10k `F_theta` adaptation steps at `lr=1e-4`, followed by 400k sCT optimizer
  steps;
- Adam with zero weight decay, using `lr=1e-5`, one epoch of LR warmup,
  `beta=(0.9, 0.99)`, `eps=1e-11`, and gradient clipping at 1.0;
- EDM2 inverse-square-root learning-rate decay with `t_ref=35000`;
- BF16, label dropout 0.1, network dropout 0.45, and no horizontal flip;
- EDM2 power-function EMA with `sigma_rel=0.05`.

The output is

```text
work_dirs/meanflow/jit_b_scm_xpred_bridge/checkpoints/latest.pth
```

## Evaluation

Use the dedicated sCM evaluator; the repository's ordinary `JiT_B` evaluation
preset uses the incompatible linear-flow sampler.

Quick visualization (default):

```bash
bash meanflow/eval_jit_b_scm.sh
```

Select a checkpoint or run two-step sampling:

```bash
CKPT=work_dirs/meanflow/jit_b_scm_xpred_bridge/checkpoints/step_0019999.pth \
NUM_STEPS=2 bash meanflow/eval_jit_b_scm.sh
```

Run a 50k-image Inception FID evaluation:

```bash
VIS_ONLY=0 NUM_IMAGES=50000 EVAL_MODELS=inception \
  bash meanflow/eval_jit_b_scm.sh
```

The evaluator reconstructs `MeanFlowJiTDenoiser(objective="scm")`, samples
TrigFlow noise with `sigma_data=0.5`, and evaluates the `power_0.05` EMA at
CFG 1.0 by default.

`model_ema` is the preferred state dict for sCM sampling. It must be loaded into
`MeanFlowJiTDenoiser(objective="scm", sigma_data=0.5, sigma_max=80)` so that the
TrigFlow consistency wrapper is applied. An ordinary linear-flow JiT sampler is
not semantically compatible with this checkpoint.

The supplied baseline JiT checkpoint was trained with the repository's original
linear endpoint parameterization, whereas the paper initializes from a fully
trained TrigFlow diffusion teacher with matching architecture. The adaptation
phase repairs the input/time/output semantics, but it is still only an
approximation to having that teacher. Set `F_ADAPT_STEPS` higher if the
`loss_f_adapt` and `f_adapt_x0_mse` metrics have not converged at the phase
boundary. If `--load_from` is already a genuine TrigFlow-F checkpoint, use
`--init_parameterization F --f_adapt_steps 0`.

## Legacy MeanFlow objective

Use the Python entry point with `--objective meanflow` to recover the previous
linear-path loss:

```text
loss = mse(u, v - dudt_weight * t * dU/dt)
```

In this mode, `--dudt_drop_prob`, `--dudt_weight`, `--dudt_clip_norm`, and
`--loss_clip` retain their old meanings. They are deliberately ignored by the
new `scm` objective.
