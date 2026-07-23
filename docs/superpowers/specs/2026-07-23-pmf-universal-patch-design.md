# Universal pMF Dose–Response Patch Experiment

Date: 2026-07-23

## Goal

Demonstrate that optimizing only Inception Fréchet distance can discover a
single universal image-space pattern that improves Inception FID while harming
post-hoc CLIP Fréchet distance.

The experiment uses the pMF presets and one-step sampling configuration from
`scripts/table_3_pMF.sh`. The pMF generator, its selected EMA weights, the
Inception encoder, and all real-reference statistics remain frozen. The only
trainable tensor is a shared `3 x 16 x 16` patch.

Success is measured on generated samples that never participate in patch
training:

- the fitted Inception-FID slope over patch strength is negative;
- the fitted FD-r-CLIP slope is positive;
- bootstrap confidence intervals and per-seed results are reported.

## Scope

The implementation will add a self-contained `hacking/` experiment. It will
reuse existing pMF construction, checkpoint conversion, sampling,
representation models, and reference statistics without changing the normal
generator-training entry points.

The initial launcher supports every preset in `scripts/table_3_pMF.sh` through
`MODEL_SIZE=B|L|H` and `RES=256|512`. Its default is `pMF_B` at 256 px.
The primary motivation experiment is the 256 px configuration. A 512 px run
must provide explicit Inception and CLIP reference-statistics paths rather
than silently reusing the 256 px defaults.

The experiment does not train, fine-tune, update, or maintain an EMA of the
generator. It does not use CLIP in the patch-training process.

## Components

### `hacking/universal_patch.py`

This module owns the representation of the universal patch and pure numerical
helpers.

- `UniversalPatch` contains the sole parameter `u` with shape
  `3 x patch_size x patch_size`.
- The bounded base tile is `tanh(u)`.
- The base tile is periodically repeated and cropped to the requested image
  resolution.
- Images follow the mathematical `[-1, 1]` convention:

  `x_tilde = clamp(x + alpha * tile(tanh(u)), -1, 1)`.

  Existing pMF utilities return `[0, 1]`, so the implementation converts to
  `[-1, 1]` before perturbation and maps back to `[0, 1]` before a
  representation model sees the image.
- Alpha values are therefore reported as `alpha_pm1`. The result files also
  record that the corresponding maximum displacement in `[0, 1]` is
  `alpha_pm1 / 2`, preventing a hidden factor-of-two ambiguity.
- The regularizer uses periodic total variation, so the boundary of the
  `16 x 16` tile is regularized consistently with tiling, plus the squared
  channel-wise mean of `tanh(u)`.
- After each optimizer update, `u` is projected to zero spatial mean per
  channel. This prevents a simple global color or brightness shift while the
  post-`tanh` mean penalty handles the remaining nonlinearity.

The module also provides:

- deterministic train/test noise-seed derivation;
- linear-slope fitting;
- conflict-score computation;
- replicate-level bootstrap confidence intervals.

### `hacking/fid_loss.py`

Training uses the same empirical Gaussian Fréchet objective as standard FID,
but computes its covariance cross term in sample space.

For a global batch of centered Inception features `Xc` with shape `B x D`,
the non-zero eigenvalues needed by the FID covariance term can be obtained
from a matrix of size at most `(B - 1) x (B - 1)`, rather than repeatedly
diagonalizing a `2048 x 2048` matrix for every alpha. A fixed orthonormal
contrast basis removes the structurally zero centered direction.

This helper:

- computes moments and eigendecompositions in float64;
- accepts fixed real mean and covariance statistics;
- returns a scalar differentiable with respect to generated features;
- is checked against the repository's full covariance-space differentiable
  FID on small synthetic cases;
- is used only for patch training.

Final reported FID values continue to use the repository's standard
float64/SciPy implementation in `frechet_distance.metrics.compute_fid`.

### `hacking/train_universal_patch.py`

The training entry point inherits the generator arguments from `main_fd.py`,
then adds patch-specific arguments.

Startup proceeds as follows:

1. initialize distributed execution and logging;
2. construct pMF with the selected Table-3 preset;
3. load the checkpoint and its EMA state through the existing checkpoint
   loader;
4. require a real EMA state in the checkpoint, select the configured EMA
   label (default `0.9999` for pMF's constant-EMA configuration), and
   materialize that EMA once into the generator;
5. discard the EMA container, set the generator to evaluation mode, and set
   every generator parameter to `requires_grad=False`;
6. load frozen Inception and the fixed Inception reference statistics;
7. construct `UniversalPatch` and an AdamW optimizer containing exactly its
   `u` parameter.

The program validates the optimizer parameter identities at startup. It also
checks after the first backward pass that `u.grad` exists and that every
generator and Inception parameter has `grad is None`.

Training uses exactly 10,000 deterministic generated samples by default.
Labels are class-balanced by global sample index. pMF receives explicit
`z_t` tensors derived from fixed split, rank, and batch seeds, so the same
world-size and batch configuration regenerates the same training set on every
epoch and after resume. The experiment metadata records those distributed
parameters. Train and test seed namespaces are disjoint.

For each batch:

1. generate the clean images with the frozen one-step EMA pMF under
   `torch.inference_mode()`;
2. detach the generated images;
3. apply the same patch at training strengths
   `0, 2/255, 4/255, 6/255, 8/255`;
4. run only the perturbed tensors through frozen Inception with image
   gradients enabled;
5. gather equal-size feature chunks across ranks with gradient support;
6. compute batch empirical FID for every strength;
7. optimize the mean non-zero-strength FID, adjacent monotonic hinge loss,
   periodic TV, and mean penalty;
8. sum the partial patch gradients across ranks, yielding the derivative of
   the global gathered-feature objective, and update only `u`.

The default objective is:

`L = mean_k>0 FID(alpha_k) + lambda_mono * sum_k
relu(FID(alpha_{k+1}) - FID(alpha_k) + margin)
+ lambda_tv * TV + lambda_mean * mean_penalty`.

The launcher defaults are `lambda_mono=1.0`, `margin=0.01`,
`lambda_tv=1e-3`, and `lambda_mean=1.0`. They are recorded in every checkpoint
and remain command-line configurable. The patch optimizer uses AdamW with
zero weight decay and a patch-specific learning rate; it does not inherit the
generator learning rate from `scripts/table_3_pMF.sh`.

The `alpha=0` feature path is detached because its FID is independent of the
patch, but it remains in the first monotonic comparison. All alpha branches
share the same clean generated batch.

The last partial global batch is allowed when every rank has the same local
batch length. The launcher validates divisibility by world size; unsupported
distributed layouts fail before training rather than silently changing the
10,000-image sample set.

The patch checkpoint contains:

- raw `u`;
- optimizer state;
- step and epoch;
- all patch hyperparameters and alpha semantics;
- generator checkpoint path, pMF preset, selected EMA label, and sampling
  parameters;
- train/test seed namespaces and distributed shape;
- random-number-generator state.

It does not duplicate generator or representation-model weights.

### `hacking/eval_universal_patch.py`

Evaluation loads a trained patch and reconstructs the same frozen EMA pMF. It
uses 50,000 samples from the disjoint test seed namespace by default.

For each clean test batch, the evaluator applies every dense alpha value in
memory before feature extraction. Perturbed images are never saved and
reloaded as PNGs, avoiding quantization of small doses.

The default dense grid is
`0, 1/255, 2/255, ..., 8/255` in `[-1, 1]` (`alpha_pm1`) units. A different
maximum or spacing may be supplied explicitly and is written to the result
metadata.

The evaluator loads:

- Inception with the same reference statistics used for patch training;
- `vit_large_patch14_clip_224.openai` with the matching 256-reference
  statistics;
- CLIP CLS features only.

CLIP is imported and instantiated only in this evaluation program. FD-r-CLIP
is the raw CLIP Fréchet distance divided by the repository's validation
normalizer, 5.60.

Evaluation outputs both pooled 50k curves and replicate curves. By default,
the 50k samples are divided into ten deterministic 5k seed replicates.
Sufficient statistics are retained per replicate and alpha, allowing exact
pooling without storing all per-image feature vectors.

The evaluator writes:

- `eval_curve.csv`: pooled FID, raw CLIP FD, FD-r-CLIP, and alpha metadata;
- `eval_replicates.csv`: the same quantities per seed replicate;
- `summary.json`: Inception and CLIP slopes, conflict score, bootstrap
  confidence intervals, monotonicity counts, and complete provenance;
- `dose_response.png`: FID and FD-r-CLIP against alpha with confidence bands;
- a visualization of the learned base tile and representative perturbed
  samples.

Bootstrap resamples the independent seed replicates and refits both slopes for
each draw. The confidence interval is the percentile 95% interval. The
conflict score is:

`max(-beta_inception, 0) * max(beta_clip, 0)`.

## Launcher

`hacking/table_3_pMF_patch.sh` mirrors the portable structure and preset table
of `scripts/table_3_pMF.sh`.

It preserves:

- pMF B/L/H checkpoint names;
- 256/512 image options;
- CFG, interval, noise-scale, RoPE, learned positional embedding, and
  `disable_v_head` settings;
- `torchrun` single- and multi-node arguments;
- environment overrides for checkpoint root, output root, GPU count, batch
  size, and W&B.

It changes the Python entry point to the patch trainer, passes only Inception
as the training representation, and exposes patch loss, alpha, EMA-label, and
train-sample settings. A separate evaluation mode invokes the post-hoc
Inception-plus-CLIP evaluator.

## Failure Handling

The programs fail early with actionable messages when:

- the pMF checkpoint or required reference-statistics files are absent;
- the checkpoint has no actual EMA weights;
- the requested EMA label is unavailable;
- an alpha is negative or the alpha sequence is not strictly increasing from
  zero;
- train and test seed namespaces overlap;
- the distributed layout cannot produce equal per-rank feature batches;
- fewer than two global samples reach an empirical FID computation;
- any generator or Inception parameter is trainable;
- the optimizer contains anything other than `u`;
- a loss, gradient, covariance, or reported metric becomes non-finite.

Resume requires the same pMF preset, EMA label, alpha semantics, world size,
and per-rank batch size. A mismatch raises an error instead of changing the
fixed sample set.

## Verification

CPU tests cover:

- patch shape, periodic tiling, clipping, and `[0,1]`/`[-1,1]` conversion;
- zero-mean projection and periodic TV;
- deterministic disjoint seed derivation;
- sample-space FID value and gradient agreement with covariance-space FID on
  small synthetic inputs;
- monotonic hinge activation and loss gradients;
- slope, conflict score, and bootstrap reproducibility;
- checkpoint validation and optimizer parameter auditing.

Static checks run `compileall` and every CLI `--help`.

A GPU smoke test uses a tiny image count, two alpha values, one optimization
step, and one evaluation replicate. It verifies that a patch checkpoint and
result files are produced and that no generator or representation parameter
receives a gradient.

Full 10k/50k execution is not part of automated tests because it requires the
external pMF checkpoint, pretrained representation weights, reference
statistics, and substantial GPU time.
