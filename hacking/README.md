# Cached Fourier universal pattern for pMF

This directory contains a fast, leakage-controlled alternative to long Adam
training of a tiled universal pattern:

1. generate and cache disjoint pMF samples (`50k optimization / 5k validation /
   50k test`);
2. unload and freeze the generator;
3. compute one joint Inception FD over all 50k optimization images and
   differentiate it in a 288-parameter Fourier subspace
   (`48 modes × RGB × cosine/sine`);
4. optionally run a small number of RMS-normalized PGD updates with random cyclic
   phase shifts;
5. select the pattern using only the fixed 5k validation Inception FID;
6. load CLIP for the first time and evaluate the selected pattern on the
   independent 50k test cache.

The pattern is exactly zero-mean per channel and RMS-normalized. The amplitude
is controlled only by `alpha`; no `tanh` amplitude warm-up is used.

Every optimization update uses the complete 50k optimization split. Images are
streamed in small batches only to control memory. A two-pass sufficient-stat
gradient first forms the global mean and covariance in float64, differentiates
FD—including the 2048-D covariance square root—in float64, then backpropagates
the resulting per-feature gradients batch by batch. Consequently this is the
gradient of one joint 50k-sample FD, not an average of statistically unreliable
batch FIDs.

## Default run

For pMF-B at 256 px on 8 GPUs:

```bash
ENABLE_WANDB=0 \
bash hacking/run_pmf_fourier_universal.sh
```

The expected checkpoint is `checkpoints/base/pMF-B_256.pth`. The run also
requires:

- Inception reference stats:
  `data/fid_stats/guided_diffusion_stats.npz`;
- CLIP reference stats inferred by the repository for
  `vit_large_patch14_clip_224.openai`.

The first run creates a sharded uint8 cache under
`work_dirs/hacking_cache/pMF_B_256`. At 256 px, 105k RGB images require about
20.6 GB. Later runs reuse it and skip generator inference.

The default is a single full-50k one-shot gradient (`PGD_STEPS=0`). Optional
refinement can be enabled explicitly, for example:

```bash
PGD_STEPS=3 bash hacking/run_pmf_fourier_universal.sh
```

Each PGD step is another two-pass gradient over all 50k optimization images,
so large PGD step counts are intentionally not the default.

## Useful modes

Strictly overfit one fixed 50k set, with no validation/test split and no CLIP:

```bash
EXP_NAME=pMF_B_256-fourier-overfit50k \
PGD_STEPS=5 \
bash hacking/run_pmf_fourier_universal.sh \
  --hack_overfit_only
```

This mode disables random phase automatically, evaluates both signs of the
one-shot direction, and uses the same 50k Inception FID for optimization,
backtracking, checkpoint selection, and the final dose response. It writes
`overfit_history.csv`, `overfit_dose_response.csv`, and
`overfit_summary.json`. The result is an optimizer/debug sanity check only;
it is not evidence of held-out generalization. If a compatible full
50k/5k/50k cache already exists, its optimization shard is reused. Otherwise
the mode creates only the 50k optimization cache.

Continue from a pattern selected at a smaller alpha instead of taking another
normalized finite jump from zero:

```bash
EXP_NAME=pMF_B_256-fourier-overfit-alpha01 \
PGD_STEPS=20 \
bash hacking/run_pmf_fourier_universal.sh \
  --hack_overfit_only \
  --hack_init_pattern_checkpoint \
  /path/to/alpha005/checkpoints/fourier_pattern_overfit_selected.pth \
  --hack_train_alpha 0.1 \
  --hack_pgd_step_size 0.02 \
  --hack_eval_alphas 0 0.05 0.1 0.15 0.2
```

The continuation run still computes the zero-pattern FID as its baseline and
will only save a nonzero final selection if that pattern, or a PGD refinement
of it, beats zero at the new target alpha. Use a new `EXP_NAME` so the source
checkpoint is never overwritten.

To search automatically from scratch for the strongest visibly textured
pattern, run the alpha curriculum:

```bash
bash hacking/run_pmf_visible_pattern_curriculum.sh
```

The checked-in defaults use pMF-B/256, 8 GPUs and the fixed 50k optimization
cache, enable W&B, and configure all curriculum/PGD hyperparameters. No
environment variables or pretrained pattern checkpoint are required.

By default it tries
`0.01, 0.02, 8/255, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2`. The small first
stage keeps the finite RMS-normalized jump within a useful local-gradient
regime. It learns from an exactly zero pattern; later stages initialize only
from the preceding successful nonzero selection. The initial stage uses 50 PGD
steps and later stages use 20. It adopts a stage only when that selection beats
the zero-pattern FID on the fixed 50k set. At the first failed stage it stops
and prints the strongest successful alpha and checkpoint instead of
accidentally continuing from the zero fallback. Override `ALPHA_STAGES`,
`INITIAL_PGD_STEPS`, `PGD_STEPS_PER_STAGE`, or `PGD_STEP_SIZE` through
environment variables when a finer or more intensive search is needed.

Generate/verify the three caches and stop:

```bash
bash hacking/run_pmf_fourier_universal.sh --hack_cache_only
```

Optimize and validate without loading CLIP or running the 50k final test:

```bash
bash hacking/run_pmf_fourier_universal.sh --hack_skip_final_eval
```

Evaluate an already selected pattern:

```bash
bash hacking/run_pmf_fourier_universal.sh \
  --hack_eval_only \
  --hack_pattern_checkpoint \
  work_dirs/hacking_pMF_fourier/pMF_B_256-cached-fourier-inception/checkpoints/fourier_pattern_selected.pth
```

Run a different pMF size/resolution:

```bash
MODEL_SIZE=L RES=256 \
bash hacking/run_pmf_fourier_universal.sh
```

If generation settings or world size change, use a different `CACHE_ROOT`.
`--hack_overwrite_cache` explicitly rewrites cache shards when the manifest
does not match.

## Main outputs

Under
`work_dirs/hacking_pMF_fourier/<experiment>/`:

- `checkpoints/fourier_pattern_best.pth`: best validation checkpoint so far;
- `checkpoints/fourier_pattern_selected.pth`: restored selected checkpoint;
- `fourier_pattern.npy`: exact normalized `3×16×16` pattern;
- `fourier_pattern.png`: tiled visualization;
- `fourier_dose_response.csv`: 50k test Inception/CLIP values by alpha;
- `fourier_summary.json`: slopes, bootstrap confidence intervals and conflict
  score.

CLIP is never used for one-shot direction estimation, PGD, early stopping, or
checkpoint selection.

## Extract a fixed natural-spectrum pattern without training

Extract the shared high-frequency RGB statistics of the cached low-FID 50k
pMF images and synthesize one deterministic full-resolution pattern:

```bash
python hacking/extract_spectral_pattern.py
```

All defaults are internalized:

```text
cache:       work_dirs/hacking_cache/pMF_B_256
split:       optimization
images:      50000
blur sigma:  2.0
seed:        2026
device:      cuda
output:      work_dirs/spectral_pattern/pMF_B_256
```

This is a single-GPU streaming statistic, not a training job: it has no loss,
backward pass, optimizer, Inception, or CLIP. It reads every rank's existing
cache shards in deterministic rank/range order, estimates the high-pass RGB
cross-spectrum, and performs seeded closed-form spectral synthesis.

The main outputs are:

- `spectral_pattern.npy`: exact zero-mean, unit-RMS `3×256×256` pattern;
- `spectral_pattern.png`: robustly mapped visualization;
- `spectral_stats.npz`: extracted RGB cross-spectrum and PSD;
- `radial_psd.csv`: source/pattern radial frequency profiles;
- `manifest.json`: cache identity, parameters, diagnostics, and hashes.

Apply it with the existing image utility:

```bash
python hacking/apply_fourier_pattern.py \
  --input_dir /path/to/input_images \
  --output_dir hacking/images/spectral_alpha_010 \
  --pattern work_dirs/spectral_pattern/pMF_B_256/spectral_pattern.npy \
  --alpha 0.1
```

Existing extraction outputs are protected. Pass `--overwrite` only when an
intentional rewrite with the same output directory is required.

## Apply a selected pattern to images

Apply the exact exported spatial pattern to every image under a directory:

```bash
python hacking/apply_fourier_pattern.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/patched_images \
  --pattern \
  work_dirs/pMF_universal_pattern/pMF_B_256-fourier-overfit50k/fourier_pattern.npy \
  --alpha 0.0313725490196
```

The default `--alpha_space model` matches the pMF experiment: images are
converted from `[0,1]` to `[-1,1]`, the tiled pattern is added and clipped,
then images are converted back. Thus model-space alpha `8/255` has
pixel-space RMS `4/255`.

The selected checkpoint can be used directly instead of the `.npy` file:

```bash
python hacking/apply_fourier_pattern.py \
  --input_dir /path/to/input_images \
  --output_dir /path/to/patched_images \
  --pattern \
  work_dirs/pMF_universal_pattern/pMF_B_256-fourier-overfit50k/checkpoints/fourier_pattern_overfit_selected.pth \
  --alpha 0.0313725490196
```

Overfit training and evaluation use phase `(0,0)`, which is also the
application default. Existing outputs are protected unless `--overwrite` is
passed. Each output directory includes `apply_pattern_manifest.json` with the
pattern, alpha, phase, and image count.

Use `--random_sample 10 --sample_seed 2026` to apply the pattern to a
reproducible random subset of ten images. The selected relative paths are
recorded in the output manifest.
