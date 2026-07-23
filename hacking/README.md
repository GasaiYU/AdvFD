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
gradient first accumulates the global feature sum
`S = sum(f)` and outer-product sum `Q = sum(ff^T)`, then backpropagates
`dFD/dS` and `dFD/dQ` batch by batch. Consequently this is the gradient of one
joint 50k-sample FD, not an average of statistically unreliable batch FIDs.

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
