# Spectral Texture Extraction Design

Date: 2026-07-24  
Status: approved for implementation planning

## Objective

Extract the shared low-level natural-image texture statistics from a fixed
50,000-image, low-FID pMF sample cache and synthesize one deterministic
full-resolution RGB pattern without optimizing FID or any pixel parameter.

The resulting pattern must:

- be produced by a closed-form, single-pass statistical procedure;
- contain no learned or gradient-optimized parameters;
- be zero-mean per channel and globally RMS-normalized;
- be reproducible from an explicit random seed;
- preserve broadband spatial-frequency and RGB cross-channel statistics;
- be directly usable by `hacking/apply_fourier_pattern.py`;
- avoid the frequency-comb artifact caused by tiling a 16×16 patch.

This component extracts and visualizes the texture. Evaluation on an
independent 50,000-image set remains a separate step so the extraction set
cannot be used as evidence of metric improvement.

## Non-goals

- Do not optimize Inception FID, CLIP distance, or another perceptual metric.
- Do not choose a seed or hyperparameter according to downstream FID.
- Do not recover semantic content, objects, class labels, or image layouts.
- Do not implement a learned generator, autoencoder, PCA model, or patch bank.
- Do not claim that the extracted pattern lowers held-out FID until it is
  evaluated independently.

## Input

The initial implementation supports the existing pMF cache format under:

```text
work_dirs/hacking_cache/pMF_B_256/
```

It discovers every `rank_*/index.json` for a requested split, reads its ordered
cache entries, and streams up to a requested number of images. Both cache
dtypes already supported by the repository are valid:

- `uint8`, decoded from `[0,255]` to `[0,1]`;
- `float16`, interpreted through the cache manifest's existing convention.

The default input is:

```text
split       = optimization
num_images  = 50000
resolution  = 256×256
```

The script rejects missing indices, duplicate or overlapping entry ranges,
mixed resolutions, unsupported channel counts, and a requested image count
larger than the cache.

## Extraction Algorithm

### 1. High-pass residual

For each RGB image \(x_i\in[0,1]^{3\times H\times W}\), compute:

\[
h_i=x_i-G_{\sigma}(x_i),\qquad \sigma=2.
\]

The Gaussian blur uses reflect padding. This removes object color and coarse
layout while retaining fur, feathers, foliage, water, rock, cloud edges, and
other texture components visible in the low-FID montage.

### 2. Boundary control

Multiply the residual by a separable two-dimensional Hann window normalized to
unit RMS. This prevents image crop boundaries from dominating the spectrum
without changing the expected residual energy.

### 3. RGB cross-spectrum

Compute an orthonormal real FFT:

\[
F_i=\operatorname{rFFT2}(\operatorname{Hann}(h_i)).
\]

For every retained frequency \(\omega\), accumulate the Hermitian
\(3\times3\) cross-spectral density:

\[
S(\omega)
=
\frac1N\sum_i
F_i(\omega)F_i(\omega)^{H}.
\]

Accumulation uses complex128 so 50,000 samples do not lose low-energy
high-frequency components. The final matrix is explicitly Hermitianized.

### 4. Closed-form deterministic synthesis

Generate a real three-channel spatial white-noise field using a fixed seed
(`2026` by default), then transform it with orthonormal `rFFT2` to obtain
frequency-domain noise \(z(\omega)\) with valid real-image symmetry.

For each frequency, eigendecompose:

\[
S(\omega)=V(\omega)\Lambda(\omega)V(\omega)^H
\]

and color the noise:

\[
\widehat r(\omega)
=
V(\omega)\sqrt{\max(\Lambda(\omega),0)}V(\omega)^H z(\omega).
\]

Set the DC coefficient to zero and reconstruct with `irFFT2`. Finally:

1. subtract each channel's spatial mean;
2. divide by the global RGB RMS;
3. verify finite values, mean tolerance, and RMS tolerance.

This produces one `3×256×256` stationary, non-periodic pattern. The seed fixes
only phase realization; its amplitude and RGB correlation come exclusively
from the measured cross-spectrum.

## Command-line Interface

The implementation adds:

```text
hacking/extract_spectral_pattern.py
```

Default invocation:

```bash
python hacking/extract_spectral_pattern.py \
  --cache_root ./work_dirs/hacking_cache/pMF_B_256 \
  --split optimization \
  --num_images 50000 \
  --output_dir ./work_dirs/spectral_pattern/pMF_B_256
```

Supported controls:

```text
--batch_size
--device
--blur_sigma
--seed
--num_images
--overwrite
```

Defaults are fixed in the parser and written to the manifest. The extraction
does not expose frequency-band selection, seed search, or FID-guided controls
in the first version.

## Outputs

The output directory contains:

- `spectral_stats.npz`:
  complex RGB cross-spectrum, mean channel residual energy, resolution, count;
- `spectral_pattern.npy`:
  exact float32 `3×H×W` zero-mean, unit-RMS pattern;
- `spectral_pattern.png`:
  robust percentile-mapped visualization;
- `spectral_pattern_tiled.png`:
  a display canvas showing the pattern at native scale without resampling;
- `radial_psd.csv`:
  radial frequency, source PSD, synthesized-pattern PSD;
- `manifest.json`:
  source cache identity, ordered index files, image count, algorithm version,
  blur sigma, seed, normalization diagnostics, and output hashes.

Writes are atomic. Existing outputs are rejected unless `--overwrite` is
explicitly supplied.

## Applying the Pattern

The `.npy` output is compatible with the existing application utility:

```bash
python hacking/apply_fourier_pattern.py \
  --input_dir /path/to/images \
  --output_dir hacking/images/spectral_alpha_010 \
  --pattern work_dirs/spectral_pattern/pMF_B_256/spectral_pattern.npy \
  --alpha 0.1
```

The pattern has native 256×256 resolution, so application to 256×256 pMF
images does not create tiled seams. The existing model-space alpha convention
is retained.

The main scientific result must later use another fixed 50,000-image sample
set. The primary condition uses exactly the same fixed phase for every image.
A deterministic cyclic roll keyed only by sample index may be reported as a
stationarity robustness check, not as the primary result.

## Diagnostics and Failure Handling

The script terminates with a clear error when:

- fewer than two images are available;
- a cache entry is missing or inconsistent with its index;
- input images have an unexpected dtype, shape, or numeric range;
- the accumulated spectrum contains NaN or infinity;
- the synthesized pattern has zero or non-finite RMS;
- output files already exist without `--overwrite`.

It logs processed images, throughput, residual RMS, spectrum trace, synthesized
mean/RMS, and output paths. It never silently falls back to a zero pattern.

## Verification

Automated tests cover:

1. deterministic equality for identical seed and input;
2. different phase realizations for different seeds while preserving radial
   PSD within tolerance;
3. per-channel mean below `1e-6`;
4. global RMS within `1e-5` of one;
5. Hermitian and positive-semidefinite cross-spectrum within numeric tolerance;
6. recovery of a known synthetic RGB-colored noise spectrum;
7. ordered multi-rank cache traversal without duplicates;
8. refusal to overwrite existing artifacts;
9. compatibility of `spectral_pattern.npy` with
   `hacking/apply_fourier_pattern.py`.

A small synthetic cache smoke test is sufficient locally. The complete
50,000-image extraction is run on the user's GPU server.

## Scientific Interpretation

This procedure estimates texture moments; it is not model training. It can
support the claim that a fixed, analytically constructed natural-spectrum
texture affects Inception FID only after held-out evaluation. CLIP must not be
used to set the spectrum, seed, alpha, or any extraction hyperparameter.
