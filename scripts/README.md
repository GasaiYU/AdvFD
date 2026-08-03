# Experiment Scripts

Run all commands from the repository root. The scripts are explicit wrappers around
`torchrun` so the model, sampling, and FD-loss settings are easy to inspect.

## Assets

The Hugging Face release is organized as:

```text
checkpoints/
  base/                 base pMF, iMF, and JiT checkpoints
  post-trained/         FD-Inception and FD-SIM post-trained checkpoints
data/
  fid_stats/
    paper_ref_stats.pkl bundled paper reference statistics
  train.txt
  val.txt
  val_labeled.txt
```

Download everything:

```bash
hf download jjiaweiyang/FD-Loss \
  --local-dir . \
  --include "checkpoints/**/*.pth" \
  --include "data/**"
```

Download only what is needed for released-checkpoint evaluation:

```bash
hf download jjiaweiyang/FD-Loss \
  --local-dir . \
  --include "checkpoints/post-trained/*.pth" \
  --include "data/**"
```

Unpack the bundled reference statistics:

```bash
python scripts/extract_paper_ref_stats.py
```

## Environment

Required inputs:

```bash
export DATA_ROOT=/path/to/imagenet
export CKPT_ROOT=./checkpoints/base
```

Single-node defaults:

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export GPUS_PER_NODE=8
```

For multi-node runs, execute the same command on every node and set `NODE_RANK`
to `0..NNODES-1`.

## Evaluation

The evaluation script uses one preset per model family:

```bash
PRESET=pMF_H_256 CKPT_PATH=checkpoints/post-trained/pMF-H_FD-SIM.pth \
GPUS_PER_NODE=8 bash scripts/evaluate_released_ckpt.sh

PRESET=iMF_XL CKPT_PATH=checkpoints/post-trained/iMF-XL_FD-SIM.pth \
GPUS_PER_NODE=8 bash scripts/evaluate_released_ckpt.sh

PRESET=JiT_H CKPT_PATH=checkpoints/post-trained/JiT-H_FD-SIM.pth \
GPUS_PER_NODE=8 bash scripts/evaluate_released_ckpt.sh
```

Available presets:

```text
pMF_B_256  pMF_L_256  pMF_H_256
pMF_B_512  pMF_L_512  pMF_H_512
iMF_B      iMF_L      iMF_XL
JiT_B      JiT_L      JiT_H
```

The evaluator writes raw FD values and the paper-normalized metrics to
`final_eval_summary.csv`:

- `fd`: raw Fréchet distance in the selected representation space.
- `fdr`: raw FD divided by the validation-set raw FD for that representation.
- `fdr6`: arithmetic mean of FDr over Inception, ConvNeXt, DINOv2, MAE,
  SigLIP, and CLIP.

The released evaluator uses these validation-set raw FD values:

| Representation | Inception | ConvNeXt | DINOv2 | MAE | SigLIP | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| valFD | 1.68 | 56.87 | 14.19 | 0.04 | 0.60 | 5.60 |

To reproduce the validation-set normalizers from ImageNet validation images:

```bash
DATA_ROOT=/path/to/imagenet \
torchrun --nproc_per_node=8 scripts/compute_valfd.py \
  --data_root "$DATA_ROOT"
```

The script writes `data/fid_stats/valfd.json` and `data/fid_stats/valfd.csv`.

For a faster smoke test:

```bash
PRESET=JiT_B \
CKPT_PATH=checkpoints/post-trained/JiT-B_FD-Inception.pth \
NUM_IMAGES=1024 \
EVAL_BSZ=64 \
GPUS_PER_NODE=1 \
bash scripts/evaluate_released_ckpt.sh
```

## Training

Training starts from the released base checkpoints:

```bash
export CKPT_ROOT=./checkpoints/base
```

| Experiment | Command |
|---|---|
| Table 1a, queue-size ablation | `bash scripts/table_1a_queue_size.sh` |
| Table 1b, EMA-beta ablation | `bash scripts/table_1b_ema_beta.sh` |
| Table 1c, single-backbone ablation | `bash scripts/table_1c_backbone_single.sh` |
| Table 1c, multi-backbone ablation | `bash scripts/table_1c_backbone_combo.sh` |
| Table 2, JiT-L repurposing | `bash scripts/table_2_repurpose_jit_L.sh` |
| Table 3, pMF scalability | `MODEL_SIZE=L RES=256 bash scripts/table_3_pMF.sh` |
| Table 3, pMF adversarial FD, multi-node | `MODEL_SIZE=L RES=256 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/table_3_pMF_adv_multi.sh` |
| Frozen pMF universal pattern | `MODEL_SIZE=B RES=256 bash scripts/table_3_pMF_universal_pattern.sh` |
| Table 3, iMF scalability | `MODEL_SIZE=L bash scripts/table_3_iMF.sh` |
| Table 3, JiT scalability | `MODEL_SIZE=L bash scripts/table_3_JiT.sh` |

Weights & Biases logging is disabled by default. Enable it explicitly with
`ENABLE_WANDB=1` and pass `--entity` if your W&B account requires an entity.

## Reference Statistics

The released `paper_ref_stats.pkl` contains the paper reference statistics. To
regenerate ImageNet-derived statistics from your local ImageNet copy:

```bash
DATA_ROOT=/path/to/imagenet GPUS_PER_NODE=8 bash scripts/compute_ref_stats.sh
```

Common reference-statistics files:

```text
guided_diffusion_stats.npz
convnext_in256_t224_stats.npz
vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz
vit_large_patch14_clip_224_openai_in256_t256_stats.npz
vit_large_patch16_224_mae_in256_t224_stats.npz
vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz
jit_in256_stats.npz
```

### Frozen-pMF universal dose-response pattern

`table_3_pMF_universal_pattern.sh` freezes pMF and both representation
encoders, then learns only one shared `3x16x16` pattern with Inception FD.
CLIP is not loaded until held-out evaluation. Except for freezing pMF and
optimizing the shared pattern, it retains the original pMF defaults:
50k FD initialization samples, `fd_ema_beta=0.999`, global batch 1024,
100x1250 training steps, five warmup epochs, and a `1e-6` cosine learning
rate. Evaluation uses a disjoint 50k-generation alpha scan:

```bash
MODEL_SIZE=B RES=256 GPUS_PER_NODE=8 \
CKPT_ROOT=./checkpoints/base \
bash scripts/table_3_pMF_universal_pattern.sh
```

Outputs are written below
`work_dirs/pMF_universal_pattern/<experiment>/`: pattern checkpoints and
previews, `pattern_dose_response.csv`, and
`pattern_dose_response_summary.json` (slopes, bootstrap confidence intervals,
conflict score, and the sign-test result).
