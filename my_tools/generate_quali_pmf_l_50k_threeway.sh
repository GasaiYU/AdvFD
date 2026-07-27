#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EVAL_BSZ="${EVAL_BSZ:-256}"
NOISE_SEED="${NOISE_SEED:-20260717}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mmu-vcg/gaomingju/data/FD-Loss/quali/pmf_l}"
ORIGINAL_CKPT="${ORIGINAL_CKPT:-checkpoints/base/pMF-L_256.pth}"
FD_CKPT="${FD_CKPT:-checkpoints/post-trained/pMF-L_FD-Inception.pth}"
ADV_CKPT="${ADV_CKPT:-work_dirs/table_3_pMF/pMF_L_256-fd-sim-advinc-w0.05-advfreq2-detachreal-2e-6/checkpoints/step_0124999.pth}"

# 1000 ImageNet classes x 50 independent noises = 50,000 aligned samples.
# Columns: pMF-L | pMF-L + FD Loss | pMF-L + Adv FD. Per-model PNGs are
# removed only after all 50,000 three-way composites pass validation.
torchrun \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    my_tools/generate_quali_pmf_l.py \
    --output_root "${OUTPUT_ROOT}" \
    --original_checkpoint "${ORIGINAL_CKPT}" \
    --fd_checkpoint "${FD_CKPT}" \
    --adv_checkpoint "${ADV_CKPT}" \
    --class_start 0 \
    --class_count 1000 \
    --samples_per_class 50 \
    --eval_bsz "${EVAL_BSZ}" \
    --noise_seed "${NOISE_SEED}" \
    --threeway_only \
    "$@"
