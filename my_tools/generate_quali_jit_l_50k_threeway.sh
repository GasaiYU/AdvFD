#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EVAL_BSZ="${EVAL_BSZ:-16}"
NOISE_SEED="${NOISE_SEED:-20260717}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mmu-vcg/gaomingju/data/FD-Loss/quali}"

# 1000 ImageNet classes x 50 independent noises = 50,000 aligned samples.
# Only jit_l_threeway/ is retained after all outputs pass validation.
torchrun \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    my_tools/generate_quali_jit_l.py \
    --output_root "${OUTPUT_ROOT}" \
    --class_start 0 \
    --class_count 1000 \
    --samples_per_class 50 \
    --eval_bsz "${EVAL_BSZ}" \
    --noise_seed "${NOISE_SEED}" \
    --threeway_only \
    "$@"
