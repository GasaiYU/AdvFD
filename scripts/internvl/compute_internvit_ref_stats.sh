#!/usr/bin/env bash
# Compute ImageNet reference FD stats for InternViT-300M vision CLS features.

export HF_HOME=${HF_HOME:-/mmu-vcg/gaomingju/data/models/}
export TORCH_HOME=${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}
# export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

set -euo pipefail

: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${GPUS_PER_NODE:=8}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=10}"
: "${IMG_SIZE:=256}"
: "${INTERNVIT:=OpenGVLab/InternViT-300M-448px}"
: "${INTERNVIT_TARGET_SIZE:=224}"
: "${OUTPUT_DIR:=data/fid_stats}"

OUT_NAME="internvit_300m_in${IMG_SIZE}_t${INTERNVIT_TARGET_SIZE}_stats.npz"

torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    compute_repr_stats.py \
    --model "$INTERNVIT" \
    --data_path "$DATA_ROOT" \
    --img_size "$IMG_SIZE" \
    --target_size "$INTERNVIT_TARGET_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUT_NAME" \
    "$@"
