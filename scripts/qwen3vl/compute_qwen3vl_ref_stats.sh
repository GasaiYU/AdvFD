#!/usr/bin/env bash
# Compute ImageNet reference FD stats for Qwen3-VL-2B vision features.

export HF_HOME=${HF_HOME:-/mmu-vcg/gaomingju/data/models/}
export TORCH_HOME=${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

set -euo pipefail

: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${GPUS_PER_NODE:=8}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=10}"
: "${IMG_SIZE:=256}"
: "${QWEN_TARGET_SIZE:=256}"
: "${OUTPUT_DIR:=data/fid_stats}"

OUT_NAME="qwen3vl_2b_in${IMG_SIZE}_t${QWEN_TARGET_SIZE}_stats.npz"

torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    compute_repr_stats.py \
    --model qwen3vl_2b \
    --data_path "$DATA_ROOT" \
    --img_size "$IMG_SIZE" \
    --target_size "$QWEN_TARGET_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUT_NAME" \
    "$@"
