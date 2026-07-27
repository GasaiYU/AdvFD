#!/usr/bin/env bash
# Calibrate B = MARGIN * Q_QUANTILE(||Inception-pool3(x)||_2) on real ImageNet.

set -euo pipefail

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root containing train/}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${NUM_IMAGES:=50000}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=10}"
: "${IMG_SIZE:=256}"
: "${SAMPLING:=stratified}"
: "${SEED:=0}"
: "${QUANTILE:=0.999}"
: "${MARGIN:=1.05}"
: "${OUTPUT_JSON:=data/fid_stats/inception_pool3_feature_bound.json}"
: "${OUTPUT_NORMS:=}"
: "${OVERWRITE:=0}"

EXTRA_OUTPUT_ARGS=()
if [ -n "$OUTPUT_NORMS" ]; then
    EXTRA_OUTPUT_ARGS+=(--output_norms "$OUTPUT_NORMS")
fi
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_OUTPUT_ARGS+=(--overwrite)
fi

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    scripts/calibrate_inception_feature_bound.py \
    --data_path "$DATA_ROOT" \
    --num_images "$NUM_IMAGES" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --sampling "$SAMPLING" \
    --seed "$SEED" \
    --quantile "$QUANTILE" \
    --margin "$MARGIN" \
    --output_json "$OUTPUT_JSON" \
    "${EXTRA_OUTPUT_ARGS[@]}" \
    "$@"
