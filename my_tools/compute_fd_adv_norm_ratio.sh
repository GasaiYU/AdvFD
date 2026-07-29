#!/usr/bin/env bash
# Compute the adversarial/original FD repr norm ratio on ImageNet.
#
# Required only when defaults do not fit your environment:
#   CKPT_DIR   checkpoint directory or file
#   DATA_ROOT  ImageNet root with train/ or val/
#
# Optional overrides:
#   GPUS_PER_NODE=8
#   SPLIT=train|val
#   NUM_IMAGES=10000
#   BATCH_SIZE=256
#   NUM_WORKERS=8
#   REPR_MODEL=inception
#   POOL_TYPE=cls|avg
#   OUTPUT_JSON=path/to/result.json

set -euo pipefail

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${CKPT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/Jit-adv-ablation/JiT_B-fd-sim-advinc-w0.1-from-base-ADVLR-1e-6/checkpoints}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K}"
: "${GPUS_PER_NODE:=8}"
: "${SPLIT:=train}"
: "${NUM_IMAGES:=10000}"
: "${BATCH_SIZE:=256}"
: "${NUM_WORKERS:=8}"
: "${REPR_MODEL:=inception}"
: "${POOL_TYPE:=cls}"

ARGS=(
    --checkpoint "$CKPT_DIR"
    --data_root "$DATA_ROOT"
    --split "$SPLIT"
    --num_images "$NUM_IMAGES"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --repr_model "$REPR_MODEL"
    --pool_type "$POOL_TYPE"
)

if [[ -n "${OUTPUT_JSON:-}" ]]; then
    ARGS+=(--output_json "$OUTPUT_JSON")
fi

exec torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" \
    my_tools/compute_fd_adv_norm_ratio.py \
    "${ARGS[@]}" \
    "$@"
