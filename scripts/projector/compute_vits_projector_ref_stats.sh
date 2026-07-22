#!/usr/bin/env bash
# Compute ImageNet train reference stats in the frozen ViT-S or ViT-B projector space.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

: "${GPUS_PER_NODE:=8}"
: "${MASTER_PORT:=29632}"
: "${BATCH_SIZE:=256}"
: "${NUM_WORKERS:=10}"
: "${IMG_SIZE:=256}"
: "${OUTPUT_DIR:=data/fid_stats}"
: "${BACKBONE:=vit_s}"
: "${PROJECTOR_VERSION:=II}"
: "${HEAD_VARIANT:=default}"
: "${HEAD_CONV_LAYERS:=2}"
: "${PROJECTOR_DTYPE:=bf16}"
: "${PROJECTOR_USE_MUON:=1}"
HEAD_VARIANT_SUFFIX=""
if [[ "${HEAD_VARIANT}" != "default" ]]; then
    HEAD_VARIANT_SUFFIX="-${HEAD_VARIANT}"
fi
OPTIM_SUFFIX=""
if [[ "${PROJECTOR_USE_MUON}" = "1" ]]; then
    OPTIM_SUFFIX="-muon"
fi

case "${BACKBONE}" in
    vit_s|vits|ViT-S|ViT_S|s|S|small|vit-small|vit_small)
        BACKBONE="vit_s"
        BACKBONE_TAG="vits"
        DEFAULT_PATCH_SIZE=16
        DEFAULT_EMBED_DIM=384
        DEFAULT_DEPTH=12
        DEFAULT_NUM_HEADS=6
        DEFAULT_HEAD_HIDDEN_DIM=1536
        ;;
    vit_b|vitb|ViT-B|ViT_B|b|B|base|vit-base|vit_base)
        BACKBONE="vit_b"
        BACKBONE_TAG="vitb"
        DEFAULT_PATCH_SIZE=16
        DEFAULT_EMBED_DIM=768
        DEFAULT_DEPTH=12
        DEFAULT_NUM_HEADS=12
        DEFAULT_HEAD_HIDDEN_DIM=3072
        ;;
    *)
        echo "[ERR] unsupported BACKBONE=${BACKBONE}; expected vit_s or vit_b" >&2
        exit 1
        ;;
esac

: "${PATCH_SIZE:=${DEFAULT_PATCH_SIZE}}"
: "${EMBED_DIM:=${DEFAULT_EMBED_DIM}}"
: "${DEPTH:=${DEFAULT_DEPTH}}"
: "${NUM_HEADS:=${DEFAULT_NUM_HEADS}}"
: "${HEAD_HIDDEN_DIM:=${DEFAULT_HEAD_HIDDEN_DIM}}"
: "${PROJECTOR_EXP:=${BACKBONE_TAG}-token-v${PROJECTOR_VERSION}-inception-mae-siglip-mse-fdema${HEAD_VARIANT_SUFFIX}${OPTIM_SUFFIX}}"
: "${PROJECTOR_CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/projector/${PROJECTOR_EXP}/checkpoints/last.pt}"

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    scripts/projector/compute_projector_ref_stats.py \
    --data_path "${DATA_ROOT}" \
    --checkpoint "${PROJECTOR_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --dtype "${PROJECTOR_DTYPE}" \
    --projector_version "${PROJECTOR_VERSION}" \
    --backbone "${BACKBONE}" \
    --patch_size "${PATCH_SIZE}" \
    --embed_dim "${EMBED_DIM}" \
    --depth "${DEPTH}" \
    --num_heads "${NUM_HEADS}" \
    --head_variant "${HEAD_VARIANT}" \
    --head_mlp_layers 1 \
    --head_hidden_dim "${HEAD_HIDDEN_DIM}" \
    --head_conv_layers "${HEAD_CONV_LAYERS}" \
    "$@"
