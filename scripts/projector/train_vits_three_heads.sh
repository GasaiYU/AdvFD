#!/usr/bin/env bash
# Train a ViT-S or ViT-B projector with heads for Inception, MAE, and SigLIP features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29630}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=2048}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${PROJECT:=projector}"
: "${BACKBONE:=vit_s}"
: "${PROJECTOR_VERSION:=II}"
: "${HEAD_VARIANT:=default}"
: "${HEAD_MLP_LAYERS:=1}"
: "${HEAD_CONV_LAYERS:=2}"
: "${INCEPTION_HEAD_WEIGHT:=2.5}" # fix 1,0
: "${MAE_HEAD_WEIGHT:=0.25}"
: "${SIGLIP_HEAD_WEIGHT:=1.0}"
: "${FD_LOSS_WEIGHT:=0.3}"
: "${FD_QUEUE_SIZE:=50000}"
: "${FD_EMA_BETA:=0.999}"
: "${USE_MUON:=1}"
: "${MUON_LR:=1e-3}"
: "${MUON_MOMENTUM:=0.95}"
: "${MUON_WEIGHT_DECAY:=0.0}"

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

HEAD_VARIANT_SUFFIX=""
if [[ "${HEAD_VARIANT}" != "default" ]]; then
    HEAD_VARIANT_SUFFIX="-${HEAD_VARIANT}"
fi
OPTIM_SUFFIX=""
if [[ "${USE_MUON}" = "1" ]]; then
    OPTIM_SUFFIX="-muon"
fi
: "${EXP_NAME:=${BACKBONE_TAG}-token-v${PROJECTOR_VERSION}-inception-mae-siglip-mse-fdema${HEAD_VARIANT_SUFFIX}${OPTIM_SUFFIX}}"
: "${ENABLE_WANDB:=1}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" = "1" ]]; then
    WANDB_ARGS+=(--enable_wandb)
fi

OPTIM_ARGS=()
if [[ "${USE_MUON}" = "1" ]]; then
    OPTIM_ARGS+=(
        --use_muon
        --muon_lr "${MUON_LR}"
        --muon_momentum "${MUON_MOMENTUM}"
        --muon_weight_decay "${MUON_WEIGHT_DECAY}"
    )
fi

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    train_projector.py \
    --data_path "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --project "${PROJECT}" \
    --exp_name "${EXP_NAME}" \
    --img_size 256 \
    --backbone "${BACKBONE}" \
    --patch_size "${PATCH_SIZE}" \
    --embed_dim "${EMBED_DIM}" \
    --depth "${DEPTH}" \
    --num_heads "${NUM_HEADS}" \
    --projector_version "${PROJECTOR_VERSION}" \
    --head_variant "${HEAD_VARIANT}" \
    --head_mlp_layers "${HEAD_MLP_LAYERS}" \
    --head_hidden_dim "${HEAD_HIDDEN_DIM}" \
    --head_conv_layers "${HEAD_CONV_LAYERS}" \
    --teacher_models inception vit_large_patch16_224.mae vit_so400m_patch16_siglip_256.v2_webli \
    --teacher_target_sizes 256 224 224 \
    --head_weights "${INCEPTION_HEAD_WEIGHT}" "${MAE_HEAD_WEIGHT}" "${SIGLIP_HEAD_WEIGHT}" \
    --loss_type mse \
    --fd_loss_weight "${FD_LOSS_WEIGHT}" \
    --fd_queue_size "${FD_QUEUE_SIZE}" \
    --fd_ema_beta "${FD_EMA_BETA}" \
    --fd_eigvalsh \
    --batch_size "${BATCH_SIZE}" \
    --epochs 300 \
    --steps_per_epoch 1250 \
    --warmup_epochs 5 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --lr_sched cosine \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    "${OPTIM_ARGS[@]}" \
    --dtype bf16 \
    --teacher_dtype bf16 \
    --num_workers 10 \
    --print_freq 20 \
    --save_every 5000 \
    --auto_resume \
    "${WANDB_ARGS[@]}" \
    "$@"
