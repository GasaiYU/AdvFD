#!/usr/bin/env bash
# JiT FD-Inception training with common/residual reweighting and bounded NormPreserve.
# Set MODEL_SIZE in {B,L,H}. Hyperparameters can be overridden via environment variables.

set -euo pipefail

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${FD_REPR_GRAD_CHECKPOINT_MODELS:=none}"
: "${FD_GRAD_ALPHA:=0.5}"
: "${FD_GRAD_BETA:=1.0}"
: "${FD_GRAD_NORMPRESERVE_MAX_SCALE:=1.5}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

WANDB_ARGS=()
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_ARGS+=(--enable_wandb)
fi

FD_MEMORY_ARGS=()
if [[ -n "$FD_REPR_GRAD_CHECKPOINT_MODELS" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "none" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "0" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "off" ]]; then
    read -r -a FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS <<< "$FD_REPR_GRAD_CHECKPOINT_MODELS"
    FD_MEMORY_ARGS+=(--fd_repr_grad_checkpoint_models "${FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS[@]}")
fi

case "$MODEL_SIZE" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *)
        echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}" >&2
        exit 1 ;;
esac

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    main_fd.py \
    --project "table_3_JiT-${MODEL_SIZE}-norm" \
    --exp_name "${MODEL}-fd-inception-norm-a${FD_GRAD_ALPHA}-b${FD_GRAD_BETA}-s${FD_GRAD_NORMPRESERVE_MAX_SCALE}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --data_path "$DATA_ROOT" \
    --load_from "$LOAD" \
    --model "$MODEL" --rope_2d --learned_pe --legacy_time_convention \
    --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
    --ema_type edm \
    --num_sampling_steps 1 \
    --eval_bsz 256 --num_images_for_eval_and_search 50000 \
    --vis_freq 100 --online_eval --eval_freq 10000 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --grad_checkpointing \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --fd_repr_models inception \
    --fd_repr_pool_types cls \
    --fd_target_sizes 256 \
    --fd_grad_normpreserve \
    --fd_grad_alpha "$FD_GRAD_ALPHA" \
    --fd_grad_beta "$FD_GRAD_BETA" \
    --fd_grad_normpreserve_max_scale "$FD_GRAD_NORMPRESERVE_MAX_SCALE" \
    --auto_resume \
    "${WANDB_ARGS[@]}" \
    "${FD_MEMORY_ARGS[@]}" \
    "$@"
