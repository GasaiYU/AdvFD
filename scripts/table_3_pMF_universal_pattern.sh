#!/usr/bin/env bash
# Learn one universal 16x16 dose-response pattern on a frozen pMF generator.
# Inception is the only training judge; CLIP is loaded only for held-out eval.

set -euo pipefail

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${EVAL_BSZ_PER_GPU:=256}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${PATTERN_EPOCHS:=100}"
: "${STEPS_PER_EPOCH:=1250}"
: "${QUEUE_SIZE:=50000}"
: "${EVAL_IMAGES:=50000}"
: "${EVAL_BLOCKS:=10}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
fi
if (( QUEUE_SIZE % TOTAL_GPUS != 0 )); then
    echo "[ERR] QUEUE_SIZE=${QUEUE_SIZE} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
fi
if (( EVAL_IMAGES % EVAL_BLOCKS != 0 )); then
    echo "[ERR] EVAL_IMAGES=${EVAL_IMAGES} must be divisible by EVAL_BLOCKS=${EVAL_BLOCKS}"
    exit 1
fi
if (( (EVAL_IMAGES / EVAL_BLOCKS) % TOTAL_GPUS != 0 )); then
    echo "[ERR] each eval block must be divisible by TOTAL_GPUS=${TOTAL_GPUS}; adjust EVAL_BLOCKS"
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

case "${MODEL_SIZE}-${RES}" in
    B-256)
        MODEL=pMF_B; CFG=8.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-B_256.pth"; EXTRA=() ;;
    L-256)
        MODEL=pMF_L; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-L_256.pth"; EXTRA=() ;;
    H-256)
        MODEL=pMF_H; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-H_256.pth"; EXTRA=(--noise_scale 2.0) ;;
    B-512)
        MODEL=pMF_B; CFG=6.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-B_512.pth"
        EXTRA=(--noise_scale 2.0 --img_size 512 --patch_size 32) ;;
    L-512)
        MODEL=pMF_L; CFG=7.5; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-L_512.pth"
        EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    H-512)
        MODEL=pMF_H; CFG=5.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-H_512.pth"
        EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE} RES=${RES}"; exit 1 ;;
esac

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    main_pmf_universal_pattern.py \
    --project pMF_universal_pattern \
    --exp_name "${MODEL}_${RES}-inception-dose-response" \
    --load_from "$LOAD" \
    --model "$MODEL" --rope_2d --learned_pe --disable_v_head \
    --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
    --num_sampling_steps 1 \
    --batch_size "$BATCH_SIZE" --eval_bsz "$EVAL_BSZ_PER_GPU" \
    --epochs "$PATTERN_EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" \
    --warmup_epochs 5 \
    --queue_size "$QUEUE_SIZE" \
    --pattern_eval_images "$EVAL_IMAGES" \
    --pattern_size 16 \
    --pattern_train_alphas 0 0.0078431372549 0.0156862745098 0.0235294117647 0.0313725490196 \
    --pattern_eval_alphas 0 0.00392156862745 0.0078431372549 0.0117647058824 0.0156862745098 0.0196078431373 0.0235294117647 0.0274509803922 0.0313725490196 \
    --pattern_mono_weight 1.0 --pattern_mono_margin 0.01 \
    --pattern_reg_weight 1e-3 --pattern_mean_weight 10.0 \
    --pattern_eval_blocks "$EVAL_BLOCKS" --pattern_bootstrap_repeats 10000 \
    --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
    --grad_checkpointing \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --pattern_grad_clip 0 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --num_images_for_eval_and_search "$EVAL_IMAGES" \
    --auto_resume --dtype bf16 \
    "$WANDB_FLAG" \
    "${EXTRA[@]}" \
    "$@"
