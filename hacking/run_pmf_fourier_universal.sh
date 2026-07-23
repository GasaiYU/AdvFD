#!/usr/bin/env bash
# Cached, low-dimensional universal Inception-FD direction for frozen pMF.
#
# First run: generate fixed 50k/5k/50k uint8 cache, optimize the Fourier
# pattern, validate with Inception only, then evaluate Inception + CLIP.
# Later runs reuse the cache and never load the generator during optimization.

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
: "${GLOBAL_CACHE_BSZ:=1024}"
: "${GRADIENT_BSZ:=256}"
: "${EVAL_BSZ_PER_GPU:=128}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${OPT_IMAGES:=50000}"
: "${VAL_IMAGES:=5000}"
: "${TEST_IMAGES:=50000}"
: "${PGD_STEPS:=0}"
: "${CACHE_ROOT:=./work_dirs/hacking_cache}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_CACHE_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_CACHE_BSZ=${GLOBAL_CACHE_BSZ} must divide by TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
fi
if (( GRADIENT_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GRADIENT_BSZ=${GRADIENT_BSZ} must divide by TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
fi
for split_size in "$OPT_IMAGES" "$VAL_IMAGES" "$TEST_IMAGES"; do
    if (( split_size % TOTAL_GPUS != 0 )); then
        echo "[ERR] cache split size ${split_size} must divide by TOTAL_GPUS=${TOTAL_GPUS}"
        exit 1
    fi
done
if (( TEST_IMAGES % 10 != 0 || (TEST_IMAGES / TOTAL_GPUS) % 10 != 0 )); then
    echo "[ERR] TEST_IMAGES and every per-rank test shard must divide by 10 eval blocks"
    exit 1
fi

LOCAL_CACHE_BSZ=$(( GLOBAL_CACHE_BSZ / TOTAL_GPUS ))
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
    *)
        echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE} RES=${RES}"
        exit 1 ;;
esac

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    hacking/pmf_fourier_universal.py \
    --project pMF_universal_pattern \
    --exp_name "${MODEL}_${RES}-cached-fourier-inception" \
    --load_from "$LOAD" \
    --model "$MODEL" --rope_2d --learned_pe --disable_v_head \
    --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
    --num_sampling_steps 1 \
    --batch_size "$LOCAL_CACHE_BSZ" \
    --eval_bsz "$EVAL_BSZ_PER_GPU" \
    --epochs 1 --steps_per_epoch 1 --warmup_epochs 0 \
    --hack_cache_dir "${CACHE_ROOT}/${MODEL}_${RES}" \
    --hack_cache_dtype uint8 \
    --hack_optimization_images "$OPT_IMAGES" \
    --hack_validation_images "$VAL_IMAGES" \
    --hack_test_images "$TEST_IMAGES" \
    --hack_gradient_batch_size "$GRADIENT_BSZ" \
    --hack_pattern_size 16 \
    --hack_fourier_modes 48 \
    --hack_fourier_min_radius 0.15 \
    --hack_fourier_max_radius 0.55 \
    --hack_train_alpha 0.0313725490196 \
    --hack_cov_eps 0 \
    --hack_pgd_steps "$PGD_STEPS" \
    --hack_pgd_step_size 0.25 \
    --hack_validate_every 5 \
    --hack_early_stop_patience 2 \
    --hack_eval_alphas 0 0.0078431372549 0.0156862745098 0.0313725490196 0.0627450980392 \
    --hack_eval_blocks 10 \
    --hack_bootstrap_repeats 10000 \
    --fd_eigvalsh \
    --num_images_for_eval_and_search "$TEST_IMAGES" \
    --dtype bf16 \
    "$WANDB_FLAG" \
    "${EXTRA[@]}" \
    "$@"
