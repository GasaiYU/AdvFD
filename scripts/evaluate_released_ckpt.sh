#!/usr/bin/env bash
# Evaluate one released checkpoint with plain torchrun.
#
# Required:
#   CKPT_PATH=/path/to/checkpoint.pth
#
# Common presets:
#   pMF_B_256, pMF_L_256, pMF_H_256, pMF_B_512, pMF_L_512, pMF_H_512
#   iMF_B, iMF_L, iMF_XL
#   JiT_B, JiT_L, JiT_H

set -euo pipefail

: "${CKPT_PATH:?Set CKPT_PATH to a released checkpoint}"
: "${PRESET:=pMF_B_256}"
: "${GPUS_PER_NODE:=8}"
: "${MASTER_PORT:=29500}"
: "${RESULT_ROOT:=./work_dirs/eval_results}"
: "${PROJECT:=eval_released}"
: "${EVAL_BSZ:=128}"
: "${NUM_IMAGES:=50000}"

EXTRA=()
case "$PRESET" in
    pMF_B_256) MODEL=pMF_B; CFG=8.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        EXTRA=(--rope_2d --learned_pe --disable_v_head) ;;
    pMF_L_256) MODEL=pMF_L; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.7
        EXTRA=(--rope_2d --learned_pe --disable_v_head) ;;
    pMF_H_256) MODEL=pMF_H; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        EXTRA=(--rope_2d --learned_pe --disable_v_head --noise_scale 2.0) ;;
    pMF_B_512) MODEL=pMF_B; CFG=6.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        EXTRA=(--rope_2d --learned_pe --disable_v_head --noise_scale 2.0 --img_size 512 --patch_size 32) ;;
    pMF_L_512) MODEL=pMF_L; CFG=7.5; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        EXTRA=(--rope_2d --learned_pe --disable_v_head --noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    pMF_H_512) MODEL=pMF_H; CFG=5.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.6
        EXTRA=(--rope_2d --learned_pe --disable_v_head --noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    iMF_B) MODEL=iMF_B; CFG=8.0; INTERVAL_MIN=0.4; INTERVAL_MAX=0.65
        EXTRA=(--tokenizer sdvae --tokenizer_patch_size 8 --patch_size 2 --disable_v_head) ;;
    iMF_L) MODEL=iMF_L; CFG=10.5; INTERVAL_MIN=0.4; INTERVAL_MAX=0.6
        EXTRA=(--tokenizer sdvae --tokenizer_patch_size 8 --patch_size 2 --disable_v_head) ;;
    iMF_XL) MODEL=iMF_XL; CFG=8.0; INTERVAL_MIN=0.42; INTERVAL_MAX=0.62
        EXTRA=(--tokenizer sdvae --tokenizer_patch_size 8 --patch_size 2 --disable_v_head) ;;
    JiT_B) MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        EXTRA=(--rope_2d --learned_pe --legacy_time_convention --ema_type edm) ;;
    JiT_L) MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        EXTRA=(--rope_2d --learned_pe --legacy_time_convention --ema_type edm) ;;
    JiT_H) MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        EXTRA=(--rope_2d --learned_pe --legacy_time_convention --ema_type edm) ;;
    *) echo "[ERR] unsupported PRESET=${PRESET}"; exit 1 ;;
esac

CFG="${CFG_OVERRIDE:-$CFG}"
EXP_NAME="${EXP_NAME:-${PRESET}-$(basename "$CKPT_PATH" .pth)}"

torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" \
    eval_all_fds.py \
    --model "$MODEL" \
    "${EXTRA[@]}" \
    --cfg "$CFG" --cfg_list "$CFG" \
    --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
    --num_sampling_steps 1 \
    --eval_ema_labels online \
    --disable_wandb --no_prc \
    --eval_bsz "$EVAL_BSZ" \
    --num_images "$NUM_IMAGES" \
    --load_from "$CKPT_PATH" \
    --output_dir "$RESULT_ROOT" \
    --project "$PROJECT" \
    --exp_name "$EXP_NAME" \
    "$@"
