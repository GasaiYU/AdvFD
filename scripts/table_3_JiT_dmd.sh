#!/usr/bin/env bash
# Table 3 JiT FD + DMD run at 256px.
# Set MODEL_SIZE in {B,L,H}. This keeps the original table_3_JiT.sh untouched.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
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

: "${LR:=1e-5}"
: "${MIN_LR:=0.0}"
: "${DMD_GUIDANCE_WEIGHT:=0.01}"
: "${DMD_GUIDANCE_LR:=2e-6}"
: "${DMD_GUIDANCE_WEIGHT_DECAY:=0.01}"
: "${DMD_GENERATOR_UPDATE_RATIO:=5}"
: "${DMD_MIN_T:=0.02}"
: "${DMD_MAX_T:=0.98}"
: "${DMD_FAKE_MIN_T:=0.0}"
: "${DMD_FAKE_MAX_T:=1.0}"
: "${DMD_GRAD_CLIP:=0.0}"
: "${DMD_DM_GRAD_MODE:=uncond_real_minus_fake}"
: "${DMD_DISCRIMINATOR_ONLY:=0}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi
DMD_DISCRIMINATOR_ONLY_ARGS=()
DMD_DISCRIMINATOR_ONLY_SUFFIX=
if [ "$DMD_DISCRIMINATOR_ONLY" = "1" ]; then
    DMD_DISCRIMINATOR_ONLY_ARGS=(--dmd_discriminator_only)
    DMD_DISCRIMINATOR_ONLY_SUFFIX="-disc-only"
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

EXP_NAME="${MODEL}-fd-inception-dmd2-${DMD_DM_GRAD_MODE}-w${DMD_GUIDANCE_WEIGHT}-r${DMD_GENERATOR_UPDATE_RATIO}-mode${DMD_DM_GRAD_MODE}${DMD_DISCRIMINATOR_ONLY_SUFFIX}"

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    main_fd.py \
    --project Jit-B-adv \
    --exp_name "$EXP_NAME" \
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
    --lr "$LR" --lr_sched cosine --min_lr "$MIN_LR" \
    --grad_checkpointing \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --fd_repr_models inception \
    --dmd_guidance_weight "$DMD_GUIDANCE_WEIGHT" \
    --dmd_guidance_lr "$DMD_GUIDANCE_LR" \
    --dmd_guidance_weight_decay "$DMD_GUIDANCE_WEIGHT_DECAY" \
    --dmd_generator_update_ratio "$DMD_GENERATOR_UPDATE_RATIO" \
    --dmd_min_t "$DMD_MIN_T" \
    --dmd_max_t "$DMD_MAX_T" \
    --dmd_fake_min_t "$DMD_FAKE_MIN_T" \
    --dmd_fake_max_t "$DMD_FAKE_MAX_T" \
    --dmd_grad_clip "$DMD_GRAD_CLIP" \
    --dmd_real_guidance_scale "$CFG" \
    --dmd_fake_guidance_scale 1.0 \
    --dmd_dm_grad_mode "$DMD_DM_GRAD_MODE" \
    "${DMD_DISCRIMINATOR_ONLY_ARGS[@]}" \
    --dmd_teacher_load_from /mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/checkpoints/base/Jit-B-teacher.pth \
    --auto_resume "$WANDB_FLAG"
