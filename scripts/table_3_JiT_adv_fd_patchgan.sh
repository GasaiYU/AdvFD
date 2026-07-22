#!/usr/bin/env bash
# Table 3: JiT scalability at 256px with adversarial FD.
# Set MODEL_SIZE in {B,L,H}.

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
: "${MODEL_SIZE:=B}"
: "${PATCH_GAN_INIT:=base}"  # base | fd75k | custom | none
: "${PATCH_GAN_LOAD_FROM:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_JiT/JiT_B-fd-inception/checkpoints/step_0075000.pth}"
: "${FD_ADV_WEIGHT:=0.05}" # Need Fixing 0.2
: "${FD_ADV_BACKBONE:=patchgan}" # Need Fixing
: "${FD_ADV_LR:=1e-5}" # Need fixing 2e-6
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_START_STEP:=3000}" # Need Fixing
: "${FD_ADV_WARMUP_STEPS:=4000}" 
: "${FD_ADV_WHITEN_EPS:=1e-3}" # Need fixing
: "${FD_ADV_WHITEN:=1}"
: "${FD_ADV_NEG_REAL_DEGRADE_RATIO:=0.25}" # Need Fixing 0 
: "${FD_ADV_LOG_RAW:=1}"
: "${FD_ADV_LOG_RAW_FREQ:=1000}"
: "${FD_ADV_EMA_BETA:=0.99}"
: "${FD_ADV_PATCH_CHANNELS:=64}"
: "${FD_ADV_PATCH_MAX_CHANNELS:=256}"
: "${FD_ADV_PATCH_LAYERS:=4}" # Need fixing 5
: "${FD_ADV_PATCH_MAX_PATCHES:=16384}" # Need fixing 32768
: "${FD_ADV_PATCH_TRAIN_LOSS:=fd}"  # fd | hinge 
: "${FD_ADV_PATCH_PRETRAIN_START_STEP:=1000}"
: "${FD_ADV_PATCH_PRETRAIN_STEPS:=2000}"
: "${FD_ADV_PATCH_PRETRAIN_LR:=1e-5}"
: "${FD_ADV_PATCH_PRETRAIN_LOG_FREQ:=100}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
FD_ADV_LOG_RAW_FLAG=
if [ "$FD_ADV_LOG_RAW" = "1" ]; then
    FD_ADV_LOG_RAW_FLAG=--fd_adv_log_raw
fi
FD_ADV_WHITEN_ARGS=()
FD_ADV_WHITEN_SUFFIX=
if [ "$FD_ADV_WHITEN" = "0" ]; then
    FD_ADV_WHITEN_ARGS=(--fd_adv_no_whiten)
    FD_ADV_WHITEN_SUFFIX="-advnowhiten"
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        case "${PATCH_GAN_INIT}" in
            base) LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
            fd75k) LOAD="$PATCH_GAN_LOAD_FROM" ;;
            custom) LOAD="$PATCH_GAN_LOAD_FROM" ;;
            none) LOAD="" ;;
            *) echo "[ERR] unsupported PATCH_GAN_INIT=${PATCH_GAN_INIT}"; exit 1 ;;
        esac ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

run_one() {
    local exp_name="$1"
    shift
    local load_args=()
    if [ -n "$LOAD" ]; then
        load_args=(--load_from "$LOAD")
    fi
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project Jit-B-adv \
        --exp_name "$exp_name" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        "${load_args[@]}" \
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
        --auto_resume --enable_wandb \
        "$@"
}

run_one "${MODEL}-fd-inception-advfd-${FD_ADV_BACKBONE}-w${FD_ADV_WEIGHT}-gclip${FD_ADV_GRAD_CLIP}-negreal${FD_ADV_NEG_REAL_DEGRADE_RATIO}${FD_ADV_WHITEN_SUFFIX}-pre${FD_ADV_PATCH_PRETRAIN_START_STEP}x${FD_ADV_PATCH_PRETRAIN_STEPS}-adv${FD_ADV_START_STEP}-from-${PATCH_GAN_INIT}" \
    --fd_repr_models inception \
    --fd_adv_weight "$FD_ADV_WEIGHT" \
    --fd_adv_backbone "$FD_ADV_BACKBONE" \
    --fd_adv_lr "$FD_ADV_LR" \
    --fd_adv_steps "$FD_ADV_STEPS" \
    --fd_adv_grad_clip "$FD_ADV_GRAD_CLIP" \
    --fd_adv_start_step "$FD_ADV_START_STEP" \
    --fd_adv_warmup_steps "$FD_ADV_WARMUP_STEPS" \
    --fd_adv_whiten_eps "$FD_ADV_WHITEN_EPS" \
    "${FD_ADV_WHITEN_ARGS[@]}" \
    --fd_adv_neg_real_degrade_ratio "$FD_ADV_NEG_REAL_DEGRADE_RATIO" \
    $FD_ADV_LOG_RAW_FLAG \
    --fd_adv_log_raw_freq "$FD_ADV_LOG_RAW_FREQ" \
    --fd_adv_ema_beta "$FD_ADV_EMA_BETA" \
    --fd_adv_patch_channels "$FD_ADV_PATCH_CHANNELS" \
    --fd_adv_patch_max_channels "$FD_ADV_PATCH_MAX_CHANNELS" \
    --fd_adv_patch_layers "$FD_ADV_PATCH_LAYERS" \
    --fd_adv_patch_max_patches "$FD_ADV_PATCH_MAX_PATCHES" \
    --fd_adv_patch_train_loss "$FD_ADV_PATCH_TRAIN_LOSS" \
    --fd_adv_patch_pretrain_start_step "$FD_ADV_PATCH_PRETRAIN_START_STEP" \
    --fd_adv_patch_pretrain_steps "$FD_ADV_PATCH_PRETRAIN_STEPS" \
    --fd_adv_patch_pretrain_lr "$FD_ADV_PATCH_PRETRAIN_LR" \
    --fd_adv_patch_pretrain_log_freq "$FD_ADV_PATCH_PRETRAIN_LOG_FREQ" \
    "$@"
