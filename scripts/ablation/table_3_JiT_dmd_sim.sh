#!/usr/bin/env bash
# Table 3 JiT FD-SIM + DMD run at 256px.
# SIM = SigLIP + MAE + Inception. Run this script from the repository root.

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

set -euo pipefail

: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${PROJECT:=Jit-adv-ablation}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"

: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${FD_REPR_GRAD_CHECKPOINT_MODELS:=siglip}"

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
: "${DMD_DM_GRAD_MODE:=original}"
: "${DMD_DISCRIMINATOR_ONLY:=0}"

if [[ ! -f main_fd.py ]]; then
    echo "[ERR] run this script from the FD-Loss-Ours repository root" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/train" || ! -d "${DATA_ROOT}/val" ]]; then
    echo "[ERR] DATA_ROOT=${DATA_ROOT} must contain train/ and val/" >&2
    exit 1
fi

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS < 1 )); then
    echo "[ERR] NNODES * GPUS_PER_NODE must be at least 1" >&2
    exit 1
fi
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

WANDB_ARGS=(--disable_wandb)
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    WANDB_ARGS=(--enable_wandb)
fi

DMD_DISCRIMINATOR_ONLY_ARGS=()
DMD_DISCRIMINATOR_ONLY_SUFFIX=
if [[ "${DMD_DISCRIMINATOR_ONLY}" == "1" ]]; then
    DMD_DISCRIMINATOR_ONLY_ARGS=(--dmd_discriminator_only)
    DMD_DISCRIMINATOR_ONLY_SUFFIX="-disc-only"
fi

FD_REPR_GRAD_CHECKPOINT_ARGS=()
if [[ -n "${FD_REPR_GRAD_CHECKPOINT_MODELS}" \
    && "${FD_REPR_GRAD_CHECKPOINT_MODELS}" != "none" \
    && "${FD_REPR_GRAD_CHECKPOINT_MODELS}" != "0" \
    && "${FD_REPR_GRAD_CHECKPOINT_MODELS}" != "off" ]]; then
    read -r -a FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS <<< "${FD_REPR_GRAD_CHECKPOINT_MODELS}"
    FD_REPR_GRAD_CHECKPOINT_ARGS=(
        --fd_repr_grad_checkpoint_models
        "${FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS[@]}"
    )
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-B.pth"
        DEFAULT_DMD_TEACHER="${CKPT_ROOT}/Jit-B-teacher.pth" ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth"
        DEFAULT_DMD_TEACHER="${CKPT_ROOT}/Jit-L-teacher.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth"
        DEFAULT_DMD_TEACHER="${CKPT_ROOT}/Jit-H-teacher.pth" ;;
    *)
        echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}; expected B, L, or H" >&2
        exit 1 ;;
esac

: "${DMD_TEACHER_LOAD_FROM:=${DEFAULT_DMD_TEACHER}}"

if [[ ! -f "${LOAD}" ]]; then
    echo "[ERR] base checkpoint not found: ${LOAD}" >&2
    exit 1
fi
if [[ ! -f "${DMD_TEACHER_LOAD_FROM}" ]]; then
    echo "[ERR] ${MODEL_SIZE}-size DMD teacher not found: ${DMD_TEACHER_LOAD_FROM}" >&2
    echo "      Set DMD_TEACHER_LOAD_FROM to a teacher matching MODEL_SIZE=${MODEL_SIZE}." >&2
    exit 1
fi

EXP_NAME="${MODEL}-fd-sim-dmd2-${DMD_DM_GRAD_MODE}-w${DMD_GUIDANCE_WEIGHT}-r${DMD_GENERATOR_UPDATE_RATIO}"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    main_fd.py \
    --project "${PROJECT}" \
    --exp_name "${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --data_path "${DATA_ROOT}" \
    --load_from "${LOAD}" \
    --model "${MODEL}" --rope_2d --learned_pe --legacy_time_convention \
    --cfg "${CFG}" --interval_min "${INTERVAL_MIN}" --interval_max "${INTERVAL_MAX}" \
    --ema_type edm \
    --num_sampling_steps 1 \
    --eval_bsz 256 --num_images_for_eval_and_search 50000 \
    --vis_freq 100 --online_eval --eval_freq 10000 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
    --lr "${LR}" --lr_sched cosine --min_lr "${MIN_LR}" \
    --grad_checkpointing \
    "${FD_REPR_GRAD_CHECKPOINT_ARGS[@]}" \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --fd_repr_models "${SIGLIP}" "${MAE}" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --dmd_guidance_weight "${DMD_GUIDANCE_WEIGHT}" \
    --dmd_guidance_lr "${DMD_GUIDANCE_LR}" \
    --dmd_guidance_weight_decay "${DMD_GUIDANCE_WEIGHT_DECAY}" \
    --dmd_generator_update_ratio "${DMD_GENERATOR_UPDATE_RATIO}" \
    --dmd_min_t "${DMD_MIN_T}" \
    --dmd_max_t "${DMD_MAX_T}" \
    --dmd_fake_min_t "${DMD_FAKE_MIN_T}" \
    --dmd_fake_max_t "${DMD_FAKE_MAX_T}" \
    --dmd_grad_clip "${DMD_GRAD_CLIP}" \
    --dmd_real_guidance_scale "${CFG}" \
    --dmd_fake_guidance_scale 1.0 \
    --dmd_dm_grad_mode "${DMD_DM_GRAD_MODE}" \
    "${DMD_DISCRIMINATOR_ONLY_ARGS[@]}" \
    --dmd_teacher_load_from "${DMD_TEACHER_LOAD_FROM}" \
    --auto_resume \
    "${WANDB_ARGS[@]}"
