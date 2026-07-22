#!/usr/bin/env bash
# Visualize a released JiT teacher checkpoint.
#
# Example:
#   CKPT_ROOT=/path/to/checkpoints/base bash scripts/visualize_jit_teacher.sh
#   TEACHER_LOAD=/path/to/Jit-L-teacher.pth VIS_STEPS="1 50" bash scripts/visualize_jit_teacher.sh

set -euo pipefail

: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=1}"
: "${MODEL_SIZE:=B}"
: "${ENABLE_WANDB:=0}"
: "${SEED:=1}"

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B
        DEFAULT_CFG=3.0
        DEFAULT_TEACHER="${CKPT_ROOT}/Jit-B-teacher.pth"
        ;;
    L)
        MODEL=JiT_L
        DEFAULT_CFG=2.4
        DEFAULT_TEACHER="${CKPT_ROOT}/Jit-L-teacher.pth"
        ;;
    H)
        MODEL=JiT_H
        DEFAULT_CFG=2.2
        DEFAULT_TEACHER="${CKPT_ROOT}/Jit-H-teacher.pth"
        ;;
    *)
        echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}; expected B, L, or H"
        exit 1
        ;;
esac

: "${TEACHER_LOAD:=${DEFAULT_TEACHER}}"
: "${CFG:=${DEFAULT_CFG}}"
: "${INTERVAL_MIN:=0.1}"
: "${INTERVAL_MAX:=1.0}"
: "${NUM_SAMPLING_STEPS:=50}"
: "${VIS_STEPS:=${NUM_SAMPLING_STEPS}}"
: "${EXP_NAME:=${MODEL}-teacher-$(basename "${TEACHER_LOAD}" .pth)-cfg${CFG}-steps${VIS_STEPS// /_}}"

WANDB_FLAG=--disable_wandb
if [ "${ENABLE_WANDB}" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

echo "[teacher-vis] model=${MODEL}"
echo "[teacher-vis] checkpoint=${TEACHER_LOAD}"
echo "[teacher-vis] cfg=${CFG}, interval=[${INTERVAL_MIN}, ${INTERVAL_MAX}], vis_steps=${VIS_STEPS}"
echo "[teacher-vis] output=${OUTPUT_DIR}/teacher_vis/${EXP_NAME}/visualization"

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    main_fd.py \
    --project teacher_vis \
    --exp_name "${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 8 \
    --load_from "${TEACHER_LOAD}" \
    --model "${MODEL}" --rope_2d --learned_pe --legacy_time_convention \
    --cfg "${CFG}" --interval_min "${INTERVAL_MIN}" --interval_max "${INTERVAL_MAX}" \
    --ema_type edm \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --vis_steps ${VIS_STEPS} \
    --eval_ema_labels online \
    --same_noise \
    --vis_only \
    --seed "${SEED}" \
    "${WANDB_FLAG}" \
    "$@"
