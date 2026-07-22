#!/usr/bin/env bash
# Table 3: JiT scalability at 256px with adversarial FD using InternViT repr.
# Set MODEL_SIZE in {B,L,H}. Compute matching InternViT reference stats first.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/mmu-vcg/gaomingju/data/models/modules}
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
: "${IMG_SIZE:=256}"
: "${INTERNVIT:=/mmu-vcg/gaomingju/data/models/OpenGVLab/InternViT-300M-448px}"
: "${INTERNVIT_TARGET_SIZE:=224}"
: "${INTERNVIT_STATS_PATH:=data/fid_stats/internvit_300m_in${IMG_SIZE}_t${INTERNVIT_TARGET_SIZE}_stats.npz}"
: "${LOAD_INIT:=base}"  # base | fd75k | custom | none
: "${LOAD_FROM:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_JiT/JiT_B-fd-inception/checkpoints/step_0075000.pth}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_LR:=2e-5}" # 2e-6 for full repr finetuning
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_LORA_RANK:=16}"
: "${FD_ADV_LORA_ALPHA:=32}"
: "${FD_ADV_LORA_TARGETS:=attn.qkv}"
: "${FD_ADV_LORA_DROPOUT:=0.0}"
: "${FD_ADV_REPR_GRAD_CHECKPOINTING:=0}"
: "${FD_ADV_DETACH_REAL:=1}"
: "${FD_ADV_REAL_UPDATE_FREQ:=2}"
: "${FD_WHITEN:=0}"
: "${FD_WHITEN_EPS:=1e-3}"
: "${FD_ADV_START_STEP:=1000}"
: "${FD_ADV_WARMUP_STEPS:=4000}"
: "${FD_ADV_WHITEN_EPS:=1e-3}"
: "${FD_ADV_WHITEN:=1}"
: "${FD_ADV_NEG_REAL_DEGRADE_RATIO:=0}"
: "${FD_ADV_LOG_RAW:=1}"
: "${FD_ADV_LOG_RAW_FREQ:=1000}"
: "${FD_ADV_EMA_BETA:=0.99}"
: "${ENABLE_WANDB:=1}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
FD_ADV_LOG_RAW_FLAG=
if [ "$FD_ADV_LOG_RAW" = "1" ]; then
    FD_ADV_LOG_RAW_FLAG=--fd_adv_log_raw
fi
WANDB_FLAG=
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi
FD_WHITEN_ARGS=()
FD_WHITEN_SUFFIX=
if [ "$FD_WHITEN" = "1" ]; then
    FD_WHITEN_ARGS=(--fd_whiten)
    FD_WHITEN_SUFFIX="-fdwhiten-eps${FD_WHITEN_EPS}"
fi
FD_ADV_WHITEN_ARGS=()
FD_ADV_WHITEN_SUFFIX=
if [ "$FD_ADV_WHITEN" = "0" ]; then
    FD_ADV_WHITEN_ARGS=(--fd_adv_no_whiten)
    FD_ADV_WHITEN_SUFFIX="-advnowhiten"
fi
FD_ADV_REPR_GRAD_CHECKPOINT_ARGS=()
FD_ADV_REPR_GRAD_CHECKPOINT_SUFFIX=
if [ "$FD_ADV_REPR_GRAD_CHECKPOINTING" = "1" ]; then
    FD_ADV_REPR_GRAD_CHECKPOINT_ARGS=(--fd_adv_repr_grad_checkpointing)
    FD_ADV_REPR_GRAD_CHECKPOINT_SUFFIX="-advckpt"
fi
FD_ADV_DETACH_REAL_ARGS=()
FD_ADV_DETACH_REAL_SUFFIX=
if [ "$FD_ADV_DETACH_REAL" = "1" ]; then
    FD_ADV_DETACH_REAL_ARGS=(--fd_adv_detach_real)
    FD_ADV_DETACH_REAL_SUFFIX="-detachreal"
fi
FD_ADV_REAL_UPDATE_SUFFIX=
if [ "$FD_ADV_REAL_UPDATE_FREQ" != "1" ]; then
    FD_ADV_REAL_UPDATE_SUFFIX="-realfreq${FD_ADV_REAL_UPDATE_FREQ}"
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        case "${LOAD_INIT}" in
            base) LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
            fd75k) LOAD="$LOAD_FROM" ;;
            custom) LOAD="$LOAD_FROM" ;;
            none) LOAD="" ;;
            *) echo "[ERR] unsupported LOAD_INIT=${LOAD_INIT}"; exit 1 ;;
        esac ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

if [ ! -f "$INTERNVIT_STATS_PATH" ]; then
    echo "[ERR] Missing InternViT FD stats: $INTERNVIT_STATS_PATH" >&2
    echo "      Compute CLS reference stats for INTERNVIT_TARGET_SIZE=${INTERNVIT_TARGET_SIZE}, or set INTERNVIT_STATS_PATH." >&2
    exit 1
fi

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
        --project InternViT-adv \
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
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume $WANDB_FLAG \
        "$@"
}

run_one "${MODEL}-internvit-300m-adv-lora-r${FD_ADV_LORA_RANK}${FD_ADV_REPR_GRAD_CHECKPOINT_SUFFIX}${FD_ADV_DETACH_REAL_SUFFIX}${FD_ADV_REAL_UPDATE_SUFFIX}${FD_WHITEN_SUFFIX}${FD_ADV_WHITEN_SUFFIX}-from-${LOAD_INIT}" \
    --fd_repr_models "$INTERNVIT" \
    --fd_repr_stats_paths "$INTERNVIT_STATS_PATH" \
    --fd_repr_pool_types cls \
    --fd_target_sizes "$INTERNVIT_TARGET_SIZE" \
    --fd_adv_weight "$FD_ADV_WEIGHT" \
    --fd_adv_backbone repr \
    --fd_adv_lora_rank "$FD_ADV_LORA_RANK" \
    --fd_adv_lora_alpha "$FD_ADV_LORA_ALPHA" \
    --fd_adv_lora_targets $FD_ADV_LORA_TARGETS \
    --fd_adv_lora_dropout "$FD_ADV_LORA_DROPOUT" \
    "${FD_ADV_REPR_GRAD_CHECKPOINT_ARGS[@]}" \
    "${FD_ADV_DETACH_REAL_ARGS[@]}" \
    --fd_adv_real_update_freq "$FD_ADV_REAL_UPDATE_FREQ" \
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
    "$@"
