#!/usr/bin/env bash
# Table 3: multi-node pMF scalability at 256px and 512px with adversarial FD.
#
# Defaults to 2 nodes x 8 GPUs with a global batch size of 1024
# (64 samples per GPU). Launch this script on every node with the same
# NNODES, MASTER_ADDR, and MASTER_PORT, and a unique NODE_RANK:
#
#   # Node 0 (replace 10.0.0.1 with node 0's reachable IP/hostname):
#   MODEL_SIZE=L RES=256 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_pMF_adv_multi.sh
#
#   # Node 1:
#   MODEL_SIZE=L RES=256 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_pMF_adv_multi.sh
#
# Set MODEL_SIZE in {B,L,H} and RES in {256,512}. All launcher settings can be
# overridden through environment variables.
#
# GLOBAL_BSZ stays at 1024 so the tuned pMF recipe (lr 1e-6, 100x1250 steps,
# 5 warmup epochs) still applies; the extra nodes buy wall-clock, not a larger
# batch. Raise GLOBAL_BSZ only if you also retune the learning rate.
#
# OUTPUT_DIR must be on a shared filesystem, otherwise --auto_resume and
# checkpointing break across nodes.

set -euo pipefail

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=2}"
# Keep these :? messages free of apostrophes: inside "${VAR:?...}" a single quote
# is still active syntax, so a stray one swallows the following lines and turns
# the next guard into dead code.
: "${NODE_RANK:?Set NODE_RANK to the zero-based rank of this node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable IP address or hostname of node 0}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${FD_MAIN_REPRS:=sim}"       # inception | sim
: "${FD_ADV_REPRS:=inception}"  # follow | inception | sim
: "${FD_ADV_WEIGHT:=0.05}" # fix 0.05
: "${FD_ADV_LR:=2e-6}" # fix 2e-6
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_UPDATE_FREQ:=2}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_DETACH_REAL:=1}"
: "${FD_WHITEN:=0}"
: "${FD_WHITEN_EPS:=1e-3}" # fix 1e-3
: "${FD_ADV_START_STEP:=1000}"
: "${FD_ADV_WARMUP_STEPS:=4000}"
: "${FD_ADV_WHITEN_EPS:=1e-3}"
: "${FD_ADV_WHITEN:=1}"
: "${FD_ADV_NEG_REAL_DEGRADE_RATIO:=0}"
: "${FD_ADV_LOG_RAW:=1}"
: "${FD_ADV_LOG_RAW_FREQ:=1000}"
: "${FD_ADV_EMA_BETA:=0.99}"

if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] NNODES must be a positive integer, got: $NNODES" >&2
    exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "[ERR] NODE_RANK must be in [0, $((NNODES - 1))], got: $NODE_RANK" >&2
    exit 1
fi
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] GPUS_PER_NODE must be a positive integer, got: $GPUS_PER_NODE" >&2
    exit 1
fi
if ! [[ "$GLOBAL_BSZ" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] GLOBAL_BSZ must be a positive integer, got: $GLOBAL_BSZ" >&2
    exit 1
fi

TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=$GLOBAL_BSZ must be divisible by total GPUs=$TOTAL_GPUS" >&2
    exit 1
fi
BATCH_SIZE=$((GLOBAL_BSZ / TOTAL_GPUS))

if (( NNODES > 1 )) && [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    echo "[ERR] NNODES=$NNODES but MASTER_ADDR=127.0.0.1; use node 0's routable address" >&2
    exit 1
fi

WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

FD_ADV_LOG_RAW_FLAG=
if [ "$FD_ADV_LOG_RAW" = "1" ]; then
    FD_ADV_LOG_RAW_FLAG=--fd_adv_log_raw
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

FD_ADV_DETACH_REAL_ARGS=()
FD_ADV_DETACH_REAL_SUFFIX=
if [ "$FD_ADV_DETACH_REAL" = "1" ]; then
    FD_ADV_DETACH_REAL_ARGS=(--fd_adv_detach_real)
    FD_ADV_DETACH_REAL_SUFFIX="-detachreal"
fi

FD_ADV_UPDATE_SUFFIX=
if [ "$FD_ADV_UPDATE_FREQ" != "1" ]; then
    FD_ADV_UPDATE_SUFFIX="-advfreq${FD_ADV_UPDATE_FREQ}"
fi

FD_REPR_ARGS=()
case "${FD_MAIN_REPRS}" in
    inception)
        FD_REPR_ARGS=(--fd_repr_models inception)
        FD_MAIN_TAG="fd-inception" ;;
    sim)
        FD_REPR_ARGS=(
            --fd_repr_models "$SIGLIP" "$MAE" inception
            --fd_repr_pool_types cls cls cls
            --fd_target_sizes 224 224 256
        )
        FD_MAIN_TAG="fd-sim" ;;
    *) echo "[ERR] unsupported FD_MAIN_REPRS=${FD_MAIN_REPRS}"; exit 1 ;;
esac

FD_ADV_REPR_ARGS=()
case "${FD_ADV_REPRS}" in
    follow)
        FD_ADV_TAG="-advfollow" ;;
    inception)
        FD_ADV_REPR_ARGS=(--fd_adv_repr_models inception)
        FD_ADV_TAG="-advinc" ;;
    sim)
        FD_ADV_REPR_ARGS=(
            --fd_adv_repr_models "$SIGLIP" "$MAE" inception
            --fd_adv_repr_pool_types cls cls cls
            --fd_adv_target_sizes 224 224 256
        )
        FD_ADV_TAG="-advsim" ;;
    *) echo "[ERR] unsupported FD_ADV_REPRS=${FD_ADV_REPRS}"; exit 1 ;;
esac

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
        LOAD="${CKPT_ROOT}/pMF-B_512.pth"; EXTRA=(--noise_scale 2.0 --img_size 512 --patch_size 32) ;;
    L-512)
        MODEL=pMF_L; CFG=7.5; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-L_512.pth"; EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    H-512)
        MODEL=pMF_H; CFG=5.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-H_512.pth"; EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE} RES=${RES}"; exit 1 ;;
esac

echo "[INFO] node_rank=${NODE_RANK}/${NNODES}, gpus_per_node=${GPUS_PER_NODE}, total_gpus=${TOTAL_GPUS}"
echo "[INFO] global_batch_size=${GLOBAL_BSZ}, batch_size_per_gpu=${BATCH_SIZE}"
echo "[INFO] rendezvous=${MASTER_ADDR}:${MASTER_PORT}"

run_one() {
    local exp_name="$1"
    shift
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project table_3_pMF \
        --exp_name "$exp_name" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --rope_2d --learned_pe --disable_v_head \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 25 --online_eval --eval_freq 1000 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_repr_grad_checkpoint_models siglip \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume "$WANDB_FLAG" \
        "${EXTRA[@]}" \
        "$@"
}

run_one "${MODEL}_${RES}-${FD_MAIN_TAG}${FD_ADV_TAG}-w${FD_ADV_WEIGHT}${FD_ADV_UPDATE_SUFFIX}${FD_ADV_DETACH_REAL_SUFFIX}${FD_WHITEN_SUFFIX}${FD_ADV_WHITEN_SUFFIX}-${FD_ADV_LR}-gbs${GLOBAL_BSZ}-n${NNODES}" \
    "${FD_REPR_ARGS[@]}" \
    "${FD_ADV_REPR_ARGS[@]}" \
    --fd_adv_weight "$FD_ADV_WEIGHT" \
    --fd_adv_backbone repr \
    --fd_adv_lr "$FD_ADV_LR" \
    --fd_adv_steps "$FD_ADV_STEPS" \
    --fd_adv_update_freq "$FD_ADV_UPDATE_FREQ" \
    --fd_adv_grad_clip "$FD_ADV_GRAD_CLIP" \
    "${FD_ADV_DETACH_REAL_ARGS[@]}" \
    --fd_adv_start_step "$FD_ADV_START_STEP" \
    --fd_adv_warmup_steps "$FD_ADV_WARMUP_STEPS" \
    --fd_adv_whiten_eps "$FD_ADV_WHITEN_EPS" \
    "${FD_ADV_WHITEN_ARGS[@]}" \
    --fd_adv_neg_real_degrade_ratio "$FD_ADV_NEG_REAL_DEGRADE_RATIO" \
    $FD_ADV_LOG_RAW_FLAG \
    --fd_adv_log_raw_freq "$FD_ADV_LOG_RAW_FREQ" \
    --fd_adv_ema_beta "$FD_ADV_EMA_BETA" \
    "$@"
