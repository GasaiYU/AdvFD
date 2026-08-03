#!/usr/bin/env bash
# Table 3: multi-node JiT scalability at 256px with SIM main FD and Inception FD-Adv.
#
# Defaults to 2 nodes x 8 GPUs with a global batch size of 2048
# (128 samples per GPU, matching the single-node per-GPU batch). Launch this
# script on every node with the same NNODES, MASTER_ADDR, and MASTER_PORT, and a
# unique NODE_RANK:
#
#   # Node 0 (replace 10.0.0.1 with node 0's reachable IP/hostname):
#   MODEL_SIZE=L NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_JiT_adv_fd_sim_multi.sh
#
#   # Node 1:
#   MODEL_SIZE=L NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_JiT_adv_fd_sim_multi.sh
#
# Set MODEL_SIZE to one of B, L, or H. All launcher settings can be overridden
# through environment variables.
#
# GLOBAL_BSZ scales with node count to keep 128 samples per GPU, matching
# scripts/table_3_JiT_multi.sh. That is what makes extra nodes reduce wall-clock:
# the FD loss all-gathers features every step, so shrinking the per-GPU batch
# instead would raise the comms-to-compute ratio and eat the speedup. The
# tradeoff is that global batch 2048 doubles the single-node batch, so the 1e-5
# learning rate may need retuning. To reproduce the single-node recipe exactly,
# set GLOBAL_BSZ=1024 and accept the weaker scaling.
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
: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${LOAD_INIT:=base}"  # base | fd75k | custom | none
: "${LOAD_FROM:=}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_LR:=2e-6}"
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_UPDATE_FREQ:=2}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_DETACH_REAL:=1}"
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

echo "[INFO] node_rank=${NODE_RANK}/${NNODES}, gpus_per_node=${GPUS_PER_NODE}, total_gpus=${TOTAL_GPUS}"
echo "[INFO] global_batch_size=${GLOBAL_BSZ}, batch_size_per_gpu=${BATCH_SIZE}"
echo "[INFO] rendezvous=${MASTER_ADDR}:${MASTER_PORT}"

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
        --project "Jit-${MODEL_SIZE}-adv" \
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
        --fd_repr_grad_checkpoint_models siglip \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume "$WANDB_FLAG" \
        "$@"
}

run_one "${MODEL}-fd-sim-advinc-w${FD_ADV_WEIGHT}-from-${LOAD_INIT}-gbs${GLOBAL_BSZ}-n${NNODES}" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --fd_adv_repr_models inception \
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
