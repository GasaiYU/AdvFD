#!/usr/bin/env bash
# Table 3: multi-node JiT scalability at 256px.
#
# Defaults to 2 nodes x 8 GPUs with a global batch size of 2048
# (128 samples per GPU). Launch this script on every node with the same
# NNODES, MASTER_ADDR, and MASTER_PORT, and a unique NODE_RANK:
#
#   # Node 0 (replace 10.0.0.1 with node 0's reachable IP/hostname):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_JiT_multi.sh
#
#   # Node 1:
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_JiT_multi.sh
#
# Set MODEL_SIZE to one of B, L, or H. All launcher settings can be
# overridden through environment variables.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=2}"
: "${NODE_RANK:?Set NODE_RANK to this node's zero-based rank (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0's reachable IP address or hostname}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=2048}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${FD_REPR_GRAD_CHECKPOINT_MODELS:=none}"
: "${FD_SEQUENTIAL_BACKWARD:=0}"

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

WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

FD_MEMORY_ARGS=()
if [[ -n "$FD_REPR_GRAD_CHECKPOINT_MODELS" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "none" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "0" \
    && "$FD_REPR_GRAD_CHECKPOINT_MODELS" != "off" ]]; then
    read -r -a FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS <<< "$FD_REPR_GRAD_CHECKPOINT_MODELS"
    FD_MEMORY_ARGS+=(--fd_repr_grad_checkpoint_models "${FD_REPR_GRAD_CHECKPOINT_MODEL_ARGS[@]}")
fi
if [ "$FD_SEQUENTIAL_BACKWARD" = "1" ]; then
    FD_MEMORY_ARGS+=(--fd_sequential_backward)
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
        --project table_3_JiT-${MODEL_SIZE} \
        --exp_name "$exp_name" \
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
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --auto_resume "$WANDB_FLAG" \
        "${FD_MEMORY_ARGS[@]}" \
        "$@"
}

# run_one "${MODEL}-fd-sim-high-lr-100x-gbs${GLOBAL_BSZ}" \
#     --fd_repr_models "$SIGLIP" "$MAE" inception \
#     --fd_repr_pool_types cls cls cls \
#     --fd_target_sizes 224 224 256
run_one "${MODEL}-fd-inception-gbs${GLOBAL_BSZ}" \
      --fd_repr_models inception \
      --fd_repr_pool_types cls \
      --fd_target_sizes 256
