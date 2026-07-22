#!/usr/bin/env bash
# Table 3: iMF scalability at 256px.
# Set MODEL_SIZE in {B,L,XL}.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

case "${MODEL_SIZE}" in
    B)
        MODEL=iMF_B; CFG=8.0; INTERVAL_MIN=0.4; INTERVAL_MAX=0.65
        LOAD="${CKPT_ROOT}/iMF-B.pth"; EXTRA=() ;;
    L)
        MODEL=iMF_L; CFG=10.5; INTERVAL_MIN=0.4; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/iMF-L.pth"; EXTRA=() ;;
    XL)
        MODEL=iMF_XL; CFG=8.0; INTERVAL_MIN=0.42; INTERVAL_MAX=0.62
        LOAD="${CKPT_ROOT}/iMF-XL.pth"; EXTRA=(--fd_queue_fill_bsz 64) ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

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
        --project table_3_iMF \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --tokenizer sdvae --tokenizer_patch_size 8 --patch_size 2 \
        --disable_v_head \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 50 --online_eval --eval_freq 1000 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --compile --auto_resume "$WANDB_FLAG" \
        "${EXTRA[@]}" \
        "$@"
}

run_one "${MODEL}-fd-inception" --fd_repr_models inception
run_one "${MODEL}-fd-sim" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256
