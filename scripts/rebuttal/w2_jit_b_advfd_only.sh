#!/usr/bin/env bash
# W2 rebuttal: JiT-B AdvFD-only, removing the frozen SIM-FD loss.
# Reference configuration: scripts/table_3_JiT_adv_fd_sim.sh (default JiT-B).
#
# Full model: L_G = L_SIM-FD + 0.1 * ramp(step) * L_AdvFD.
# This run:   L_G =          0.1 * ramp(step) * L_AdvFD.
# Set all THREE static loss weights to zero and explicitly keep the learned
# Inception representation weight at 1. Do not set fd_adv_weight to zero.
# The independent learned Inception still trains with the original FD/whitening
# D-step; no GAN, DMD, or denoising loss is added.
#
# Keep frozen feature models/queues for the SAME adversarial-stat initialization
# and monitoring as the full model. They supply ZERO static-loss gradient.
# This is a loss ablation, not an implementation that removes their compute.
#
# Keep the original generator checkpoint, 125k-step schedule, global batch 1024,
# generator LR 1e-5, AdvFD weight 0.1, critic LR 1e-6, one D-step every two G
# steps, real whitening eps=1e-3, real detach and adversarial EMA beta=0.99.
# Keep auxiliary start=1000 and warmup=4000: before step 1000 there is no
# effective generator loss; at step 1000 the ramp is zero, then increases.
# Do not introduce a SIM-FD-only warm-start or resume a combined-model run.
#
# Run from the repository root:
#   bash scripts/rebuttal/w2_jit_b_advfd_only.sh
# Preview without loading data/models:
#   DRY_RUN=1 bash scripts/rebuttal/w2_jit_b_advfd_only.sh
# Compare the final result with SIM-FD-only and SIM-FD+AdvFD using the SAME
# checkpoint step, EMA label, sampling seed and FD-r3 evaluation protocol.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${SEED:=1}"
: "${ENABLE_WANDB:=1}"
: "${DRY_RUN:=0}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS < 1 || GLOBAL_BSZ < 1 || GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by NNODES * GPUS_PER_NODE" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
LOAD_FROM="${CKPT_ROOT}/JiT-B.pth"
PROJECT=rebuttal_w2
EXP_NAME="JiT_B-advfd-only-static0-w0.1-start1k-warmup4k-125k-seed${SEED}"
RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
WANDB_ARGS=(--disable_wandb)
if [[ "$ENABLE_WANDB" == "1" ]]; then
    WANDB_ARGS=(--enable_wandb)
fi

CMD=(
    torchrun
    --nnodes="$NNODES" --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    --nproc_per_node="$GPUS_PER_NODE"
    main_fd.py
    --project "$PROJECT" --exp_name "$EXP_NAME" --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "$LOAD_FROM"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models siglip
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception
    --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256
    --fd_repr_weights 0 0 0
    --fd_adv_repr_models inception --fd_adv_repr_weights 1
    --fd_adv_weight 0.1 --fd_adv_backbone repr --fd_adv_lr 1e-6
    --fd_adv_steps 1 --fd_adv_update_freq 2 --fd_adv_grad_clip 1.0
    --fd_adv_detach_real --fd_adv_start_step 1000 --fd_adv_warmup_steps 4000
    --fd_adv_whiten_eps 1e-3 --fd_adv_neg_real_degrade_ratio 0
    --fd_adv_log_raw --fd_adv_log_raw_freq 1000 --fd_adv_ema_beta 0.99
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5
    --auto_resume "${WANDB_ARGS[@]}"
)

printf '[W2] Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi
if [[ ! -f "$LOAD_FROM" ]]; then
    echo "[ERR] JiT-B checkpoint not found: $LOAD_FROM" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/train" ]]; then
    echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
TRAIN_START=$SECONDS
"${CMD[@]}"
TRAIN_SECONDS=$(( SECONDS - TRAIN_START ))
if [[ "$NODE_RANK" == "0" ]]; then
    printf 'training_wall_seconds=%s total_gpus=%s gpu_seconds=%s\n' \
        "$TRAIN_SECONDS" "$TOTAL_GPUS" "$(( TRAIN_SECONDS * TOTAL_GPUS ))" \
        | tee -a "${RUN_DIR}/runtime.txt"
fi
