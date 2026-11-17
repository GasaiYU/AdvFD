#!/usr/bin/env bash
# W4 rebuttal: JiT-B + SIM-FD with the main-FD Inception UNFROZEN, no AdvFD.
# Reference configuration: scripts/table_3_JiT_adv_fd_sim.sh (default JiT-B).
#
# Full model: L_G = L_SIM-FD + 0.1 * ramp(step) * L_AdvFD  (independent critic).
# This run:   L_G = L_SIM-FD only; no fd_adv_* flag is set, so there is no
#             adversarial branch, critic copy, or real-image forward pass.
#
# The only change from SIM-FD is that the Inception inside SIM-FD is trainable.
# G and Inception minimize the SAME self-normalized FD loss (joint minimization):
# the loss gradient w.r.t. the generated image trains G, and its gradient w.r.t.
# the Inception weights trains Inception. This is NOT a min-max game. SigLIP and
# MAE stay frozen. Reference statistics (mu_ref, sigma_ref) stay those of the
# pretrained Inception, as in SIM-FD. Real images never pass through Inception.
# BatchNorm runs in eval mode (fixed running stats); conv and BN-affine weights
# train. The Inception fc head is unused by the loss and gets no gradient.
#
# Inception optimizer follows the AdvFD critic: AdamW(0.9, 0.999), weight decay 0,
# LR 1e-6, grad-norm clip 1.0. It steps on EVERY G step (AdvFD's critic steps
# every second G step) and is active from step 0 (AdvFD starts at step 1000).
# Everything else matches SIM-FD and the W1/W2 runs: JiT-B checkpoint, global
# batch 1024, 125k G steps, generator LR 1e-5 with cosine schedule, 5-epoch warmup.
#
# Online and final FD evaluation use a SEPARATE pretrained Inception, so the
# reported numbers are not computed by the drifted copy. Checkpoints store the
# trained Inception (key fd_repr_states) so --auto_resume restores it.
#
# Run from the repository root:
#   bash scripts/rebuttal/w4_jit_b_sim_unfrozen_inception.sh
# Preview without loading data/models:
#   DRY_RUN=1 bash scripts/rebuttal/w4_jit_b_sim_unfrozen_inception.sh
# Compare with SIM-FD-only and SIM-FD+AdvFD at the SAME checkpoint step, EMA
# label, sampling seed and FD-r3 evaluation protocol.
# Optional: UNFREEZE_LR=... UNFREEZE_GRAD_CLIP=... to change the Inception optimizer.

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
: "${UNFREEZE_LR:=1e-6}"
: "${UNFREEZE_GRAD_CLIP:=1.0}"
: "${ENABLE_WANDB:=1}"
: "${DRY_RUN:=0}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS < 1 || GLOBAL_BSZ < 1 || GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by NNODES * GPUS_PER_NODE" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
LOAD_FROM="${CKPT_ROOT}/JiT-B.pth"
PROJECT=rebuttal_w4
EXP_NAME="JiT_B-sim-unfrozen-inception-lr${UNFREEZE_LR}-clip${UNFREEZE_GRAD_CLIP}-125k-seed${SEED}"
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
    --fd_repr_unfreeze_models inception
    --fd_repr_unfreeze_lr "$UNFREEZE_LR" --fd_repr_unfreeze_grad_clip "$UNFREEZE_GRAD_CLIP"
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5
    --auto_resume "${WANDB_ARGS[@]}"
)

printf '[W4] Command: '
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
    # Append so separate resumed sessions retain their individual costs.
    printf 'training_wall_seconds=%s total_gpus=%s gpu_seconds=%s\n' \
        "$TRAIN_SECONDS" "$TOTAL_GPUS" "$(( TRAIN_SECONDS * TOTAL_GPUS ))" \
        | tee -a "${RUN_DIR}/runtime.txt"
fi
