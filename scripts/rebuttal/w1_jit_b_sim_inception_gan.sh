#!/usr/bin/env bash
# W1 rebuttal: one JiT-B + SIM-FD + Inception-GAN experiment, no sweep.
#
# Replace the scratch PatchGAN discriminator with the SAME pretrained Inception
# feature trunk used by AdvFD, followed by a new Linear(2048, 1) scalar head.
# Both trunk and head train; BN stays in train mode, as in the old PatchGAN.
# The main SIM-FD feature networks remain frozen and use separate weights.
#
# D loss = 0.5 * [mean(relu(1-D(real))) + mean(relu(1+D(fake.detach())))].
# G loss = SIM-FD + 0.1 * ramp * adaptive_weight * [-mean(D(fake))].
# Align all training settings with scripts/ablation/table_3_JiT_patch_gan.sh:
# same JiT-B checkpoint, global batch 1024, 125k G steps, auxiliary start 1k,
# ramp 9k, one D update every G step, LR 1e-6 for BOTH Inception and head,
# AdamW (0.5, 0.9), WD 0, no D gradient clipping, adaptive loss weighting ON.
# Fake images are clamped to [0, 1], exactly as in the original GAN branch.
# Only the discriminator architecture/pretrained initialization changes.
#
# Run from any directory:
#   DATA_ROOT=/path/to/ImageNet CKPT_ROOT=/path/to/base \
#     bash scripts/rebuttal/w1_jit_b_sim_inception_gan.sh
# Preview the command without loading data/models: DRY_RUN=1 bash <this script>
#
# Evaluate the final checkpoint with the SAME EMA/seed/FD-r3 protocol as AdvFD.
# Visualization/online-eval settings also match the old script. runtime.txt
# includes model/queue initialization and the full training run. GPU-hours =
# training_wall_seconds * total_gpus / 3600. There is no weight search.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
PROJECT=rebuttal_w1
EXP_NAME="JiT_B-sim-inception-gan-patchgan-aligned-w0.1-lr1e-6-warmup9k-125k-seed${SEED}"
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
    --model JiT_B --img_size 256 --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models siglip mae
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception
    --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256
    --fd_eigvalsh --fd_ema_beta 0.999
    --patch_gan_backbone inception --patch_gan_weight 0.1
    --patch_gan_lr 1e-6 --patch_gan_weight_decay 0
    --patch_gan_start_step 1000 --patch_gan_warmup_steps 9000
    --patch_gan_grad_checkpointing --patch_gan_adaptive_weight
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5
    --auto_resume "${WANDB_ARGS[@]}"
)

printf '[W1] Command: '
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
