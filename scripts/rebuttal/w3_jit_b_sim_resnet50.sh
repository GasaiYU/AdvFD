#!/usr/bin/env bash
# Extra frozen encoder controls: JiT-B + SIM-FD + ResNet-50 [+ Swin-T] FD.
# ImageNet-1K weights: resnet50.a1_in1k (224 input, 2048-d GAP features),
# swin_tiny_patch4_window7_224.ms_in1k (224 input, 768-d spatial mean features).
# All four/five FD encoders stay frozen/eval. G gradients flow THROUGH them.
# L_G = L_SIM-FD + EXTRA_FD_WEIGHT * normalized_FD_ResNet50
#       [+ SWIN_FD_WEIGHT * normalized_FD_SwinT]; both extra weights default=1.
# There is no learned adversarial branch in either frozen-encoder experiment.
#
# Default MODE=overhead: measure SIM, SIM+ResNet50, SIM+ResNet50+SwinT, SIM+AdvFD
# with the actual loop, identical GPUs/batch/checkpoint, 50 warmup + 200 steps.
# Start all short runs at virtual step 5000 so AdvFD's 1k/4k ramp is complete.
# These are cost measurements, not 5000-step-trained generator checkpoints.
# Include input backward, FD covariance/eigensolves, and AdvFD D/G work; exclude
# initialization, reference-stat preparation, checkpoint I/O and evaluation.
# AdvFD uses its original 1 D-step every 2 G-steps and global weight 0.1.
# Report incremental latency and peak ALLOCATED memory relative to SIM.
#
# MODE=train: run 125k G steps. EXTRA_ENCODERS=resnet50 (default) or resnet50_swin_t.
# MODE=stats: prepare both new encoders' ImageNet train reference statistics.
# Missing required statistics are prepared automatically before overhead/train.
# Neither new encoder belongs to the current six evaluation encoders. Keep
# FD-r3's original held-out encoders; do not add ResNet/Swin to that score.
#
# MODE=overhead bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh
# MODE=train    bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh
# EXTRA_ENCODERS=resnet50_swin_t MODE=train bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh
# DRY_RUN=1    bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${MODE:=overhead}"
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
: "${EXTRA_ENCODERS:=resnet50}"
: "${EXTRA_FD_WEIGHT:=1.0}"
: "${SWIN_FD_WEIGHT:=1.0}"
: "${PROFILE_WARMUP_STEPS:=50}"
: "${PROFILE_STEPS:=200}"
: "${ENABLE_WANDB:=1}"
: "${DRY_RUN:=0}"

case "$MODE" in overhead|train|stats) ;; *) echo "[ERR] MODE must be overhead, train, or stats" >&2; exit 1 ;; esac
case "$EXTRA_ENCODERS" in resnet50|resnet50_swin_t) ;; *) echo "[ERR] EXTRA_ENCODERS must be resnet50 or resnet50_swin_t" >&2; exit 1 ;; esac
TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS < 1 || GLOBAL_BSZ < 1 || GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by NNODES * GPUS_PER_NODE" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
RESNET=resnet50.a1_in1k
RESNET_STATS=data/fid_stats/resnet50_a1_in1k_in256_t224_stats.npz
SWIN=swin_tiny_patch4_window7_224.ms_in1k
SWIN_STATS=data/fid_stats/swin_tiny_patch4_window7_224_ms_in1k_in256_t224_stats.npz
PROJECT=rebuttal_w3
PROFILE_DIR="${OUTPUT_DIR}/${PROJECT}/overhead-seed${SEED}-gbs${GLOBAL_BSZ}-gpus${TOTAL_GPUS}-extra${EXTRA_FD_WEIGHT}-swin${SWIN_FD_WEIGHT}"
LAUNCH=(torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --nproc_per_node="$GPUS_PER_NODE")

run_command() {
    printf '[W3] Command: '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -d "${DATA_ROOT}/train" ]]; then
        echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
    fi
    if [[ "$MODE" != "stats" && ! -f "${CKPT_ROOT}/JiT-B.pth" ]]; then
        echo "[ERR] Missing ${CKPT_ROOT}/JiT-B.pth" >&2; exit 1
    fi
fi

if [[ ! -f "$RESNET_STATS" ]]; then
    run_command "${LAUNCH[@]}" compute_repr_stats.py --model "$RESNET" \
        --data_path "$DATA_ROOT" --img_size 256 --target_size 224 --batch_size 128
fi
if [[ "$MODE" != "train" || "$EXTRA_ENCODERS" == "resnet50_swin_t" ]]; then
    if [[ ! -f "$SWIN_STATS" ]]; then
        run_command "${LAUNCH[@]}" compute_repr_stats.py --model "$SWIN" \
            --data_path "$DATA_ROOT" --img_size 256 --target_size 224 --batch_size 128
    fi
fi
if [[ "$MODE" == "stats" ]]; then exit 0; fi

COMMON=(--project "$PROJECT" --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "${CKPT_ROOT}/JiT-B.pth"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models siglip
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5)
SIM=(--fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception
    --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256)
SIM_RESNET=(--fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception "$RESNET"
    --fd_repr_pool_types cls cls cls cls --fd_target_sizes 224 224 256 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT")
SIM_RESNET_SWIN=(--fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception "$RESNET" "$SWIN"
    --fd_repr_pool_types cls cls cls cls cls --fd_target_sizes 224 224 256 224 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT" "$SWIN_FD_WEIGHT")
ADV=(--fd_adv_repr_models inception --fd_adv_weight 0.1 --fd_adv_backbone repr
    --fd_adv_lr 1e-6 --fd_adv_steps 1 --fd_adv_update_freq 2 --fd_adv_grad_clip 1.0
    --fd_adv_detach_real --fd_adv_start_step 1000 --fd_adv_warmup_steps 4000
    --fd_adv_whiten_eps 1e-3 --fd_adv_neg_real_degrade_ratio 0
    --fd_adv_log_raw --fd_adv_log_raw_freq 1000 --fd_adv_ema_beta 0.99)

if [[ "$MODE" == "overhead" ]]; then
    for CASE in sim sim_resnet50 sim_resnet50_swin_t sim_advfd; do
        PROFILE=(--current_step 5000 --profile_warmup_steps "$PROFILE_WARMUP_STEPS"
            --profile_steps "$PROFILE_STEPS" --profile_output "${PROFILE_DIR}/${CASE}.json")
        case "$CASE" in
            sim) run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
                "${COMMON[@]}" "${SIM[@]}" "${PROFILE[@]}" --exp_name overhead-sim ;;
            sim_resnet50) run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
                "${COMMON[@]}" "${SIM_RESNET[@]}" "${PROFILE[@]}" --exp_name overhead-sim-resnet50 ;;
            sim_resnet50_swin_t) run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
                "${COMMON[@]}" "${SIM_RESNET_SWIN[@]}" "${PROFILE[@]}" --exp_name overhead-sim-resnet50-swin-t ;;
            sim_advfd) run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
                "${COMMON[@]}" "${SIM[@]}" "${ADV[@]}" "${PROFILE[@]}" --exp_name overhead-sim-advfd ;;
        esac
    done
    if [[ "$NODE_RANK" == "0" ]]; then
        run_command python scripts/rebuttal/profile_fd_overhead.py --summarize "$PROFILE_DIR"
    fi
else
    TRAIN_REPR=("${SIM_RESNET[@]}")
    EXP_NAME="JiT_B-sim-frozen-resnet50-w${EXTRA_FD_WEIGHT}-125k-seed${SEED}"
    if [[ "$EXTRA_ENCODERS" == "resnet50_swin_t" ]]; then
        TRAIN_REPR=("${SIM_RESNET_SWIN[@]}")
        EXP_NAME="JiT_B-sim-frozen-resnet50-swin-t-w${EXTRA_FD_WEIGHT}-${SWIN_FD_WEIGHT}-125k-seed${SEED}"
    fi
    WANDB_ARGS=(--disable_wandb)
    if [[ "$ENABLE_WANDB" == "1" ]]; then WANDB_ARGS=(--enable_wandb); fi
    run_command "${LAUNCH[@]}" main_fd.py "${COMMON[@]}" "${TRAIN_REPR[@]}" \
        --exp_name "$EXP_NAME" \
        --auto_resume "${WANDB_ARGS[@]}"
fi
