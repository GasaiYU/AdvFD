#!/usr/bin/env bash
# W3 large control: JiT-B + SIM-FD + N extra frozen ImageNet encoders, 16 GPUs.
# Two machines, eight GPUs each. Global batch=1024 => batch=64/GPU.
# Every representation model stays frozen/eval; gradients through their
# generated-image inputs train JiT-B only. No AdvFD branch.
# Each FD loss is self-normalized (FD / stopgrad(FD)) and weighted 1.0, so the
# generator gradient grows with the number of encoders; that is part of what
# this control tests and is the same for the ResNet-50/101 runs.
# Same base checkpoint, LR, 125k steps and warmup as the other W3 runs.
#
# EXTRA_ENCODERS="timm_name ..." picks the extra encoders (default below).
# Never add DINOv2/CLIP/ConvNeXt or their relatives: they are the held-out
# FDr3 evaluation encoders. The script refuses them.
#
# 1) One machine, once: check names, cost and memory (no training):
#      CUDA_VISIBLE_DEVICES=0 python scripts/rebuttal/probe_frozen_encoders.py --models $EXTRA_ENCODERS
# 2) One machine, once: reference statistics for every extra encoder:
#      STAGE=stats bash scripts/rebuttal/w3_jit_b_sim_many_frozen_2nodes.sh
# 3) Both machines: train.
#      NODE_RANK=0 bash scripts/rebuttal/w3_jit_b_sim_many_frozen_2nodes.sh   # on MASTER_ADDR
#      NODE_RANK=1 bash scripts/rebuttal/w3_jit_b_sim_many_frozen_2nodes.sh
# Preview: DRY_RUN=1 NODE_RANK=0 bash scripts/rebuttal/w3_jit_b_sim_many_frozen_2nodes.sh

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${STAGE:=train}"
: "${EXTRA_ENCODERS:=resnet50.a1_in1k resnet101.a1_in1k resnet152.a1_in1k swin_tiny_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in1k deit3_base_patch16_224.fb_in1k}"
: "${EXTRA_FD_WEIGHT:=1.0}"
# Gradient checkpointing for repr models whose name contains one of these
# substrings (SigLIP as in the other W3 runs, plus the large extra encoders).
: "${GRAD_CKPT_MODELS:=siglip resnet152 swin_base deit3}"
# SEQUENTIAL_BACKWARD=1 backpropagates one encoder at a time (--fd_sequential_backward):
# same gradient, lower peak memory, a bit slower. Use it if the 9-encoder run OOMs.
: "${SEQUENTIAL_BACKWARD:=0}"
: "${MASTER_ADDR:=10.82.122.142}"
: "${MASTER_PORT:=29500}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${GLOBAL_BSZ:=1024}"
: "${SEED:=1}"
: "${PROJECT:=rebuttal_w3}"
: "${TAG:=}"
: "${STATS_GPUS:=8}"
: "${STATS_DECODE_THREADS:=8}"
: "${DRY_RUN:=0}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TIMM_USE_OLD_CACHE="${TIMM_USE_OLD_CACHE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a EXTRA <<< "$EXTRA_ENCODERS"
read -r -a GRAD_CKPT <<< "$GRAD_CKPT_MODELS"
if (( ${#EXTRA[@]} == 0 )); then echo "[ERR] EXTRA_ENCODERS is empty" >&2; exit 1; fi
for MODEL in "${EXTRA[@]}"; do
    case "$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')" in
        *dino*|*clip*|*convnext*|*siglip*|*mae*|*inception*)
            echo "[ERR] $MODEL overlaps the evaluation encoders (FDr6/FDr3) or SIM; drop it" >&2; exit 1 ;;
    esac
done
if (( $(printf '%s\n' "${EXTRA[@]}" | sort -u | wc -l) != ${#EXTRA[@]} )); then
    echo "[ERR] EXTRA_ENCODERS has duplicates" >&2; exit 1
fi

stats_path() {  # same naming as compute_repr_stats.py / infer_stats_path, 256px images at 224
    local safe="${1//\//_}"; safe="${safe//./_}"; safe="${safe//:/_}"; safe="${safe//-/_}"
    echo "data/fid_stats/$(printf '%s' "$safe" | tr '[:upper:]' '[:lower:]')_in256_t224_stats.npz"
}

run_command() {
    printf '[W3-many] Command: '; printf '%q ' "$@"; printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

case "$STAGE" in
    stats)
        if [[ "$DRY_RUN" != "1" && ! -d "${DATA_ROOT}/train" ]]; then
            echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
        fi
        for MODEL in "${EXTRA[@]}"; do
            STATS="$(stats_path "$MODEL")"
            if [[ -f "$STATS" ]]; then echo "[W3-many] stats exist: $STATS"; continue; fi
            run_command torchrun --standalone --nproc_per_node="$STATS_GPUS" compute_repr_stats.py \
                --model "$MODEL" --data_path "$DATA_ROOT" --img_size 256 --target_size 224 \
                --batch_size 128 --decode_threads "$STATS_DECODE_THREADS"
        done
        exit 0 ;;
    train) ;;
    *) echo "[ERR] STAGE must be stats or train" >&2; exit 1 ;;
esac

case "${NODE_RANK:-}" in
    0|1) ;;
    *) echo "[ERR] Set NODE_RANK=0 on the first machine, NODE_RANK=1 on the second" >&2; exit 1 ;;
esac
case "$MASTER_ADDR" in
    localhost|127.*|::1|0.0.0.0)
        echo "[ERR] MASTER_ADDR must be reachable from the second machine, not a loopback/bind address" >&2; exit 1 ;;
esac
case "$GLOBAL_BSZ" in
    ''|*[!0-9]*) echo "[ERR] GLOBAL_BSZ must be a positive integer divisible by 16" >&2; exit 1 ;;
esac
if (( GLOBAL_BSZ < 1 || GLOBAL_BSZ % 16 != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by 16" >&2; exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / 16 ))
export WANDB_MODE=online

REPRS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception "${EXTRA[@]}")
POOLS=(); SIZES=(224 224 256); WEIGHTS=(1 1 1)
for _ in "${REPRS[@]}"; do POOLS+=(cls); done
for _ in "${EXTRA[@]}"; do SIZES+=(224); WEIGHTS+=("$EXTRA_FD_WEIGHT"); done

if [[ "$DRY_RUN" != "1" ]]; then
    MISSING=0
    for REQUIRED in "${CKPT_ROOT}/JiT-B.pth"; do
        [[ -f "$REQUIRED" ]] || { echo "[ERR] Missing on node ${NODE_RANK}: $REQUIRED" >&2; MISSING=1; }
    done
    for MODEL in "${EXTRA[@]}"; do
        STATS="$(stats_path "$MODEL")"
        [[ -f "$STATS" ]] || { echo "[ERR] Missing on node ${NODE_RANK}: $STATS" >&2; MISSING=1; }
    done
    if (( MISSING )); then
        echo "[ERR] Prepare stats first: STAGE=stats bash $0" >&2; exit 1
    fi
    if [[ ! -d "${DATA_ROOT}/train" ]]; then
        echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
    fi
fi

SEQ_ARGS=()
if [[ "$SEQUENTIAL_BACKWARD" == "1" ]]; then SEQ_ARGS=(--fd_sequential_backward); fi
N_EXTRA=${#EXTRA[@]}
: "${EXP_NAME:=JiT_B-sim-frozen-plus${N_EXTRA}${TAG:+-$TAG}-w${EXTRA_FD_WEIGHT}-125k-gbs${GLOBAL_BSZ}-16gpu-seed${SEED}}"

COMMAND=(torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    main_fd.py --project "$PROJECT" --exp_name "$EXP_NAME" --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "${CKPT_ROOT}/JiT-B.pth"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models "${GRAD_CKPT[@]}"
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --fd_repr_models "${REPRS[@]}"
    --fd_repr_pool_types "${POOLS[@]}" --fd_target_sizes "${SIZES[@]}"
    --fd_repr_weights "${WEIGHTS[@]}" --fd_adv_weight 0.0 ${SEQ_ARGS[@]+"${SEQ_ARGS[@]}"}
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5 --auto_resume --enable_wandb)

printf '[W3-many] node=%s/2, GPUs=16, batch/GPU=%s, global batch=%s, steps=125000\n' "$NODE_RANK" "$BATCH_SIZE" "$GLOBAL_BSZ"
printf '[W3-many] %d frozen encoders: SIM + %s\n' "${#REPRS[@]}" "${EXTRA[*]}"
printf '[W3-many] W&B project=%s, run=%s (global rank 0 only)\n' "$PROJECT" "$EXP_NAME"
printf '[W3-many] Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then exec "${COMMAND[@]}"; fi
