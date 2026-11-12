#!/usr/bin/env bash
# Q1 / Table 6 cost: FD-loss (SIM) vs AdvFD-frozen vs AdvFD (trainable Inception).
# Each case runs the real main_fd.py loop on the same GPUs/batch/checkpoint:
# 50 warmup + 200 measured iterations, starting at virtual step 5000 so the
# AdvFD 1k/4k ramp is complete. These are cost measurements, not checkpoints.
#
# fd_loss:      SigLIP + MAE + Inception FD, as in table_3_JiT_adv_fd_sim.sh minus AdvFD.
# advfd_frozen: + frozen Inception adversarial FD, no D-step
#               (as in ablation/table_3_JiT_adv_fd_sim_forzen_adv.sh).
# advfd:        + trainable Inception, 1 D-step every 2 G-steps (table_3_JiT_adv_fd_sim.sh).
#
# Reports per-step wall-clock ratio vs FD-loss, peak allocated/reserved memory,
# D-step time (iterations with D-step minus iterations without) and its
# amortized share, and N = T * (ratio - 1) extra FD-loss steps.
#
# bash scripts/rebuttal/q1_cost_table6.sh
# TOTAL_STEPS=125000 CASES="advfd" bash scripts/rebuttal/q1_cost_table6.sh
# DRY_RUN=1 bash scripts/rebuttal/q1_cost_table6.sh

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${SEED:=1}"
: "${TOTAL_STEPS:=125000}"
: "${PROFILE_WARMUP_STEPS:=50}"
: "${PROFILE_STEPS:=200}"
: "${CASES:=fd_loss advfd_frozen advfd}"
: "${DRY_RUN:=0}"

if (( GLOBAL_BSZ % GPUS_PER_NODE != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be divisible by GPUS_PER_NODE" >&2; exit 1
fi
if (( PROFILE_STEPS % 2 != 0 )); then
    echo "[ERR] PROFILE_STEPS must be even so D-step/no-D-step iterations are balanced" >&2; exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE ))
PROFILE_DIR="${OUTPUT_DIR}/rebuttal_q1/cost-seed${SEED}-gbs${GLOBAL_BSZ}-gpus${GPUS_PER_NODE}"
LAUNCH=(torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1
    --master_port="$MASTER_PORT" --nproc_per_node="$GPUS_PER_NODE")

run_command() {
    printf '[Q1] Command: '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -d "${DATA_ROOT}/train" ]]; then
        echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
    fi
    if [[ ! -f "${CKPT_ROOT}/JiT-B.pth" ]]; then
        echo "[ERR] Missing ${CKPT_ROOT}/JiT-B.pth" >&2; exit 1
    fi
fi

COMMON=(--project rebuttal_q1 --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "${CKPT_ROOT}/JiT-B.pth"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models siglip
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception
    --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256
    --current_step 5000 --profile_warmup_steps "$PROFILE_WARMUP_STEPS"
    --profile_steps "$PROFILE_STEPS")
ADV_SHARED=(--fd_adv_repr_models inception --fd_adv_weight 0.1 --fd_adv_backbone repr
    --fd_adv_detach_real --fd_adv_start_step 1000 --fd_adv_warmup_steps 4000
    --fd_adv_whiten_eps 1e-3 --fd_adv_neg_real_degrade_ratio 0
    --fd_adv_log_raw --fd_adv_log_raw_freq 1000 --fd_adv_ema_beta 0.99)
ADV_TRAIN=(--fd_adv_lr 1e-6 --fd_adv_steps 1 --fd_adv_update_freq 2 --fd_adv_grad_clip 1.0)
ADV_FROZEN=(--fd_adv_lr 0.0 --fd_adv_steps 0 --fd_adv_update_freq 1 --fd_adv_grad_clip 0.0)

for CASE in $CASES; do
    case "$CASE" in
        fd_loss) EXTRA=() ;;
        advfd_frozen) EXTRA=("${ADV_SHARED[@]}" "${ADV_FROZEN[@]}") ;;
        advfd) EXTRA=("${ADV_SHARED[@]}" "${ADV_TRAIN[@]}") ;;
        *) echo "[ERR] unknown case: $CASE (fd_loss | advfd_frozen | advfd)" >&2; exit 1 ;;
    esac
    run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
        "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --exp_name "cost-${CASE}" \
        --profile_output "${PROFILE_DIR}/${CASE}.json"
done

if [[ -f "${PROFILE_DIR}/fd_loss.json" && -f "${PROFILE_DIR}/advfd_frozen.json" && -f "${PROFILE_DIR}/advfd.json" ]] \
    || [[ "$DRY_RUN" == "1" ]]; then
    run_command python scripts/rebuttal/profile_fd_overhead.py --summarize_q1 "$PROFILE_DIR" "$TOTAL_STEPS"
fi
