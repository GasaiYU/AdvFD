#!/usr/bin/env bash
# Find the strongest visible Fourier pattern that still beats zero-pattern FID.
#
# Start from a zero Fourier pattern at a small alpha, then increase alpha in
# stages. Each stage fits/selects on the same fixed 50k optimization set. A
# failed stage is not adopted: the script stops and reports the last successful
# nonzero checkpoint.

set -euo pipefail

# Complete default configuration. With these values the script can be invoked
# without any environment variables; every entry remains overrideable.
: "${CKPT_ROOT:=./checkpoints/base}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${GPUS_PER_NODE:=8}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GLOBAL_CACHE_BSZ:=1024}"
: "${GRADIENT_BSZ:=256}"
: "${EVAL_BSZ_PER_GPU:=128}"
: "${OPT_IMAGES:=50000}"
: "${VAL_IMAGES:=5000}"
: "${TEST_IMAGES:=50000}"
: "${CACHE_ROOT:=./work_dirs/hacking_cache}"
: "${ENABLE_WANDB:=1}"

: "${ALPHA_STAGES:=0.01 0.02 0.0313725490196 0.05 0.075 0.1 0.125 0.15 0.175 0.2}"
: "${EVAL_ALPHAS:=0 0.01 0.02 0.0313725490196 0.05 0.075 0.1 0.125 0.15 0.175 0.2}"
: "${INITIAL_PGD_STEPS:=50}"
: "${PGD_STEPS_PER_STAGE:=20}"
: "${PGD_STEP_SIZE:=0.02}"
: "${BASE_EXP_NAME:=pMF_B_256-visible-from-scratch}"
: "${OUTPUT_ROOT:=./work_dirs}"

read -r -a STAGES <<< "$ALPHA_STAGES"
read -r -a DOSE_VALUES <<< "$EVAL_ALPHAS"

current_pattern=""
last_success_alpha="none"
last_success_pattern="none"

for stage_index in "${!STAGES[@]}"; do
    target_alpha="${STAGES[$stage_index]}"
    alpha_label="${target_alpha//./p}"
    stage_exp="${BASE_EXP_NAME}-alpha${alpha_label}"

    echo "[curriculum] target_alpha=${target_alpha}"
    continuation_args=()
    if (( stage_index == 0 )); then
        echo "[curriculum] init=zero-pattern (from scratch)"
        stage_pgd_steps="$INITIAL_PGD_STEPS"
    else
        echo "[curriculum] init=${current_pattern}"
        stage_pgd_steps="$PGD_STEPS_PER_STAGE"
        continuation_args=(
            --hack_init_pattern_checkpoint
            "$current_pattern"
        )
    fi

    CKPT_ROOT="$CKPT_ROOT" \
    MODEL_SIZE="$MODEL_SIZE" \
    RES="$RES" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    NNODES="$NNODES" \
    NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    GLOBAL_CACHE_BSZ="$GLOBAL_CACHE_BSZ" \
    GRADIENT_BSZ="$GRADIENT_BSZ" \
    EVAL_BSZ_PER_GPU="$EVAL_BSZ_PER_GPU" \
    OPT_IMAGES="$OPT_IMAGES" \
    VAL_IMAGES="$VAL_IMAGES" \
    TEST_IMAGES="$TEST_IMAGES" \
    CACHE_ROOT="$CACHE_ROOT" \
    ENABLE_WANDB="$ENABLE_WANDB" \
    EXP_NAME="$stage_exp" \
    PGD_STEPS="$stage_pgd_steps" \
    bash hacking/run_pmf_fourier_universal.sh \
        --output_dir "$OUTPUT_ROOT" \
        --hack_overfit_only \
        "${continuation_args[@]}" \
        --hack_train_alpha "$target_alpha" \
        --hack_pgd_step_size "$PGD_STEP_SIZE" \
        --hack_eval_alphas "${DOSE_VALUES[@]}"

    stage_root="${OUTPUT_ROOT}/pMF_universal_pattern/${stage_exp}"
    summary_path="${stage_root}/overfit_summary.json"
    selected_path="${stage_root}/checkpoints/fourier_pattern_overfit_selected.pth"
    if [[ ! -f "$summary_path" || ! -f "$selected_path" ]]; then
        echo "[ERR] stage outputs are incomplete under: $stage_root"
        exit 1
    fi

    fit_success="$(
        python -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["fit_success"] else "0")' \
            "$summary_path"
    )"
    if [[ "$fit_success" != "1" ]]; then
        echo "[curriculum] target alpha ${target_alpha} did not beat the zero-pattern baseline."
        if (( stage_index == 0 )); then
            echo "[curriculum] from-scratch initialization failed; no nonzero pattern was selected."
        else
            echo "[curriculum] stopping before adopting the stage's zero fallback."
        fi
        break
    fi

    current_pattern="$selected_path"
    last_success_alpha="$target_alpha"
    last_success_pattern="$selected_path"
    echo "[curriculum] accepted alpha=${target_alpha}: ${selected_path}"
done

echo "[curriculum] strongest successful alpha: ${last_success_alpha}"
echo "[curriculum] selected pattern: ${last_success_pattern}"
