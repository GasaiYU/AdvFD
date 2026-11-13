#!/usr/bin/env bash
# Evaluate every JiT-B training checkpoint with step in [MIN_STEP, MAX_STEP] of two runs
# (AdvFD and FD-Loss) with the standard evaluator, then collect FID(ADM), FDr6 and FDr3.
# Per checkpoint this is the same call as scripts/evaluate_ours.sh (PRESET=JiT_B,
# 50k images, online weights, 1 sampling step); the six-space FD of one checkpoint is
# in its final_eval_summary.csv.
#
# Results: $RESULT_ROOT/rebuttal_sweep_<run>/JiT_B-step_NNNNNNN/final_eval_summary.csv
# Summary: $RESULT_ROOT/rebuttal_sweep_summary.csv   (run, step, fid, fdr6, fdr3)
#
# bash scripts/rebuttal/eval_ckpt_sweep.sh                    # both runs, then summary
# DRY_RUN=1 bash scripts/rebuttal/eval_ckpt_sweep.sh          # list what would run
# RUNS=advfd bash scripts/rebuttal/eval_ckpt_sweep.sh         # one run only
# STAGE=summarize bash scripts/rebuttal/eval_ckpt_sweep.sh    # only rebuild the table
# NUM_IMAGES=1024 EVAL_BSZ=64 GPUS_PER_NODE=1 MIN_STEP=124000 bash scripts/rebuttal/eval_ckpt_sweep.sh  # smoke test
#
# Finished checkpoints are skipped (SKIP_DONE=1), so an interrupted sweep can be restarted.

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${ADV_DIR:=work_dirs/Jit-B-adv/JiT_B-fd-sim-advinc-w0.1-from-base/checkpoints}"
: "${FD_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_JiT-B/JiT_B-fd-sim/checkpoints}"
: "${RUNS:=advfd fd_loss}"
: "${STAGE:=all}"
: "${MIN_STEP:=0}"
: "${MAX_STEP:=125000}"
: "${GPUS_PER_NODE:=8}"
: "${NUM_IMAGES:=50000}"
: "${EVAL_BSZ:=256}"
: "${MASTER_PORT:=29612}"
: "${RESULT_ROOT:=./work_dirs/eval_results/rebuttal}"
: "${SKIP_DONE:=1}"
: "${DRY_RUN:=0}"
: "${PYTHON:=python}"

case "$STAGE" in eval|summarize|all) ;; *) echo "[ERR] STAGE must be eval | summarize | all" >&2; exit 1 ;; esac

run_dir() {
    case "$1" in
        advfd) echo "$ADV_DIR" ;;
        fd_loss) echo "$FD_DIR" ;;
        *) echo "[ERR] unknown run '$1' (advfd | fd_loss)" >&2; return 1 ;;
    esac
}

# step_NNNNNNN.pth with MIN_STEP <= N <= MAX_STEP, ascending; latest.pth is not matched.
list_ckpts() {
    local dir="$1" file name step
    for file in "$dir"/step_*.pth; do
        [[ -f "$file" ]] || continue
        name="${file##*/}"; name="${name#step_}"; name="${name%.pth}"
        [[ "$name" =~ ^[0-9]+$ ]] || continue
        step=$((10#$name))
        if (( step >= MIN_STEP && step <= MAX_STEP )); then printf '%s\n' "$file"; fi
    done
}

# Done = FID(ADM), FDr6 and the FDr3 encoders are all in the checkpoint's summary CSV.
is_done() {
    "$PYTHON" utils/compute_fdr_subsets.py "$1" 2>/dev/null | grep -q '^FDr6:'
}

FAILED=()

eval_run() {
    local run="$1" dir project ckpt name exp out
    local ckpts=()
    dir="$(run_dir "$run")"
    project="rebuttal_sweep_${run}"
    if [[ ! -d "$dir" ]]; then echo "[ERR] checkpoint dir not found: $dir" >&2; exit 1; fi
    while IFS= read -r ckpt; do ckpts+=("$ckpt"); done < <(list_ckpts "$dir")
    if (( ${#ckpts[@]} == 0 )); then
        echo "[ERR] no step_*.pth with step in [$MIN_STEP, $MAX_STEP] under $dir" >&2; exit 1
    fi
    echo "[sweep] $run: ${#ckpts[@]} checkpoints in $dir"
    for ckpt in "${ckpts[@]}"; do
        name="${ckpt##*/}"; name="${name%.pth}"
        exp="JiT_B-${name}"
        out="${RESULT_ROOT}/${project}/${exp}"
        if [[ "$SKIP_DONE" == "1" ]] && is_done "$out"; then
            echo "[sweep] $run $name: done, skipping"; continue
        fi
        echo "[sweep] $run $name: evaluating $ckpt"
        if [[ "$DRY_RUN" == "1" ]]; then continue; fi
        if ! PRESET=JiT_B CKPT_PATH="$ckpt" PROJECT="$project" EXP_NAME="$exp" \
            RESULT_ROOT="$RESULT_ROOT" GPUS_PER_NODE="$GPUS_PER_NODE" NUM_IMAGES="$NUM_IMAGES" \
            EVAL_BSZ="$EVAL_BSZ" MASTER_PORT="$MASTER_PORT" \
            bash scripts/evaluate_released_ckpt.sh < /dev/null; then
            echo "[sweep] FAILED: $run $name" >&2
            FAILED+=("$run/$name")
        fi
    done
}

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    for run in $RUNS; do eval_run "$run"; done
fi

if [[ ( "$STAGE" == "summarize" || "$STAGE" == "all" ) && "$DRY_RUN" != "1" ]]; then
    SUMMARY_ARGS=()
    for run in $RUNS; do
        run_dir "$run" > /dev/null
        SUMMARY_ARGS+=(--run "${run}=${RESULT_ROOT}/rebuttal_sweep_${run}")
    done
    "$PYTHON" scripts/rebuttal/summarize_ckpt_sweep.py "${SUMMARY_ARGS[@]}" \
        --out "${RESULT_ROOT}/rebuttal_sweep_summary.csv"
fi

if (( ${#FAILED[@]} > 0 )); then
    echo "[sweep] ${#FAILED[@]} evaluation(s) failed: ${FAILED[*]}" >&2
    echo "[sweep] rerun the same command to retry them; finished ones are skipped" >&2
    exit 1
fi
