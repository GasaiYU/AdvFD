#!/usr/bin/env bash
# Clean FD of JiT-B checkpoints with the standard evaluator (six encoders, 50k images).
# Thin wrapper over scripts/evaluate_released_ckpt.sh, same pattern as scripts/evaluate_ours.sh.
# The generated images are kept on disk so the same images can be perturbed later
# and scored in the adaptive Inception space (scripts/rebuttal/adv_inception_fd.py).
#
# Results: $RESULT_ROOT/$PROJECT/$EXP_NAME/final_eval_summary.csv   (fd, fdr, fdr6)
# Images:  $RESULT_ROOT/$PROJECT/$EXP_NAME/eval_images/ema=online-cfg=3.0-steps=1-interval_min=0.1-interval_max=1.0/NNNNNN.png
#
# CASE=advfd     AdvFD JiT-B (needs ADV_CKPT)
# CASE=fd_loss   FD-Loss JiT-B (needs FD_CKPT)
# CASE=all       both, one after the other
#
# CASE=advfd FD_CKPT=... ADV_CKPT=... bash scripts/rebuttal/eval_clean_fd.sh
# NUM_IMAGES=1024 EVAL_BSZ=64 GPUS_PER_NODE=1 CASE=advfd bash scripts/rebuttal/eval_clean_fd.sh   # smoke test

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${CASE:=advfd}"
: "${ADV_CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/Jit-adv-ablation/JiT_B-fd-sim-advinc-w0.1-from-base-ADVLR-1e-6/checkpoints/step_0124999.pth}"
: "${FD_CKPT:=}"
: "${GPUS_PER_NODE:=8}"
: "${NUM_IMAGES:=50000}"
: "${EVAL_BSZ:=256}"
: "${MASTER_PORT:=29612}"
: "${RESULT_ROOT:=./work_dirs/eval_results/rebuttal}"
: "${PROJECT:=rebuttal_clean_fd}"
: "${SAVE_IMAGES:=1}"

run_case() {
    local name="$1" ckpt="$2"
    shift 2  # what remains is passed through to eval_all_fds.py
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[ERR] checkpoint for case '$name' not found: '${ckpt}'" >&2; exit 1
    fi
    local extra=()
    if [[ "$SAVE_IMAGES" == "1" ]]; then extra=(--save_eval_images --keep_eval_folder); fi
    echo "[clean-fd] case=$name ckpt=$ckpt"
    PRESET=JiT_B CKPT_PATH="$ckpt" EXP_NAME="JiT_B-${name}-clean" \
        GPUS_PER_NODE="$GPUS_PER_NODE" NUM_IMAGES="$NUM_IMAGES" EVAL_BSZ="$EVAL_BSZ" \
        MASTER_PORT="$MASTER_PORT" RESULT_ROOT="$RESULT_ROOT" PROJECT="$PROJECT" \
        bash scripts/evaluate_released_ckpt.sh ${extra[@]+"${extra[@]}"} "$@"
}

case "$CASE" in
    advfd) run_case advfd "$ADV_CKPT" "$@" ;;
    fd_loss) run_case fd_loss "$FD_CKPT" "$@" ;;
    all) run_case fd_loss "$FD_CKPT" "$@"; run_case advfd "$ADV_CKPT" "$@" ;;
    *) echo "[ERR] CASE must be advfd | fd_loss | all" >&2; exit 1 ;;
esac
