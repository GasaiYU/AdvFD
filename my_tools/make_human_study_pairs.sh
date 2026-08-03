#!/usr/bin/env bash
# Build shuffled FD-Loss vs AdvFD pairs for a human study.
#
# Outputs (under $OUTPUT_DIR):
#   images/          rater-facing pairs, e.g. 001_07513.png
#   manifest.json    full record: source paths, seed, per-trial shuffle
#   answer_key.csv   trial,image,id,left,right,ours_side,swapped
#
# Optional overrides:
#   MODEL=jit_l|pmf_l          picks the default INPUT_DIR
#   ID_FILE=my_tools/paper_id.txt
#   INPUT_DIR=path/to/<model>_threeway
#   OUTPUT_DIR=paper/human_study_<model>
#   QUALI_ROOT=/mmu-vcg/gaomingju/data/FD-Loss/quali
#   SEED=20260731              fixed so a study can be reproduced
#   GAP=8                      white separator in pixels
#   PANEL_SIZE=256             default: source width / 3
#   BASELINE_NAME / FD_NAME / OURS_NAME    answer-key labels
#   OVERWRITE=1                replace an existing output directory
#   SHUFFLE_ORDER=0            keep ID-file order
#   BALANCED=0                 flip each trial independently
#
# Extra flags are passed straight through to the Python script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL:=jit_l}"
: "${QUALI_ROOT:=/mmu-vcg/gaomingju/data/FD-Loss/quali}"
: "${ID_FILE:=$SCRIPT_DIR/paper_id.txt}"
: "${SEED:=20260731}"
: "${GAP:=8}"
: "${PYTHON:=python}"

case "$MODEL" in
    jit_l)
        : "${INPUT_DIR:=$QUALI_ROOT/jit_l_threeway}"
        : "${BASELINE_NAME:=JiT-L}"
        : "${FD_NAME:=JiT-L + FD Loss}"
        : "${OURS_NAME:=JiT-L + AdvFD}"
        ;;
    pmf_l)
        : "${INPUT_DIR:=$QUALI_ROOT/pmf_l/pmf_l_threeway}"
        : "${BASELINE_NAME:=pMF-L}"
        : "${FD_NAME:=pMF-L + FD Loss}"
        : "${OURS_NAME:=pMF-L + AdvFD}"
        ;;
    *)
        # Unknown model: INPUT_DIR must be supplied explicitly.
        : "${BASELINE_NAME:=Baseline}"
        : "${FD_NAME:=FD-Loss}"
        : "${OURS_NAME:=AdvFD}"
        if [[ -z "${INPUT_DIR:-}" ]]; then
            echo "[ERR] MODEL=$MODEL is not a known preset;" \
                 "set INPUT_DIR explicitly" >&2
            exit 1
        fi
        ;;
esac

: "${OUTPUT_DIR:=$REPO_ROOT/paper/human_study_$MODEL}"

if [[ ! -f "$ID_FILE" ]]; then
    echo "[ERR] ID file not found: $ID_FILE" >&2
    echo "      Create it with one image ID per line, for example:" >&2
    echo "        7513      # FD-Loss vs AdvFD" >&2
    echo "        2091*     # baseline vs AdvFD instead" >&2
    echo "        5005-     # skip this ID" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERR] input directory not found: $INPUT_DIR" >&2
    exit 1
fi

ARGS=(
    --id_file "$ID_FILE"
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --gap "$GAP"
    --baseline_name "$BASELINE_NAME"
    --fd_name "$FD_NAME"
    --ours_name "$OURS_NAME"
)

if [[ -n "${PANEL_SIZE:-}" ]]; then
    ARGS+=(--panel_size "$PANEL_SIZE")
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${SHUFFLE_ORDER:-1}" == "0" ]]; then
    ARGS+=(--no_shuffle_order)
fi
if [[ "${BALANCED:-1}" == "0" ]]; then
    ARGS+=(--no_balanced)
fi

echo "[run] model=$MODEL seed=$SEED"
echo "[run] ids   = $ID_FILE"
echo "[run] input = $INPUT_DIR"
echo "[run] output= $OUTPUT_DIR"

exec "$PYTHON" my_tools/make_human_study_pairs.py "${ARGS[@]}" "$@"
