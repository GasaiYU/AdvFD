#!/usr/bin/env bash
# MIND sample-size curve from 5k to 50k for Inception, SigLIP, and MAE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PRESET="${PRESET:-JiT_B}"
export CKPT_PATH="${CKPT_PATH:-/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/JiT-B-MIND/JiT_B-mind-inception-fq10000-rq50000-s10000-g1.0-p1000/checkpoints/step_0082599.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-256}"
export MASTER_PORT="${MASTER_PORT:-29613}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_mind_curve}"
export EXP_NAME="${EXP_NAME:-JiT_B-mind-test-MIND-Loss}"

SIGLIP="${SIGLIP:-vit_so400m_patch16_siglip_256.v2_webli}"
MAE="${MAE:-vit_large_patch16_224.mae}"
export MIND_MODELS="${MIND_MODELS:-inception ${SIGLIP} ${MAE}}"
export MIND_SAMPLE_SIZES="${MIND_SAMPLE_SIZES:-10000}"
export MIND_REF_DIR="${MIND_REF_DIR:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/train}"
export MIND_NUM_PROJECTIONS="${MIND_NUM_PROJECTIONS:-1000}"
export MIND_PROJECTION_BATCH_SIZE="${MIND_PROJECTION_BATCH_SIZE:-1000}"
export MIND_RNG_SEED="${MIND_RNG_SEED:-2030}"
export PLOT_MIND_CURVE="${PLOT_MIND_CURVE:-0}"

read -r -a MIND_MODEL_ARGS <<< "$MIND_MODELS"
read -r -a MIND_SAMPLE_SIZE_ARGS <<< "$MIND_SAMPLE_SIZES"

bash scripts/evaluate_released_ckpt.sh \
    --enable_vis --vis_steps 1 --num_sampling_steps 1 \
    --models "${MIND_MODEL_ARGS[@]}" \
    --eval_mind \
    --mind_models "${MIND_MODEL_ARGS[@]}" \
    --mind_ref_dir "$MIND_REF_DIR" \
    --mind_sample_sizes "${MIND_SAMPLE_SIZE_ARGS[@]}" \
    --mind_num_projections "$MIND_NUM_PROJECTIONS" \
    --mind_projection_batch_size "$MIND_PROJECTION_BATCH_SIZE" \
    --mind_rng_seed "$MIND_RNG_SEED" \
    "$@"

CSV_PATH="${RESULT_ROOT}/${PROJECT}/${EXP_NAME}/final_eval_summary.csv"
CURVE_CSV="${CSV_PATH%.csv}_mind_curve.csv"
CURVE_PNG="${CSV_PATH%.csv}_mind_curve.png"

if [ "$PLOT_MIND_CURVE" = "1" ] && [ -f "$CSV_PATH" ]; then
    CSV_PATH="$CSV_PATH" CURVE_CSV="$CURVE_CSV" CURVE_PNG="$CURVE_PNG" python - <<'PY'
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"[WARN] Could not plot MIND curve: {exc}")
    raise SystemExit(0)

csv_path = os.environ["CSV_PATH"]
curve_csv = os.environ["CURVE_CSV"]
curve_png = os.environ["CURVE_PNG"]

df = pd.read_csv(csv_path, on_bad_lines="skip")
if not {"model", "mind", "mind_n"}.issubset(df.columns):
    print("[WARN] No MIND columns found; skipping curve plot")
    raise SystemExit(0)

curve = df.dropna(subset=["mind", "mind_n"]).copy()
curve["mind"] = pd.to_numeric(curve["mind"], errors="coerce")
curve["mind_n"] = pd.to_numeric(curve["mind_n"], errors="coerce")
curve = curve.dropna(subset=["mind", "mind_n"])

# Inception has two FID reference rows but a shared MIND value. Keep one.
curve = curve[curve["model"] != "FID(JiT)"]
curve["plot_model"] = curve["model"].replace({"FID(ADM)": "inception"})
curve = curve.sort_values(["plot_model", "mind_n"])
curve = curve.drop_duplicates(["plot_model", "mind_n"], keep="last")

if curve.empty:
    print("[WARN] No MIND rows found; skipping curve plot")
    raise SystemExit(0)

curve[["plot_model", "mind_n", "mind", "mind_m", "n", "cfg", "ema_label"]].to_csv(
    curve_csv, index=False
)

plt.figure(figsize=(8, 5))
for name, sub in curve.groupby("plot_model", sort=True):
    sub = sub.sort_values("mind_n")
    plt.plot(sub["mind_n"], sub["mind"], marker="o", linewidth=1.8, label=name)

plt.xlabel("MIND sample size")
plt.ylabel("MIND")
plt.title("MIND sample-size curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(curve_png, dpi=160)
print(f"[MIND] curve CSV: {curve_csv}")
print(f"[MIND] curve PNG: {curve_png}")
PY
fi
