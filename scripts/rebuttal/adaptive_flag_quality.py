#!/usr/bin/env python3
"""Do humans agree with what the adaptive representation flags?

Rank the 50k FD-loss samples by their distance to the real distribution in
the adaptive (AdvFD-trained) Inception space; compare the top, a random and
the bottom 1% with MUSIQ and with a blind 1-5 human rating study (MOS).
The pretrained Inception is ranked the same way as a MUSIQ-only control.

Per-sample distance, with (mu_r, Sigma_r) estimated from the same real
ImageNet images in each feature space and Sigma_r + eps*I = V Lambda V^T:

    d(x) = || Lambda^{-1/2} V^T (f(x) - mu_r) ||^2

i.e. the per-sample mean term of the real-whitened FD that AdvFD optimizes,
using the same eps as ``--fd_adv_whiten_eps``. Larger d = flagged.

Stages (single GPU or CPU, see adaptive_flag_quality.sh):
  rank   features + distances for both encoders -> scores.csv, groups.json
  musiq  MUSIQ (pyiqa, KonIQ-trained, ~0-100) for every grouped image
  human  N images per adaptive group, shuffled, anonymized, plus rate.html
  mos    MOS + 95% CI, Krippendorff's alpha, final MUSIQ/MOS table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ENCODERS = ("adaptive", "initial")
# (key, table label); MOS is collected for the first three only.
GROUPS = (
    ("adaptive_top", "Top 1% (flagged by adaptive repr.)"),
    ("random", "Random 1%"),
    ("adaptive_bottom", "Bottom 1% (adaptive repr.)"),
    ("initial_top", "Top 1% (flagged by initial Inception)"),
    ("initial_bottom", "Bottom 1% (initial Inception)"),
)
HUMAN_GROUPS = ("adaptive_top", "random", "adaptive_bottom")


# ---------------------------------------------------------------- utilities

def load_panel(path, panel):
    """Return one square panel of a horizontal strip (or the image itself)."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    if width % height:
        raise ValueError(f"{path}: {width}x{height} is not a strip of square panels")
    if not 0 <= panel < width // height:
        raise ValueError(f"{path}: panel {panel} outside 0..{width // height - 1}")
    return image.crop((panel * height, 0, (panel + 1) * height, height))


class PanelDataset:
    def __init__(self, root, names, panel):
        self.root, self.names, self.panel = Path(root), list(names), panel

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        import torchvision.transforms.functional as TF

        return TF.to_tensor(load_panel(self.root / self.names[index], self.panel)), index


def list_samples(samples_dir):
    names = sorted(
        p.name for p in Path(samples_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )
    if not names:
        raise FileNotFoundError(f"no images in {samples_dir}")
    return names


def real_image_paths(data_root, per_class, seed):
    """Class-balanced, seeded subset of ImageNet train."""
    train_dir = Path(data_root)
    if (train_dir / "train").is_dir():
        train_dir = train_dir / "train"
    rng = random.Random(seed)
    paths = []
    for wnid in sorted(os.listdir(train_dir)):
        class_dir = train_dir / wnid
        if not class_dir.is_dir():
            continue
        files = sorted(f for f in os.listdir(class_dir) if Path(f).suffix.lower() in IMAGE_EXTS)
        paths += [str(class_dir / f) for f in rng.sample(files, min(per_class, len(files)))]
    if not paths:
        raise FileNotFoundError(f"no ImageNet class folders under {train_dir}")
    return paths


def describe(values, confidence=0.95):
    from scipy import stats

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def mann_whitney_p(a, b):
    from scipy import stats

    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def krippendorff_alpha(units, metric="interval"):
    """Krippendorff's alpha; ``units`` holds the pooled ratings of each image."""
    values = sorted({v for unit in units for v in unit})
    if len(values) < 2:
        return float("nan")
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for unit in units:
        if len(unit) < 2:
            continue
        counts = np.bincount([index[v] for v in unit], minlength=len(values)).astype(float)
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (len(unit) - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if metric == "interval":
        v = np.asarray(values, dtype=float)
        delta = (v[:, None] - v[None, :]) ** 2
    elif metric == "ordinal":
        i = np.arange(len(values))
        lo, hi = np.minimum.outer(i, i), np.maximum.outer(i, i)
        cum = np.cumsum(n_c)
        delta = (cum[hi] - cum[lo] + n_c[lo] - (n_c[lo] + n_c[hi]) / 2) ** 2
    else:
        raise ValueError(metric)
    observed = (coincidence * delta).sum() / n
    expected = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1.0 - observed / expected)


def load_groups(out_dir):
    path = Path(out_dir) / "groups.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the rank stage first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------- rank

def load_encoders(checkpoint):
    import torch
    from frechet_distance.repr_models import load_repr_model
    from my_tools.compute_fd_adv_norm_ratio import resolve_checkpoint_path, select_adv_state

    checkpoint_path = resolve_checkpoint_path(Path(checkpoint))
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not state_dict.get("fd_adv_states"):
        raise KeyError(f"{checkpoint_path} has no fd_adv_states; pass an AdvFD checkpoint")
    state = select_adv_state(state_dict, repr_model="inception", adv_state_name=None)
    transform = state.get("feature_transform", "none")
    if transform not in ("none", "hard_l2_v1"):
        raise NotImplementedError(f"feature_transform={transform!r} depends on the batch")
    cap = float(state.get("feature_norm_cap", 0.0)) if transform == "hard_l2_v1" else 0.0

    initial, _, _, _ = load_repr_model("inception", device="cuda")
    adaptive, _, _, _ = load_repr_model("inception", device="cuda")
    adaptive.load_state_dict(state["model"], strict=True)
    del state_dict
    for model in (initial, adaptive):
        model.eval().requires_grad_(False)
    print(f"[rank] adaptive Inception from {checkpoint_path} "
          f"(state={state.get('name')}, transform={transform})", flush=True)
    return {"adaptive": (adaptive, cap), "initial": (initial, 0.0)}, str(checkpoint_path)


def extract_features(encoders, dataset, batch_size, num_workers, desc):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from frechet_distance.adversarial import hard_l2_feature_cap

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats = {name: [] for name in encoders}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).cuda(non_blocking=True)
            for name, (model, cap) in encoders.items():
                features = model(images)[0].float()  # Inception in FP32, as in training
                feats[name].append(hard_l2_feature_cap(features, cap).cpu())
    return {name: torch.cat(chunks) for name, chunks in feats.items()}


def whitened_distances(real, fake, eps):
    import torch
    from frechet_distance.adversarial import build_real_whitening

    real = real.cuda().double()
    mu = real.mean(dim=0)
    centered = real - mu
    cov = centered.T @ centered / (real.shape[0] - 1)
    mu, eigvecs, inv_sqrt = build_real_whitening(mu, cov, eps=eps)
    out = []
    for chunk in fake.split(4096):
        z = ((chunk.cuda().double() - mu) @ eigvecs) * inv_sqrt
        out.append(z.square().sum(dim=1).cpu())
    return torch.cat(out).numpy()


def stage_rank(args):
    import torch
    from scipy import stats
    from frechet_distance.datasets import ImageListDataset

    if not torch.cuda.is_available():
        raise SystemExit("rank stage needs a CUDA GPU")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list_samples(args.samples_dir)
    encoders, checkpoint_path = load_encoders(args.checkpoint)

    real_paths = real_image_paths(args.data_root, args.real_per_class, args.seed)
    print(f"[rank] {len(real_paths)} real ImageNet images, {len(names)} FD-loss samples", flush=True)
    real = extract_features(encoders, ImageListDataset(real_paths, img_size=args.img_size),
                            args.batch_size, args.num_workers, "real")
    fake = extract_features(encoders, PanelDataset(args.samples_dir, names, args.panel),
                            args.batch_size, args.num_workers, "FD-loss")
    dist = {name: whitened_distances(real[name], fake[name], args.whiten_eps) for name in ENCODERS}

    n = len(names)
    k = max(1, round(args.fraction * n))
    groups = {"random": sorted(random.Random(args.seed).sample(names, k))}
    rank_of = {}
    for name in ENCODERS:
        order = np.argsort(-dist[name], kind="stable")  # most flagged first
        groups[f"{name}_top"] = [names[i] for i in order[:k]]
        groups[f"{name}_bottom"] = [names[i] for i in order[-k:][::-1]]
        rank_of[name] = np.empty(n, dtype=np.int64)
        rank_of[name][order] = np.arange(1, n + 1)

    write_csv(out_dir / "scores.csv", [
        {"filename": names[i],
         **{f"dist_{name}": f"{dist[name][i]:.6f}" for name in ENCODERS},
         **{f"rank_{name}": int(rank_of[name][i]) for name in ENCODERS}}
        for i in range(n)
    ])
    top_a, top_i = set(groups["adaptive_top"]), set(groups["initial_top"])
    rho = float(stats.spearmanr(dist["adaptive"], dist["initial"])[0])
    meta = {
        "samples_dir": str(Path(args.samples_dir).resolve()), "panel": args.panel,
        "checkpoint": checkpoint_path, "data_root": str(args.data_root),
        "real_images": len(real_paths), "real_per_class": args.real_per_class,
        "whiten_eps": args.whiten_eps, "num_samples": n, "fraction": args.fraction,
        "group_size": k, "seed": args.seed,
        "top_overlap_adaptive_vs_initial": len(top_a & top_i),
        "spearman_adaptive_vs_initial": rho,
    }
    (out_dir / "groups.json").write_text(json.dumps({"meta": meta, "groups": groups}, indent=2) + "\n")
    print(f"[rank] top-{k} overlap adaptive vs initial: {len(top_a & top_i)}/{k}; "
          f"Spearman(d_adaptive, d_initial) = {rho:.3f}")
    print(f"[rank] saved {out_dir / 'scores.csv'} and {out_dir / 'groups.json'}")


# -------------------------------------------------------------------- musiq

def stage_musiq(args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit("MUSIQ needs pyiqa: pip install pyiqa") from exc

    out_dir = Path(args.out_dir)
    data = load_groups(out_dir)
    groups, meta = data["groups"], data["meta"]
    names = sorted(set().union(*groups.values()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric("musiq", device=device)  # KonIQ-10k, higher is better
    loader = DataLoader(PanelDataset(meta["samples_dir"], names, meta["panel"]),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    scores = {}
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="MUSIQ"):
            values = metric(images.to(device)).flatten().float().cpu().tolist()
            for index, value in zip(indices.tolist(), values):
                scores[names[index]] = value
    write_csv(out_dir / "musiq_scores.csv",
              [{"filename": name, "musiq": f"{scores[name]:.4f}"} for name in names])

    rows = []
    for key, label in GROUPS:
        values = [scores[name] for name in groups[key]]
        p = "" if key == "random" else mann_whitney_p(values, [scores[x] for x in groups["random"]])
        rows.append({"group": key, "label": label, **describe(values), "p_vs_random": p})
    write_csv(out_dir / "musiq_summary.csv", rows)
    print(f"{'group':40} {'n':>4} {'MUSIQ':>8} {'95% CI':>18} {'p vs random':>12}")
    for row in rows:
        p = f"{row['p_vs_random']:.2e}" if row["p_vs_random"] != "" else "-"
        print(f"{row['label']:40} {row['n']:4d} {row['mean']:8.2f} "
              f"[{row['ci95_low']:6.2f}, {row['ci95_high']:6.2f}] {p:>12}")
    top_a = [scores[x] for x in groups["adaptive_top"]]
    top_i = [scores[x] for x in groups["initial_top"]]
    print(f"adaptive top vs initial top: dMUSIQ = {np.mean(top_a) - np.mean(top_i):+.2f}, "
          f"p = {mann_whitney_p(top_a, top_i):.2e} "
          f"(overlap {meta['top_overlap_adaptive_vs_initial']}/{meta['group_size']})")
    print(f"[musiq] saved {out_dir / 'musiq_summary.csv'}")


# -------------------------------------------------------------------- human

RATE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Image quality rating</title>
<style>
body{font-family:sans-serif;max-width:760px;margin:24px auto;text-align:center}
img{width:512px;height:512px;max-width:92vw;max-height:92vw;object-fit:contain;background:#eee}
button{font-size:18px;margin:4px;padding:8px 18px}
.hint{color:#555}
</style></head><body>
<div id="start">
  <h2>Image quality rating</h2>
  <p>Rate the overall visual quality of each image on a 1&ndash;5 scale.<br>
  <b>1</b> = obvious artifacts &nbsp;&middot;&nbsp; <b>5</b> = looks completely real.</p>
  <p class="hint">Keys 1&ndash;5 also work; Backspace goes back. Progress is saved in this browser.</p>
  <input id="rater" placeholder="rater ID" autofocus> <button onclick="begin()">Start</button>
</div>
<div id="task" hidden>
  <p id="progress"></p><img id="img" alt="">
  <div>__BUTTONS__</div>
  <p class="hint">1 = obvious artifacts &middot; 5 = looks completely real</p>
  <button onclick="back()">Back</button>
</div>
<div id="done" hidden>
  <p>All images rated. Thank you!</p>
  <button onclick="save()">Download ratings</button> <button onclick="back()">Back</button>
</div>
<script>
const IMAGES = __IMAGES__;
let rater, order, scores, pos;
const $ = id => document.getElementById(id);
function hash(s){let h=2166136261;for(const c of s){h^=c.charCodeAt(0);h=Math.imul(h,16777619)}return h>>>0}
function shuffled(seed){
  let s=seed;const rnd=()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^t>>>15,t|1);
    t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296};
  const a=IMAGES.slice();
  for(let i=a.length-1;i>0;i--){const j=Math.floor(rnd()*(i+1));[a[i],a[j]]=[a[j],a[i]]}
  return a;
}
function begin(){
  rater=$("rater").value.trim().replace(/[^\\w-]/g,"_"); if(!rater) return;
  order=shuffled(hash(rater));  // each rater sees an independent order
  scores=JSON.parse(localStorage.getItem("rating:"+rater)||"{}");
  pos=order.findIndex(n=>!(n in scores)); if(pos<0) pos=order.length;
  $("start").hidden=true; show();
}
function show(){
  const done=pos>=order.length; $("task").hidden=done; $("done").hidden=!done;
  if(done) return;
  $("img").src="images/"+order[pos];
  if(pos+1<order.length) new Image().src="images/"+order[pos+1];
  const cur=scores[order[pos]];
  $("progress").textContent=`${pos+1} / ${order.length}`+(cur?`  (current: ${cur})`:"");
}
function rate(s){scores[order[pos]]=s;localStorage.setItem("rating:"+rater,JSON.stringify(scores));pos++;show()}
function back(){if(pos>0){pos--;show()}}
function save(){
  const rows=["rater,image,score"].concat(order.filter(n=>n in scores).map(n=>`${rater},${n},${scores[n]}`));
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([rows.join("\\n")+"\\n"],{type:"text/csv"}));
  a.download=`ratings_${rater}.csv`; a.click();
}
document.addEventListener("keydown",e=>{
  if(!rater||$("task").hidden) return;
  if(e.key>="1"&&e.key<="5") rate(+e.key); else if(e.key==="Backspace"){e.preventDefault();back()}
});
</script></body></html>
"""


def stage_human(args):
    out_dir = Path(args.out_dir)
    data = load_groups(out_dir)
    groups, meta = data["groups"], data["meta"]
    study_dir = out_dir / "human_study"
    if study_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{study_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(study_dir)
    (study_dir / "images").mkdir(parents=True)

    rng = random.Random(args.seed)
    per_group = args.human_per_group
    picked = {}
    for key in ("adaptive_top", "adaptive_bottom"):
        picked[key] = rng.sample(groups[key], per_group)
    taken = set(picked["adaptive_top"]) | set(picked["adaptive_bottom"])
    # Each image appears once: drop the few random-group images already shown.
    picked["random"] = rng.sample([x for x in groups["random"] if x not in taken], per_group)
    trials = [(key, name) for key in HUMAN_GROUPS for name in picked[key]]
    rng.shuffle(trials)

    key_rows, images = [], []
    for i, (group, name) in enumerate(trials, start=1):
        image = f"{i:03d}.png"
        load_panel(Path(meta["samples_dir"]) / name, meta["panel"]).save(study_dir / "images" / image)
        key_rows.append({"image": image, "group": group, "source": name})
        images.append(image)
    write_csv(study_dir / "answer_key.csv", key_rows)  # keep away from raters
    buttons = "".join(f'<button onclick="rate({s})">{s}</button>' for s in range(1, 6))
    html = RATE_HTML.replace("__BUTTONS__", buttons).replace("__IMAGES__", json.dumps(images))
    (study_dir / "rate.html").write_text(html, encoding="utf-8")
    print(f"[human] {len(trials)} images ({per_group} per group) in {study_dir}")
    print("[human] give raters images/ + rate.html only; answer_key.csv stays with you.")
    print("[human] each rater downloads ratings_<id>.csv; put them in one folder for the mos stage.")


# ---------------------------------------------------------------------- mos

def stage_mos(args):
    out_dir = Path(args.out_dir)
    study_dir = out_dir / "human_study"
    key = {row["image"]: row["group"] for row in read_csv(study_dir / "answer_key.csv")}
    paths = sorted(Path(args.ratings_dir).glob("*.csv"))
    if not paths:
        raise SystemExit(f"no rating CSVs in {args.ratings_dir}")
    ratings = {}
    for path in paths:
        for row in read_csv(path):
            image, rater, score = row["image"].strip(), row["rater"].strip(), row["score"].strip()
            if image not in key:
                raise ValueError(f"{path}: unknown image {image!r}")
            if score not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"{path}: score must be 1-5, got {score!r}")
            ratings[(rater, image)] = int(score)  # a later file overrides a duplicate
    raters = sorted({rater for rater, _ in ratings})
    per_image = {image: [] for image in key}
    for (_, image), score in ratings.items():
        per_image[image].append(score)
    counts = [len(v) for v in per_image.values()]
    if min(counts) < args.min_raters:
        short = sum(c < args.min_raters for c in counts)
        print(f"[mos] warning: {short} images have fewer than {args.min_raters} ratings")
    rated = {image: scores for image, scores in per_image.items() if scores}
    mos = {image: float(np.mean(scores)) for image, scores in rated.items()}

    rows = []
    for group in HUMAN_GROUPS:
        values = [mos[i] for i in mos if key[i] == group]
        n_ratings = sum(len(rated[i]) for i in mos if key[i] == group)
        p = "" if group == "random" else mann_whitney_p(
            values, [mos[i] for i in mos if key[i] == "random"])
        rows.append({"group": group, **describe(values), "n_ratings": n_ratings, "p_vs_random": p})
    alpha_interval = krippendorff_alpha(list(rated.values()), "interval")
    alpha_ordinal = krippendorff_alpha(list(rated.values()), "ordinal")
    write_csv(out_dir / "mos_summary.csv", rows)
    (out_dir / "mos_agreement.json").write_text(json.dumps({
        "raters": raters, "num_raters": len(raters), "num_ratings": len(ratings),
        "ratings_per_image_min": min(counts), "ratings_per_image_max": max(counts),
        "krippendorff_alpha_interval": alpha_interval,
        "krippendorff_alpha_ordinal": alpha_ordinal,
    }, indent=2) + "\n")
    print(f"[mos] {len(raters)} raters, {len(ratings)} ratings, "
          f"{min(counts)}-{max(counts)} ratings per image")
    print(f"[mos] Krippendorff's alpha: interval = {alpha_interval:.3f}, ordinal = {alpha_ordinal:.3f}")

    musiq = {}
    if (out_dir / "musiq_summary.csv").exists():
        musiq = {row["group"]: row for row in read_csv(out_dir / "musiq_summary.csv")}
    mos_rows = {row["group"]: row for row in rows}

    def cell(row):
        if row is None:
            return "–"
        return f"{float(row['mean']):.2f} [{float(row['ci95_low']):.2f}, {float(row['ci95_high']):.2f}]"

    lines = ["| Subset of FD-loss samples | MUSIQ↑ | MOS↑ |", "|---|---|---|"]
    for group, label in GROUPS:
        if group in musiq or group in mos_rows:
            lines.append(f"| {label} | {cell(musiq.get(group))} | {cell(mos_rows.get(group))} |")
    table = "\n".join(lines)
    per_group = "/".join(str(row["n"]) for row in rows)
    table += (f"\n\nMean [95% CI]. MOS: {len(raters)} raters, {per_group} images per group "
              f"(top/random/bottom), 1-5 scale; Krippendorff's alpha (interval) = {alpha_interval:.3f}.\n")
    (out_dir / "final_table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"[mos] saved {out_dir / 'final_table.md'}")


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("rank", "musiq", "human", "mos"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--samples_dir", help="FD-loss samples, e.g. .../quali/jit_l_threeway")
    parser.add_argument("--panel", type=int, default=1, help="panel of a strip image (threeway: 1 = FD-loss)")
    parser.add_argument("--checkpoint", help="AdvFD checkpoint (file or checkpoints/ dir) with fd_adv_states")
    parser.add_argument("--data_root", help="ImageNet root containing train/")
    parser.add_argument("--real_per_class", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--whiten_eps", type=float, default=1e-3, help="match --fd_adv_whiten_eps")
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--human_per_group", type=int, default=100)
    parser.add_argument("--ratings_dir", help="folder of ratings_<rater>.csv files")
    parser.add_argument("--min_raters", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = {"rank": ("samples_dir", "checkpoint", "data_root"), "mos": ("ratings_dir",)}
    missing = [f"--{name}" for name in required.get(args.stage, ()) if getattr(args, name) is None]
    if missing:
        parser.error(f"stage {args.stage} needs {' '.join(missing)}")
    {"rank": stage_rank, "musiq": stage_musiq, "human": stage_human, "mos": stage_mos}[args.stage](args)


if __name__ == "__main__":
    main()
