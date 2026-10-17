"""Compare optimizer dynamics (train.py --dyn_every, llama_train/optim_dynamics.py) across runs, by block type and stage.

    sbatch --partition=dev --gpus-per-node=1 --cpus-per-task=4 --time=0:20:00 --output=logs/%x-%j.out --job-name=dyn-plots \
        --wrap="source ./.venv/bin/activate && python analyze_dynamics.py --out plots/dyn \
            AdamW=/work/$USER/llama_models/run-a/optim_dynamics.jsonl Muon=/work/$USER/llama_models/run-b/optim_dynamics.jsonl"

Stages from the WSD schedule (--warmup_frac / --decay_frac of --max_steps): warmup, stable, decay. Writes
  <out>/<metric>.png        one panel per run, a line per block type (y log-scale for magnitudes), stages shaded
  <out>/layers_<metric>_<type>.png   layer x step heatmaps, one row per run
  <out>/summary.md          medians per stage, run and block type for every metric
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from llama_train.optim_dynamics import load_dynamics  # noqa: E402

METRICS = [  # (metric, log y)
    ("upd_rel", True), ("upd_rms", True), ("grad_norm", True), ("w_norm", True), ("w_sigma1", True),
    ("upd_grad_align", False), ("upd_cos_prev", False), ("grad_cos_prev", False),
    ("dir_sharpness", False), ("D1", False), ("curvature_K", True),
    ("upd_spec_rel", True), ("upd_srank", True), ("w_srank", True), ("upd_row_cv", False),
    ("upd_erank_frac", False), ("delta_erank_frac", False), ("delta_rel", True),
]
TYPE_ORDER = ["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down", "norm"]
HEATMAPS = [("upd_rel", True), ("upd_erank_frac", False), ("upd_grad_align", False)]
HEATMAP_TYPES = ["attn_q", "attn_o", "mlp_down"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="NAME=path/to/optim_dynamics.jsonl")
ap.add_argument("--out", default="plots/dyn")
ap.add_argument("--max_steps", type=int, default=11444, help="Run length in steps (stages; 6B proxy: 11,444)")
ap.add_argument("--warmup_frac", type=float, default=0.02)
ap.add_argument("--decay_frac", type=float, default=0.1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

warm_end = int(a.warmup_frac * a.max_steps)
decay_start = a.max_steps - int(a.decay_frac * a.max_steps)


def stage(step):
    return "warmup" if step <= warm_end else "stable" if step <= decay_start else "decay"


runs = {}
for spec in a.runs:
    name, path = spec.split("=", 1)
    df = load_dynamics(path)
    df["stage"] = df.step.map(stage)
    runs[name] = df
    print(f"{name}: {df.step.nunique()} measurements, steps {df.step.min()}-{df.step.max()}", flush=True)


def shade(ax):
    ax.axvspan(0, warm_end, color="0.9", lw=0)
    ax.axvspan(decay_start, a.max_steps, color="0.9", lw=0)


types_seen = [t for t in TYPE_ORDER if any((df.type == t).any() for df in runs.values())]
colors = {t: plt.cm.tab10(i) for i, t in enumerate(types_seen)}

# 1. per metric: one panel per run, a line per block type, plus the global value dashed
for metric, logy in METRICS:
    if not any((df.metric == metric).any() for df in runs.values()):
        continue
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 3.8), sharey=True, squeeze=False)
    for ax, (name, df) in zip(axes[0], runs.items()):
        shade(ax)
        sub = df[(df.metric == metric) & (df.scope == "type")]
        for t in types_seen:
            s = sub[sub.type == t].sort_values("step")
            if len(s):
                y = s.value.abs() if logy else s.value
                ax.plot(s.step, y, color=colors[t], lw=1.2, label=t)
        g = df[(df.metric == metric) & (df.scope == "global")].sort_values("step")
        if len(g):
            ax.plot(g.step, g.value.abs() if logy else g.value, color="k", ls="--", lw=1, label="global")
        if logy:
            ax.set_yscale("log")
        if metric in ("dir_sharpness", "D1"):  # noisy, and huge where the LR is ~0: show the central range
            vals = sub.value[np.isfinite(sub.value)]
            if len(vals):
                lo, hi = np.percentile(vals, [5, 95])
                pad = 0.1 * (hi - lo + 1e-12)
                ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(name)
        ax.set_xlabel("step")
    axes[0][0].set_ylabel(("|" + metric + "|") if logy else metric)
    axes[0][-1].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(f"{metric} by block type (grey: warmup / decay)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{metric}.png"), dpi=110)
    plt.close(fig)

# 2. layer x step heatmaps, one row per run, a column per block type
for metric, logy in HEATMAPS:
    fig, axes = plt.subplots(len(runs), len(HEATMAP_TYPES), figsize=(4.5 * len(HEATMAP_TYPES), 2.6 * len(runs)), squeeze=False)
    for r, (name, df) in enumerate(runs.items()):
        for c, t in enumerate(HEATMAP_TYPES):
            ax = axes[r][c]
            sub = df[(df.scope == "layer") & (df.metric == metric) & (df.type == t)]
            if not len(sub):
                ax.set_axis_off()
                continue
            grid = sub.pivot_table(index="layer", columns="step", values="value")
            data = np.log10(grid.abs().values + 1e-30) if logy else grid.values
            im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                           extent=[grid.columns.min(), grid.columns.max(), -0.5, len(grid.index) - 0.5])
            ax.set_title(f"{name}: {t}", fontsize=9)
            ax.set_ylabel("layer")
            plt.colorbar(im, ax=ax)
    fig.suptitle(f"{'log10 ' if logy else ''}{metric} per layer")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"layers_{metric}.png"), dpi=110)
    plt.close(fig)

# 3. medians per stage x run x block type
lines = [f"# Optimizer dynamics: medians per stage\n",
         f"Stages: warmup steps 1-{warm_end}, stable {warm_end + 1}-{decay_start}, decay {decay_start + 1}-{a.max_steps}. "
         f"Runs: {', '.join(runs)}. Block types: rows; global = the whole model as one vector.\n"]
rows = []
for name, df in runs.items():
    sub = df[df.scope.isin(["type", "global"])].copy()
    sub["block"] = np.where(sub.scope == "global", "global", sub.type)
    rows.append(sub.groupby(["metric", "block", "stage"]).value.median().rename(name))
med = pd.concat(rows, axis=1)
fmt = lambda v: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3g}"
for metric, _ in METRICS:
    if metric not in med.index.get_level_values(0):
        continue
    m = med.loc[metric]
    lines.append(f"\n## {metric}\n")
    lines.append("| block | " + " | ".join(f"{r} {s}" for r in runs for s in ("warmup", "stable", "decay")) + " |")
    lines.append("|---|" + "---|" * (3 * len(runs)))
    for blk in ["global"] + types_seen:
        if blk not in m.index.get_level_values(0):
            continue
        cells = [fmt(m.loc[blk].get(r, {}).get(s) if s in m.loc[blk].index else None) for r in runs for s in ("warmup", "stable", "decay")]
        lines.append(f"| {blk} | " + " | ".join(cells) + " |")
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("wrote", a.out)
