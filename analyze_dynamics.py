"""Compare optimizer dynamics (train.py --dyn_every, llama_train/optim_dynamics.py) across runs, by block type and stage.

    sbatch --partition=dev --gpus-per-node=1 --cpus-per-task=4 --time=0:20:00 --output=logs/%x-%j.out --job-name=dyn-plots \
        --wrap="source ./.venv/bin/activate && python analyze_dynamics.py --out plots/dyn \
            AdamW=/work/$USER/llama_models/run-a/optim_dynamics.jsonl Muon=/work/$USER/llama_models/run-b/optim_dynamics.jsonl"

Stages from the WSD schedule (--warmup_frac / --decay_frac of --max_steps): warmup, stable, decay. Writes
  <out>/<metric>.png        one panel per run, a line per block type (y log-scale for magnitudes), stages shaded
  <out>/layers_<metric>_<type>.png   layer x step heatmaps, one row per run
  <out>/summary.md          medians per stage, run and block type for every metric
  <out>/phases.md           medians over step windows (--gate, stable, decay) incl. overshoot D1/grad_norm and the
                            fraction of D1 > 0, deltas vs --baseline; with --logs NAME=slurm.out (one per run) also eval losses
  <out>/gns.md              (--dyn_noise runs) gradient noise scale per --gns_window steps and block type, half-window check
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
ap.add_argument("--gate", default="500:1500", help="phases.md: 'gate' window lo:hi in steps (mechanism checks)")
ap.add_argument("--stable_from", type=int, default=500, help="phases.md: the stable window runs from here to decay start - 50")
ap.add_argument("--baseline", default=None, help="phases.md: run NAME to subtract")
ap.add_argument("--logs", action="append", default=[], help="NAME=slurm log for eval losses (phases.md); repeat per run")
ap.add_argument("--tokens_per_step", type=int, default=262144, help="eval steps from tokens (proxy: 262144 at batch 128, 524288 at 256)")
ap.add_argument("--noise", type=float, default=0.004, help="flag |final − baseline| below this")
ap.add_argument("--gns_window", type=int, default=500, help="gns.md (--dyn_noise runs): window in steps for B_noise")
ap.add_argument("--gns_tol", type=float, default=0.3, help="gns.md: the two half-window estimates must agree within this")
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

# 4. phases.md: medians over fixed step windows (gate / stable / decay), overshoot D1 / grad_norm and the fraction of
# D1 > 0, deltas vs --baseline, and eval losses from the Slurm logs. dir_sharpness / curvature_K are left out on purpose:
# both carry a batch-noise bias ~ 1/LR (they rise in the decay for that reason alone).
g0, g1 = (int(x) for x in a.gate.split(":"))
windows = {"gate": (g0, g1), "stable": (a.stable_from, decay_start - 50), "decay": (decay_start + 1, a.max_steps)}
PHASE_METRICS = ["grad_cos_prev", "overshoot_norm", "overshoot", "upd_grad_align", "D1>0", "upd_rel", "upd_row_cv", "upd_erank_frac", "delta_erank_frac", "w_srank"]


def phase_values(df):
    """(metric, block, window) -> median; overshoot = D1 / grad_norm per measurement, D1>0 = fraction of measurements."""
    sub = df[df.scope.isin(["type", "global"])].copy()
    sub["block"] = np.where(sub.scope == "global", "global", sub.type)
    wide = sub.pivot_table(index=["block", "step"], columns="metric", values="value")
    if "D1" in wide and "grad_norm" in wide:
        wide["overshoot"] = wide["D1"] / wide["grad_norm"]
        if "upd_grad_align" in wide:  # D1 / grad_norm is cos(g_next, update): it also rises when the update follows -g
            wide["overshoot_norm"] = wide["overshoot"] / wide["upd_grad_align"]  # more closely; normalize by alignment
        wide["D1>0"] = (wide["D1"] > 0).astype(float).where(wide["D1"].notna())
    out = {}
    steps = wide.index.get_level_values("step")
    for w, (lo, hi) in windows.items():
        part = wide[(steps >= lo) & (steps <= hi)]
        agg = part.groupby(level="block").agg(lambda s: s.mean() if s.name == "D1>0" else s.median())
        for m in PHASE_METRICS:
            if m in agg:
                for blk, v in agg[m].items():
                    out[(m, blk, w)] = v
    return out


def evals(path):
    """[(step, eval_loss)] from a train.py Slurm log; steps from total_tokens_seen / --tokens_per_step."""
    import re
    seen = {}
    for m in re.finditer(r"'eval_loss': '([0-9.]+)'.{0,300}?'total_tokens_seen': ([0-9]+)", open(path, errors="replace").read()):
        seen[round(int(m.group(2)) / a.tokens_per_step)] = float(m.group(1))
    return sorted(seen.items())


pv = {name: phase_values(df) for name, df in runs.items()}
base = a.baseline
plines = [f"# Optimizer dynamics by phase\n",
          f"Windows (steps): " + ", ".join(f"{w} {lo}-{hi}" for w, (lo, hi) in windows.items())
          + f". Medians, except D1>0 (fraction of measurements). overshoot = D1 / grad_norm = cos(next gradient, update) "
          f"(> 0: the step went past the line minimum); it grows with the update's alignment to -g too, so compare "
          f"overshoot_norm = overshoot / upd_grad_align. Use global metrics for gates: oscillation is a whole-network "
          f"property. Baseline: {base or 'none'}.\n"]
logs = dict(s.split("=", 1) for s in a.logs)
if logs:
    plines += ["\n## Eval loss\n", "| run | final | pre-decay (step) | decay gain | final − baseline |", "|---|---|---|---|---|"]
    ev = {n: evals(p) for n, p in logs.items()}
    fin = {n: e[-1][1] for n, e in ev.items() if e}
    for n, e in ev.items():
        if not e:
            plines.append(f"| {n} | no evals | | | |")
            continue
        pre = [x for x in e if x[0] <= decay_start]
        pre_s, pre_v = pre[-1] if pre else (None, float("nan"))
        d = "" if base not in fin or n == base else f"{fin[n] - fin[base]:+.4f}" + (" (< noise)" if abs(fin[n] - fin[base]) < a.noise else "")
        plines.append(f"| {n} | {fin[n]:.4f} | {pre_v:.4f} ({pre_s}) | {pre_v - fin[n]:.4f} | {d} |")
blocks = ["global"] + types_seen
for m in PHASE_METRICS:
    if not any(k[0] == m for v in pv.values() for k in v):
        continue
    plines += [f"\n## {m}\n", "| block | " + " | ".join(f"{r} {w}" for r in runs for w in windows) + " |",
               "|---|" + "---|" * (len(runs) * len(windows))]
    for blk in blocks:
        cells = [fmt(pv[r].get((m, blk, w))) for r in runs for w in windows]
        if any(cells):
            plines.append(f"| {blk} | " + " | ".join(cells) + " |")
    if base in pv and len(runs) > 1:
        others = [r for r in runs if r != base]
        plines += [f"\nΔ vs {base}:\n", "| block | " + " | ".join(f"{r} {w}" for r in others for w in windows) + " |",
                   "|---|" + "---|" * (len(others) * len(windows))]
        for blk in blocks:
            cells = []
            for r in others:
                for w in windows:
                    x, b = pv[r].get((m, blk, w)), pv[base].get((m, blk, w))
                    cells.append("" if x is None or b is None else f"{x - b:+.3g}")
            if any(cells):
                plines.append(f"| {blk} | " + " | ".join(cells) + " |")
open(os.path.join(a.out, "phases.md"), "w").write("\n".join(plines) + "\n")


# --dyn_noise: B_noise per window = mean(gns_trS) / mean(gns_G2) (never the mean of per-point gns_B: one G2 can be ~0 or
# < 0), in tokens, with the same ratio over each half of the window as a precision check (trust if within --gns_tol).
def gns_tables(name, df):
    sub = df[df.metric.isin(["gns_trS", "gns_G2", "gns_batch"]) & df.scope.isin(["global", "type"])].copy()
    if sub.empty:
        return []
    sub["block"] = np.where(sub.scope == "global", "global", sub.type)
    wide = sub.pivot_table(index=["block", "step"], columns="metric", values="value")
    batch = wide.loc["global", "gns_batch"] if "gns_batch" in wide else None
    steps = sorted(wide.index.get_level_values("step").unique())
    edges = list(range(0, max(steps) + a.gns_window, a.gns_window))
    lines = [f"\n## {name}\n", "| steps | block | n | B_noise (tokens) | 1st half | 2nd half | halves agree | train batch (tokens) | batch / B |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for blk in ["global"] + [t for t in types_seen if t in wide.index.get_level_values("block")]:
            part = wide.loc[blk]
            part = part[(part.index > lo) & (part.index <= hi)].dropna(subset=["gns_trS", "gns_G2"])
            if len(part) < 2:
                continue
            ratio = lambda p: p.gns_trS.mean() / p.gns_G2.mean() if p.gns_G2.mean() > 0 else float("nan")
            B, h1, h2 = ratio(part), ratio(part.iloc[:len(part) // 2]), ratio(part.iloc[len(part) // 2:])
            ok = math.isfinite(h1 * h2) and abs(h1 - h2) <= a.gns_tol * (h1 + h2) / 2
            bt = batch[(batch.index > lo) & (batch.index <= hi)].mean() if batch is not None else float("nan")
            lines.append(f"| {lo + 1}-{hi} | {blk} | {len(part)} | {B:,.0f} | {h1:,.0f} | {h2:,.0f} | {'yes' if ok else 'NO'} "
                         f"| {bt:,.0f} | {bt / B:.2f} |")
    return lines


glines = sum((gns_tables(n, df) for n, df in runs.items()), [])
if glines:
    open(os.path.join(a.out, "gns.md"), "w").write(
        f"# Gradient noise scale B_noise = tr(Sigma) / |G|^2 (McCandlish et al. 2018, B_simple)\n\n"
        f"Per {a.gns_window}-step window: mean(gns_trS) / mean(gns_G2), in tokens (treats tokens as independent: compare "
        f"with the training batch in the same unit, prefer trends). Halves agree: within {a.gns_tol:.0%}. batch / B < 1: "
        f"a larger batch is nearly free; > 1: tokens are wasted on noise reduction.\n" + "\n".join(glines) + "\n")
print("wrote", a.out)
