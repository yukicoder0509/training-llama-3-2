"""Optimizer dynamics (train.py --dyn_every) against the peak LR: an LR sweep per optimizer, read as metric vs LR / LR_base.

    sbatch --partition=dev --gpus-per-node=1 --cpus-per-task=4 --time=0:30:00 --output=logs/%x-%j.out --job-name=dynlr-plots \
        --wrap="source ./.venv/bin/activate && python analyze_lr_dynamics.py --out plots/dynlr \
            AdamW:1e-3:1e-3=/work/$USER/llama_models/run-base/optim_dynamics.jsonl@logs/run-base-123.out \
            AdamW:3e-3:1e-3=/work/$USER/llama_models/run-hi/optim_dynamics.jsonl@logs/run-hi-124.out ..."

Each run is OPT:LR:LR_BASE=jsonl[@slurm.out] (the slurm log gives the final eval loss). Windows (steps): early
(--early lo:hi, after warmup: can a metric flag a bad LR early?), stable (--stable_from to decay start - 50), decay.
Writes
  <out>/lr_<metric>.png   global median per window vs LR / LR_base (log x), a line per optimizer; the top-left panel
                          of lr_eval.png is the final eval loss
  <out>/lr_summary.md     final eval and global medians per optimizer x LR x window; per-block-type medians (stable);
                          Spearman rank correlation of each metric with log LR per optimizer and window
"""

import argparse
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from llama_train.optim_dynamics import load_dynamics  # noqa: E402

METRICS = [  # (metric, log y)
    ("upd_rel", True), ("upd_rms", True), ("grad_norm", True), ("w_norm", True), ("w_sigma1", True),
    ("upd_grad_align", False), ("upd_cos_prev", False), ("grad_cos_prev", False),
    ("dir_sharpness", True), ("D1", False), ("curvature_K", True),
    ("upd_spec_rel", True), ("upd_srank", True), ("w_srank", True), ("upd_row_cv", False),
    ("upd_erank_frac", False), ("delta_erank_frac", False), ("delta_rel", True),
    ("probe_sharpness", True), ("probe_D1", False), ("probe_slope", False), ("probe_dloss", False),  # --dyn_probe
]
TYPE_ORDER = ["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down", "norm"]
TYPE_METRICS = ["upd_rel", "dir_sharpness", "probe_sharpness", "grad_cos_prev", "upd_erank_frac", "w_sigma1"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="OPT:LR:LR_BASE=path/to/optim_dynamics.jsonl[@slurm.out]")
ap.add_argument("--out", default="plots/dynlr")
ap.add_argument("--max_steps", type=int, default=11444, help="Run length in steps (6B proxy: 11,444)")
ap.add_argument("--decay_frac", type=float, default=0.1)
ap.add_argument("--early", default="500:1500", help="early window lo:hi in steps")
ap.add_argument("--stable_from", type=int, default=500)
ap.add_argument("--series", action="append", default=[],
                help="NAME:OPT=lr,lr*,lr,... (repeatable): an LR series around the centre LR marked * (default the middle one), e.g. geometric or arithmetic. "
                     "Writes <out>/series_<NAME>.md and .png: stable medians and final eval by step index -k..+k")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

decay_start = a.max_steps - int(a.decay_frac * a.max_steps)
e_lo, e_hi = map(int, a.early.split(":"))
WINDOWS = {"early": (e_lo, e_hi), "stable": (a.stable_from, decay_start - 50), "decay": (decay_start + 1, a.max_steps)}


def final_eval(log):
    vals = re.findall(r"'eval_loss': '([0-9.naif]+)'", open(log).read()) if log and os.path.exists(log) else []
    return float(vals[-1]) if vals else math.nan


rows = []  # opt, lr, ratio, window, scope, type, metric, median
evals = []
for spec in a.runs:
    key, src = spec.split("=", 1)
    opt, lr, base = key.split(":")
    path, _, log = src.partition("@")
    ratio = float(lr) / float(base)
    df = load_dynamics(path)
    df = df[df.scope.isin(["global", "type"])]
    for w, (lo, hi) in WINDOWS.items():
        sub = df[(df.step >= lo) & (df.step <= hi)]
        med = sub.groupby(["scope", "type", "metric"], dropna=False).value.median().reset_index()
        for r in med.itertuples():
            rows.append((opt, float(lr), ratio, w, r.scope, r.type, r.metric, r.value))
    evals.append((opt, float(lr), ratio, final_eval(log)))
    print(f"{opt} {lr}: {df.step.nunique()} measurements, final eval {evals[-1][3]}", flush=True)

med = pd.DataFrame(rows, columns=["opt", "lr", "ratio", "window", "scope", "type", "metric", "value"])
ev = pd.DataFrame(evals, columns=["opt", "lr", "ratio", "eval"])
opts = list(dict.fromkeys(ev.opt))
colors = {o: plt.cm.tab10(i) for i, o in enumerate(opts)}
glob = med[med.scope == "global"]


def lines(ax, data, logy):
    for o in opts:
        s = data[data.opt == o].sort_values("ratio")
        if len(s):
            y = s.value.abs() if logy else s.value
            ax.plot(s.ratio, y, "o-", color=colors[o], label=o, ms=4)
    ax.set_xscale("log", base=2)
    ax.axvline(1, color="0.6", lw=0.8, ls=":")
    if logy:
        ax.set_yscale("log")


# 1. final eval vs LR, and per metric: a panel per window
fig, ax = plt.subplots(figsize=(5.5, 3.8))
lines(ax, ev.rename(columns={"eval": "value"}), False)
ax.set_xlabel("LR / tuned LR")
ax.set_ylabel("final eval loss")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "lr_eval.png"), dpi=110)
plt.close(fig)

for metric, logy in METRICS:
    sub = glob[glob.metric == metric]
    if not len(sub):
        continue
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5.2 * len(WINDOWS), 3.8), squeeze=False)
    for ax, w in zip(axes[0], WINDOWS):
        lines(ax, sub[sub.window == w], logy)
        ax.set_title(f"{w} {WINDOWS[w]}")
        ax.set_xlabel("LR / tuned LR")
    axes[0][0].set_ylabel(f"global {'|' + metric + '|' if logy else metric} (median)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"lr_{metric}.png"), dpi=110)
    plt.close(fig)


# 2. summary tables
def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx, ry = pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def fmt(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.3g}"


out = ["# Optimizer dynamics vs peak LR\n",
       f"Windows (steps): " + ", ".join(f"{w} {lo}-{hi}" for w, (lo, hi) in WINDOWS.items())
       + ". Values: medians of the global metric (whole model as one vector) over the window.\n"]
out += ["## Final eval loss\n", "| Optimizer | LR (×base) | eval |", "|---|---|---|"]
for r in ev.sort_values(["opt", "ratio"]).itertuples():
    out.append(f"| {r.opt} | {r.lr:g} (×{r.ratio:g}) | {r.eval:.4f} |")

out += ["\n## Spearman ρ(metric, log LR) per optimizer\n",
        "ρ over the optimizer's LRs (5 points): ±1 = monotone in LR.\n"]
for w in WINDOWS:
    out += [f"### {w}\n", "| Metric | " + " | ".join(opts) + " |", "|---" * (len(opts) + 1) + "|"]
    for metric, _ in METRICS:
        cells = []
        for o in opts:
            s = glob[(glob.metric == metric) & (glob.window == w) & (glob.opt == o)]
            cells.append(fmt(spearman(np.log(s.lr.values), s.value.values)))
        out.append(f"| {metric} | " + " | ".join(cells) + " |")
    out.append("")

for w in WINDOWS:
    out += [f"## Global medians, {w}\n"]
    sub = glob[glob.window == w]
    for o in opts:
        so = sub[sub.opt == o]
        lrs = sorted(so.lr.unique())
        out += [f"**{o}**\n", "| Metric | " + " | ".join(f"{l:g}" for l in lrs) + " |", "|---" * (len(lrs) + 1) + "|"]
        for metric, _ in METRICS:
            vals = [so[(so.metric == metric) & (so.lr == l)].value for l in lrs]
            out.append(f"| {metric} | " + " | ".join(fmt(v.iloc[0]) if len(v) else "–" for v in vals) + " |")
        out.append("")

out += ["## Per block type, stable\n"]
typ = med[(med.scope == "type") & (med.window == "stable")]
for metric in TYPE_METRICS:
    out += [f"### {metric}\n"]
    for o in opts:
        so = typ[(typ.opt == o) & (typ.metric == metric)]
        lrs = sorted(so.lr.unique())
        types = [t for t in TYPE_ORDER if (so.type == t).any()]
        out += [f"**{o}**\n", "| LR | " + " | ".join(types) + " |", "|---" * (len(types) + 1) + "|"]
        for l in lrs:
            vals = [so[(so.lr == l) & (so.type == t)].value for t in types]
            out.append(f"| {l:g} | " + " | ".join(fmt(v.iloc[0]) if len(v) else "–" for v in vals) + " |")
        out.append("")

with open(os.path.join(a.out, "lr_summary.md"), "w") as f:
    f.write("\n".join(out) + "\n")


# 3. LR series: each optimizer's LRs indexed -k..+k around the centre, metric change vs the centre per step, and the
# high / low asymmetry |Δ(+i)| / |Δ(-i)| (1 = symmetric in the series' own spacing)
def close(x, y):
    return abs(x - y) <= 1e-3 * abs(y)


series = {}  # name -> {opt: [lr, ...]}
for s in a.series:
    key, lrs = s.split("=", 1)
    name, opt = key.split(":")
    parts = lrs.split(",")
    centre = next((i for i, l in enumerate(parts) if l.endswith("*")), len(parts) // 2)
    series.setdefault(name, {})[opt] = ([float(l.rstrip("*")) for l in parts], centre)

stable = glob[glob.window == "stable"]
for name, by_opt in series.items():
    out = [f"# LR series: {name}\n",
           "Stable-window global medians and final eval, by step index around the centre LR (0). Δ% = change vs the "
           "centre; asym i = |Δ(+i)| / |Δ(−i)| (> 1: the high side moves more; for a constant-ratio series the metric Δ is in log v, so a power law in LR gives 1).\n"]
    curves = {}  # opt -> {metric: (idx, values)}
    for o, (lrs, k) in by_opt.items():
        idx = list(range(-k, len(lrs) - k))
        na = min(k, len(lrs) - 1 - k)  # steps present on both sides

        def val(metric, l):
            if metric == "eval":
                v = [r.eval for r in ev.itertuples() if r.opt == o and close(r.lr, l)]
            else:
                v = list(stable[(stable.opt == o) & (stable.metric == metric) & np.isclose(stable.lr, l, rtol=1e-3)].value)
            return v[0] if v else math.nan

        ratios = [lrs[i + 1] / lrs[i] for i in range(len(lrs) - 1)]
        steps = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        geo = max(ratios) / min(ratios) < 1.05  # constant ratio: compare metrics in log space
        out += [f"## {o}\n", f"LRs {', '.join(f'{l:g}' for l in lrs)} (ratios {', '.join(f'{r:.3g}' for r in ratios)}; "
                f"steps {', '.join(f'{d:.3g}' for d in steps)}; metric asym on {'log v' if geo else 'v'})\n",
                "| Metric | " + " | ".join(f"{i:+d} ({l:g})" for i, l in zip(idx, lrs)) + " | "
                + " | ".join(f"asym {i}" for i in range(1, na + 1)) + " |",
                "|---" * (len(lrs) + na + 1) + "|"]
        curves[o] = {}
        for metric in ["eval"] + [m for m, _ in METRICS]:
            v = [val(metric, l) for l in lrs]
            c = v[k]
            curves[o][metric] = (idx, v)
            if metric == "eval":
                cells = [f"{x:.4f}" if i == 0 else f"{x:.4f} ({x - c:+.4f})" for i, x in zip(idx, v)]
                asym = [abs(v[k + i] - c) / abs(v[k - i] - c) if v[k - i] != c else math.nan for i in range(1, na + 1)]
            else:
                cells = [fmt(x) if i == 0 else f"{fmt(x)} ({100 * (x - c) / abs(c):+.1f}%)" for i, x in zip(idx, v)]
                if geo:  # a power law v ∝ LR^p is symmetric in log v
                    lg = [math.log(x / c) if x / c > 0 else math.nan for x in v]
                    asym = [abs(lg[k + i]) / abs(lg[k - i]) if lg[k - i] else math.nan for i in range(1, na + 1)]
                else:
                    asym = [abs(v[k + i] - c) / abs(v[k - i] - c) if v[k - i] != c else math.nan for i in range(1, na + 1)]
            out.append(f"| {metric} | " + " | ".join(cells) + " | " + " | ".join(fmt(x) for x in asym) + " |")
        out.append("")
    with open(os.path.join(a.out, f"series_{name}.md"), "w") as f:
        f.write("\n".join(out) + "\n")

    shown = ["eval"] + [m for m, _ in METRICS]
    ncol = 4
    fig, axes = plt.subplots(-(-len(shown) // ncol), ncol, figsize=(4.2 * ncol, 3.0 * -(-len(shown) // ncol)), squeeze=False)
    for ax, metric in zip(axes.flat, shown):
        for o, cv in curves.items():
            idx, v = cv[metric]
            v = np.array(v)
            if metric != "eval":
                c = v[idx.index(0)]
                v = 100 * (v - c) / abs(c)
            ax.plot(idx, v, "o-", color=colors[o], label=o, ms=4)
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        ax.set_title(metric if metric == "eval" else f"{metric} (Δ% vs centre)", fontsize=9)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"LR series: {name} (x = step index, 0 = centre LR)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"series_{name}.png"), dpi=100)
    plt.close(fig)
print(f"wrote {a.out}", flush=True)
