"""Plot train and eval loss / perplexity against tokens seen for W&B runs.

Login-node friendly (reads W&B history, no GPU):
    python plot_runs.py dvqj42sv dkyqsbe6 --out plots/wsd_vs_cosine.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help="W&B run ids")
parser.add_argument("--project", default=f"{os.environ.get('WANDB_ENTITY', 'cerulean-labs')}/{os.environ.get('WANDB_PROJECT', 'lab5-training-llama')}")
parser.add_argument("--out", default="plots/loss_vs_tokens.png")
parser.add_argument("--smooth", type=int, default=5, help="Rolling mean over this many train log points (10 steps each)")
args = parser.parse_args()

api = wandb.Api(timeout=60)
X = "train/total_tokens_seen"  # also present in eval rows (see train.py WandbTokenAxisCallback)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for run_id in args.runs:
    run = api.run(f"{args.project}/{run_id}")
    label = f"{run.name} ({run_id})"
    train = [r for r in run.scan_history(keys=[X, "train/loss"]) if r.get("train/loss") is not None]
    evals = [r for r in run.scan_history(keys=[X, "eval/loss"]) if r.get("eval/loss") is not None]
    tx = [r[X] / 1e9 for r in train]
    tl = [r["train/loss"] for r in train]
    k = args.smooth
    tl = [sum(tl[max(0, i - k + 1): i + 1]) / len(tl[max(0, i - k + 1): i + 1]) for i in range(len(tl))]
    ex = [r[X] / 1e9 for r in evals]
    el = [r["eval/loss"] for r in evals]
    axes[0, 0].plot(tx, tl, label=label)
    axes[0, 1].plot(tx, [2.718281828 ** v for v in tl], label=label)
    axes[1, 0].plot(ex, el, marker="o", ms=3, label=label)
    axes[1, 1].plot(ex, [2.718281828 ** v for v in el], marker="o", ms=3, label=label)
    print(f"{label}: {len(train)} train points, {len(evals)} evals, final eval loss {el[-1]:.3f} (ppl {2.718281828 ** el[-1]:.2f}) at {ex[-1]:.3f}B tokens")

titles = [f"Train loss (rolling mean of {args.smooth} logs)", "Train perplexity", "Eval loss", "Eval perplexity"]
for ax, title in zip(axes.flat, titles):
    ax.set_title(title)
    ax.set_xlabel("Tokens seen (B)")
    ax.grid(alpha=0.3)
    ax.legend()
for ax in axes[:, 1]:
    ax.set_yscale("log")
for ax in axes[:, 0]:  # zoom past the first ~10% (loss falls from 11.8) so the runs can be told apart
    ax.set_ylim(top=6)
for ax in axes[:, 1]:
    ax.set_ylim(top=400)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=120)
print("saved", args.out)
