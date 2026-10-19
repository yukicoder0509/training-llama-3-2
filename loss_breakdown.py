"""Held-out loss of saved models split by target-token frequency and by domain, with paired bootstrap CIs vs a baseline.

    sbatch --account=ACD115198 --partition=dev --gpus-per-node=1 --cpus-per-task=12 --time=1:00:00 \
        --output=logs/%x-%j.out --job-name=loss-breakdown --wrap="source ./.venv/bin/activate && python loss_breakdown.py \
        --out results/loss_breakdown_proxy.md NorMuon=/work/$USER/llama_models/run-a SOAP=/work/$USER/llama_models/run-b"

Data: val.bin (the first 10k of the 50k held-out docs, 15.2M tokens; prepare_data.py: BOS text EOS per doc), each doc
scored in --window-token windows (scoring.score_stream). Frequency = count of the target token in the first
--count_tokens tokens of train.bin (shuffled docs, so a prefix is a uniform sample), in buckets per 1B tokens. Domain =
the doc's source / topic cell (scoring.group_name), from its rank as in score_docs.py. Each table: mean loss per bucket
for the first model, then the others' difference to it with a 95% bootstrap CI over docs (paired: same docs).
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from prepare_data import NUM_EVAL_DOCS, SEED
from scoring import group_name, parse_dir, score_stream

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="NAME=model dir; the first is the baseline")
ap.add_argument("--data", default=os.path.expandvars("/work/$USER/dolma3_llama"))
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--count_tokens", type=float, default=1e9)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
models = [m.split("=", 1) for m in args.models]

# Unigram counts of train.bin's prefix (cached)
cpath = os.path.join(args.data, f"unigram_first{args.count_tokens:.0e}.npy")
if not os.path.exists(cpath):
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint32, mode="r")
    counts = np.zeros(128256, np.int64)
    for i in range(0, int(args.count_tokens), 100_000_000):
        counts += np.bincount(train[i:i + 100_000_000], minlength=len(counts))
    np.save(cpath, counts)
counts = np.load(cpath) * (1e9 / args.count_tokens)

# val.bin docs (split at BOS) and their domains
val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint32, mode="r")
starts = np.flatnonzero(val == 128000)
doc_start = np.append(starts, len(val))
n_docs = len(starts)
ranks = []
with open(os.path.join(args.data, "eval_docs.jsonl")) as f:
    for line in f:
        ranks.append(json.loads(line)["rank"])
        if len(ranks) == n_docs:
            break
dc = json.load(open(os.path.join(args.data, "doc_counts.json")))
files, per_file = dc["files"], dc["counts"]
n = sum(per_file)
perm = np.random.default_rng(SEED).permutation(n)  # index -> rank is perm's inverse: the doc with rank r is perm[r]
file_of = np.searchsorted(np.cumsum(per_file), perm[np.array(ranks)], side="right")
del perm
groups = np.array([group_name(*parse_dir(files[i].split("/")[1])[:2]) for i in file_of])
print(f"{n_docs:,} val docs, {len(val):,} tokens; {len(set(groups))} domain groups", flush=True)

# Per-token losses
from transformers import AutoModelForCausalLM  # noqa: E402

tgt = np.append(np.asarray(val[1:]).astype(np.int64), 0)
doc_of = np.repeat(np.arange(n_docs), np.diff(doc_start))
losses = {}
for name, path in models:
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="flash_attention_3", dtype=torch.bfloat16).cuda().eval()
    tl = torch.full((len(val),), float("nan"), device="cuda")
    s, p = score_stream(model, val, doc_start, args.window, log=lambda m: None, tok_loss=tl)
    losses[name] = tl.cpu().numpy()
    print(f"{name}: mean loss {s.sum() / p.sum():.4f} over {int(p.sum()):,} predictions", flush=True)
    del model
    torch.cuda.empty_cache()
valid = ~np.isnan(losses[models[0][0]])

FREQ_EDGES = [0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf]
FREQ_NAMES = ["<1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7", ">=1e7"]
freq_bucket = np.digitize(counts[tgt], FREQ_EDGES[1:-1])
dom_names = sorted(set(groups), key=lambda g: -np.sum(np.diff(doc_start)[groups == g]))
dom_bucket = np.array([dom_names.index(g) for g in groups])[doc_of]
sources = np.array([g.split(":")[0] if ":" in g else "other" for g in groups])  # cc / pdf / code / other sources
src_names = sorted(set(sources), key=lambda g: -np.sum(np.diff(doc_start)[sources == g]))
src_bucket = np.array([src_names.index(g) for g in sources])[doc_of]


def table(title, bucket, names, min_share=0.0):
    k = len(names)
    m = valid
    idx = doc_of[m] * k + bucket[m]
    cnt = np.bincount(idx, minlength=n_docs * k).reshape(n_docs, k)
    sums = {nm: np.bincount(idx, weights=losses[nm][m], minlength=n_docs * k).reshape(n_docs, k) for nm, _ in models}
    base = models[0][0]
    rng = np.random.default_rng(0)
    boot = rng.integers(0, n_docs, (args.boot, n_docs))
    lines = [f"\n### {title}\n", "| bucket | share of predictions | " + base + " | "
             + " | ".join(f"Δ {nm} (95% CI)" for nm, _ in models[1:]) + " |", "|---|---|---|" + "---|" * (len(models) - 1)]
    for j, b in enumerate(names):
        c = cnt[:, j]
        if c.sum() < max(1, min_share * cnt.sum()):
            continue
        row = [b, f"{c.sum() / cnt.sum():.1%}", f"{sums[base][:, j].sum() / c.sum():.4f}"]
        for nm, _ in models[1:]:
            d = sums[nm][:, j] - sums[base][:, j]
            bs = d[boot].sum(1) / np.maximum(c[boot].sum(1), 1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.append(f"{d.sum() / c.sum():+.4f} ({lo:+.4f}, {hi:+.4f})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


out = [f"# Loss breakdown (val.bin, {n_docs:,} docs, window {args.window})\n",
       "Models: " + ", ".join(f"{nm} = `{p}`" for nm, p in models),
       f"\nFrequency: target token's count per 1B train tokens (from the first {args.count_tokens:.0e} of train.bin).",
       table("By target-token frequency", freq_bucket, FREQ_NAMES),
       table("By source", src_bucket, src_names),
       table("By domain (>= 0.2% of predictions)", dom_bucket, dom_names, 0.002),
       table("All", np.zeros_like(freq_bucket), ["all"])]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
