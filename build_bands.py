"""Write a perplexity-filtered copy of train.bin: drop each group's lowest / highest mean-loss docs (score_bin.py scores).

    sbatch --gpus-per-node=1 --cpus-per-task=12 --time=1:00:00 --job-name=build-bands --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python build_bands.py --drop_high 5"

Cut points are percentiles of mean_loss within each group (cc:<topic>, pdf:<topic>, code:<language>, other sources;
weighted by docs), so the group mix stays as it was, or over all docs with --global_cut. Kept docs keep their order.
Out dir (default <data>_band_lo<L>_hi<H>[_global]): train.bin, val.bin (symlink to the original: same eval for all),
meta.json (original + the filter and what it kept), band_cuts.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.expandvars("/work/$USER/dolma3_llama"))
ap.add_argument("--scores", default=os.path.expandvars("/work/$USER/dolma3_llama/scores/scorer-muon0.01-adamw3e-3-6b/train_scores.parquet"))
ap.add_argument("--drop_low", type=float, default=0.0, help="Drop the lowest-loss pct of docs in each group")
ap.add_argument("--drop_high", type=float, default=0.0, help="Drop the highest-loss pct of docs in each group")
ap.add_argument("--global_cut", action="store_true", help="Percentiles over all docs instead of per group")
ap.add_argument("--out_dir", default=None)
args = ap.parse_args()
out_dir = args.out_dir or f"{args.data}_band_lo{args.drop_low:g}_hi{args.drop_high:g}{'_global' if args.global_cut else ''}"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_parquet(args.scores)
key = np.zeros(len(df), np.int64) if args.global_cut else df.group.factorize()[0]
keep = np.ones(len(df), bool)
cuts = []
for g, idx in pd.Series(np.arange(len(df))).groupby(key).groups.items():
    idx = np.asarray(idx)
    x = df.mean_loss.to_numpy()[idx]
    lo = np.quantile(x, args.drop_low / 100) if args.drop_low else -np.inf
    hi = np.quantile(x, 1 - args.drop_high / 100) if args.drop_high else np.inf
    keep[idx] = (x >= lo) & (x <= hi)
    name = "ALL" if args.global_cut else df.group.iloc[idx[0]]
    t = df.n_tokens.to_numpy()[idx]
    cuts.append({"group": name, "docs": len(idx), "tokens": int(t.sum()), "low_cut": lo, "high_cut": hi,
                 "docs_kept": keep[idx].mean(), "tokens_kept": t[keep[idx]].sum() / t.sum()})
pd.DataFrame(cuts).sort_values("tokens", ascending=False).to_csv(os.path.join(out_dir, "band_cuts.csv"), index=False, float_format="%.4f")

# Copy the kept docs: runs of consecutive kept docs are contiguous token ranges
src = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint32, mode="r")
start = df.start.to_numpy()
end = start + df.n_tokens.to_numpy()
edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
runs = edges.reshape(-1, 2)  # [first kept doc, one past the last] per run
written = 0
tmp = os.path.join(out_dir, "train.bin.tmp")
with open(tmp, "wb") as f:
    for a, b in runs:
        s, e = start[a], end[b - 1]
        for c in range(s, e, 1 << 26):
            chunk = np.asarray(src[c:min(e, c + (1 << 26))])
            f.write(chunk.tobytes())
            written += len(chunk)
    f.flush()
    os.fsync(f.fileno())
assert written == df.n_tokens.to_numpy()[keep].sum()
os.replace(tmp, os.path.join(out_dir, "train.bin"))
val = os.path.join(out_dir, "val.bin")
if not os.path.exists(val):
    os.symlink(os.path.join(args.data, "val.bin"), val)

meta = json.load(open(os.path.join(args.data, "meta.json")))
meta.update({"train_docs": int(keep.sum()), "train_tokens": int(written),
             "band_filter": {"scores": args.scores, "drop_low_pct": args.drop_low, "drop_high_pct": args.drop_high,
                             "per_group": not args.global_cut, "source_data": args.data,
                             "docs_kept": float(keep.mean()), "tokens_kept": float(written / len(src))}})
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
print(f"{out_dir}: kept {keep.sum():,} / {len(df):,} docs ({keep.mean():.1%}), {written:,} / {len(src):,} tokens ({written / len(src):.1%})")
