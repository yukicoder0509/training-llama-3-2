"""Loss-based soft reweighting, step 4: draw rows from pool.parquet (build_pool.py) with probability ~ exp(beta * z).

    sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=200G --time=1:00:00 --job-name=select --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python select_rows.py --beta 0.25 --budget 1e9 --seed 0 --subpool_mult 3"

Rows are drawn without replacement by Efraimidis-Spirakis keys log(u) / w, w = exp(beta * z), until the next row would
pass --budget tokens. beta = 0 is a uniform random draw, beta > 0 favors high-loss (harder) rows, beta < 0 easier
ones; no row is excluded outright or copied. --subpool_mult k (proxy mode) first takes a seeded uniform subpool of
k x budget tokens and selects from it, so proxy and final runs keep the same pool:budget ratio. Everything follows
from --seed.

Out (--out_dir, default <pool>/sel_b<beta>_<budget>_s<seed>): train.bin (selected rows, shuffled, BOS text EOS),
val.bin -> the official val.bin, dev.bin -> <pool>/dev.bin, meta.json, select_report.txt.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import write_tokens  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
POOL = os.environ.get("POOL_DIR", cfg["pool"]["dir"])  # POOL_DIR: a test dir
ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--budget", type=float, required=True, help="tokens to select (BOS and EOS included)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--subpool_mult", type=float, default=0, help="proxy mode: select from a random subpool of this x budget tokens (0: whole pool)")
ap.add_argument("--out_dir", default=None)
args = ap.parse_args()
out_dir = args.out_dir or os.path.join(POOL, f"sel_b{args.beta:g}_{args.budget:.3g}_s{args.seed}" + (f"_sub{args.subpool_mult:g}" if args.subpool_mult else ""))
if os.path.exists(os.path.join(out_dir, "train.bin")):
    sys.exit(f"{out_dir}/train.bin exists, nothing to do")
os.makedirs(out_dir, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def select(pool, beta, budget_tokens, rng):
    w = np.exp(beta * pool.z.to_numpy(np.float64))
    u = rng.random(len(pool))
    key = np.log(u) / w  # equals log(u ** (1 / w)); larger is drawn first
    order = np.argsort(-key, kind="stable")
    cum = pool.n_tokens_full.to_numpy(np.int64)[order].cumsum()
    return order[cum <= budget_tokens]


pool = pd.read_parquet(os.path.join(POOL, "pool.parquet"), columns=["row_id", "shard", "topic", "n_tokens_full", "tok_start", "z"])
rng = np.random.default_rng(args.seed)
if args.subpool_mult:
    perm = rng.permutation(len(pool))
    cum = pool.n_tokens_full.to_numpy(np.int64)[perm].cumsum()
    pool = pool.iloc[np.sort(perm[cum <= args.subpool_mult * args.budget])].reset_index(drop=True)
sel = select(pool, args.beta, args.budget, rng)
chosen = np.zeros(len(pool), bool)
chosen[sel] = True
S = pool.iloc[sel]
T, PT = int(S.n_tokens_full.sum()), int(pool.n_tokens_full.sum())

rep = []


def say(s=""):
    print(s, flush=True)
    rep.append(s)


say(f"beta {args.beta:g}, seed {args.seed}, budget {args.budget:.4g}: {len(S):,} of {len(pool):,} rows, {T:,} of {PT:,} tokens "
    f"({PT / T:.2f}:1 pool:selection)")
say(f"mean z: selection {S.z.mean():+.4f} (token-weighted {np.average(S.z, weights=S.n_tokens_full):+.4f}), pool {pool.z.mean():+.4f}")
ps = pool.groupby("topic").n_tokens_full.sum() / PT
ss = S.groupby("topic").n_tokens_full.sum().reindex(ps.index).fillna(0) / T
say(f"topic token shares: max |selection - pool| {((ss - ps).abs().max() * 100):.3f} pp")
say(f"{'topic':<34} {'pool %':>8} {'sel %':>8} {'diff pp':>8}")
for g in ps.sort_values(ascending=False).index[:25]:
    say(f"{g:<34} {ps[g] * 100:>8.3f} {ss[g] * 100:>8.3f} {(ss[g] - ps[g]) * 100:>+8.3f}")
say("\nz bin             pool rows   selected rows  tokens   (fraction of the pool's rows / tokens that were selected)")
edges = [-np.inf, -2.5, -2, -1.5, -1, -0.5, -1e-9, 1e-9, 0.5, 1, 1.5, 2, 2.5, np.inf]
b = np.digitize(pool.z.to_numpy(), edges) - 1
nt = pool.n_tokens_full.to_numpy()
for k in range(len(edges) - 1):
    m = b == k
    if m.any():
        name = "z = 0 (untrusted/mean)" if edges[k] == -1e-9 else f"[{edges[k]:+.1f}, {edges[k + 1]:+.1f})"
        say(f"{name:<22} {m.sum():>10,} {chosen[m].mean():>14.3f} {nt[m & chosen].sum() / nt[m].sum():>7.3f}"
            + ("   <- saturating" if chosen[m].mean() > 0.9 else ""))

# Write the selected rows in shuffled order
order = rng.permutation(len(S))
S = S.iloc[order]
n = S.n_tokens_full.to_numpy(np.int64)
pos = np.cumsum(n) - n
out = np.empty(T, np.uint32)
for i, idx in pd.Series(np.arange(len(S))).groupby(S.shard.to_numpy()):
    t = np.memmap(os.path.join(POOL, "tokens", f"{i:04d}.bin"), dtype=np.uint32, mode="r")
    idx = idx.to_numpy()
    for p, s, m in zip(pos[idx], S.tok_start.to_numpy()[idx], n[idx]):
        out[p:p + m] = t[s:s + m]
assert out[pos[0]] == 128000 and (out[pos] == 128000).all() and (out[pos + n - 1] == 128001).all()
write_tokens(os.path.join(out_dir, "train.bin"), [out])
for name, target in (("val.bin", cfg["training"]["official_val"]), ("dev.bin", os.path.join(POOL, "dev.bin"))):
    if not os.path.lexists(os.path.join(out_dir, name)):
        os.symlink(target, os.path.join(out_dir, name))
json.dump({"beta": args.beta, "budget": args.budget, "seed": args.seed, "subpool_mult": args.subpool_mult, "pool": POOL,
           "rows": len(S), "tokens": T, "pool_rows": len(pool), "pool_tokens": PT, "mean_z": float(S.z.mean())},
          open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
open(os.path.join(out_dir, "select_report.txt"), "w").write("\n".join(rep) + "\n")
log(f"wrote {out_dir}")
