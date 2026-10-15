"""Score held-out Dolma 3 docs with a small LM (train_scorer.py) for perplexity-based filtering: per-doc mean token loss.

    sbatch --gpus-per-node=1 --cpus-per-task=12 --time=3:00:00 --job-name=score-docs --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python score_docs.py --model /work/$USER/llama_models/scorer-6b"

Docs: the ones right after train.bin in the seed-42 shuffle (ranks [train_cut, train_cut + K), K ~ --tokens worth), so a
uniform sample of the mix that neither the scorer (trained on train.bin) nor eval (the last 50k ranks) contains.
Each doc is tokenized as in prepare_data.py (BOS text EOS) and scored in windows of --window tokens (the scorer's
context; a longer doc is split, each window scored on its own): mean_loss = mean CE over every next-token prediction
inside its windows. Output (--out_dir):
  scores.parquet: rank, text_hash (common_crawl metadata, else ""), key (64-bit dedup key as in prepare_data.py),
                  source, topic, vigintile (common_crawl quality bucket, else ""), n_tokens, n_pred, mean_loss
  percentiles.csv: mean_loss percentiles per topic and over all docs, weighted by docs and by tokens
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import KEY_MASK, NUM_EVAL_DOCS, SEED, iter_lines  # noqa: E402
from scoring import group_name, parse_dir, score_stream  # noqa: E402

REPO_DIR = os.path.expandvars("/work/$USER/hf_cache/hub/datasets--allenai--dolma3_mix-150B-1025/snapshots")
PCTS = [1, 2, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 98, 99]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default=os.path.expandvars("/work/$USER/dolma3_llama"), help="prepare_data.py output (doc_counts.json, meta.json)")
ap.add_argument("--out_dir", default=None, help="default: <data>/scores/<model name>")
ap.add_argument("--tokens", type=float, default=1e9, help="Tokens to score (docs are picked by the train set's tokens per doc)")
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--batch_tokens", type=int, default=65536)
ap.add_argument("--num_proc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
args = ap.parse_args()
out_dir = args.out_dir or os.path.join(args.data, "scores", os.path.basename(args.model.rstrip("/")))
os.makedirs(out_dir, exist_ok=True)

counts = json.load(open(os.path.join(args.data, "doc_counts.json")))
meta = json.load(open(os.path.join(args.data, "meta.json")))
files, per_file = counts["files"], counts["counts"]
n = sum(per_file)
eval_cut = n - NUM_EVAL_DOCS
train_cut = int(0.05 * eval_cut)
assert meta["train_docs"] == train_cut and meta.get("dedup") is None, "--data must be the original (non-dedup) prepare_data output"
k = int(np.ceil(args.tokens / (meta["train_tokens"] / meta["train_docs"])))
lo, hi = train_cut, train_cut + k
assert hi <= eval_cut
root = glob.glob(os.path.join(REPO_DIR, counts["revision"]))[0]
print(f"scoring ranks [{lo:,}, {hi:,}) = {k:,} docs (~{args.tokens / 1e9:g}B tokens), disjoint from train [0, {train_cut:,}) and eval", flush=True)

perm = np.random.default_rng(SEED).permutation(n)
rank = np.empty(n, np.int64)
rank[perm] = np.arange(n)
del perm
offsets = np.concatenate([[0], np.cumsum(per_file)])
file_rank = [rank[offsets[i]:offsets[i + 1]] for i in range(len(files))]
del rank

tokenizer = None


def load_file(i):
    global tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.model_max_length = int(1e9)
    ranks = file_rank[i]
    if not ((ranks >= lo) & (ranks < hi)).any():
        return None
    rs, texts, th, keys = [], [], [], []
    for line, r in zip(iter_lines(os.path.join(root, files[i])), ranks, strict=True):
        if lo <= r < hi:
            d = json.loads(line)
            t = (d.get("metadata") or {}).get("text_hash")
            rs.append(int(r)), texts.append(d["text"]), th.append("" if t is None else str(t))
            keys.append(int(t) & KEY_MASK if t is not None else int.from_bytes(hashlib.blake2b(d["text"].encode(), digest_size=8).digest(), "little"))
    ids = [np.asarray(x + [tokenizer.eos_token_id], np.uint32) for x in tokenizer(texts)["input_ids"]]
    return i, rs, th, keys, ids


docs = []  # (rank, file index, text_hash, key, token ids)
with Pool(args.num_proc) as pool:
    for j, res in enumerate(pool.imap_unordered(load_file, range(len(files)), chunksize=2)):
        if res is not None:
            i, rs, th, keys, ids = res
            docs += [(r, i, h, kk, x) for r, h, kk, x in zip(rs, th, keys, ids)]
        if j % 500 == 0:
            print(f"read {j + 1}/{len(files)} files, {len(docs):,} docs", flush=True)
assert len(docs) == k, (len(docs), k)
docs.sort(key=lambda d: d[0])
n_tok = np.array([len(d[4]) for d in docs], np.int64)
tokens = np.concatenate([d[4] for d in docs])
print(f"{k:,} docs, {n_tok.sum():,} tokens; scoring with {args.model}", flush=True)

from transformers import AutoModelForCausalLM  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="flash_attention_3", dtype=torch.bfloat16).cuda().eval()
loss_sum, pred = score_stream(model, tokens, np.concatenate([[0], np.cumsum(n_tok)]), args.window, args.batch_tokens,
                              log=lambda m: print(m, flush=True))
mean_loss = loss_sum / np.maximum(pred, 1)
assert (pred > 0).all()
import pandas as pd  # noqa: E402

cells = [parse_dir(files[d[1]].split("/")[1]) for d in docs]
df = pd.DataFrame({
    "rank": [d[0] for d in docs], "text_hash": [d[2] for d in docs], "key": np.array([d[3] for d in docs], np.uint64),
    "source": [c[0] for c in cells], "topic": [c[1] for c in cells], "vigintile": [c[2] for c in cells],
    "n_tokens": n_tok, "n_pred": pred.astype(np.int64), "mean_loss": mean_loss.astype(np.float32),
})
df.to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
print(f"token-weighted mean loss {np.average(mean_loss, weights=pred):.4f}; saved {out_dir}/scores.parquet", flush=True)


def pct(x, w, q):  # weighted percentiles: smallest x with cumulative weight >= q
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][np.minimum(np.searchsorted(c, np.array(q) / 100), len(x) - 1)]


rows = []
df["group"] = [group_name(a, b) for a, b in zip(df.source, df.topic)]
for name, g in [("ALL", df), *sorted(df.groupby("group"))]:
    x = g.mean_loss.to_numpy(np.float64)
    for weight in ("docs", "tokens"):
        w = np.ones(len(g)) if weight == "docs" else g.n_tokens.to_numpy(np.float64)
        rows.append({"group": name, "weight": weight, "docs": len(g), "tokens": int(g.n_tokens.sum()),
                     "token_share": g.n_tokens.sum() / df.n_tokens.sum(), "mean": np.average(x, weights=w),
                     **{f"p{q}": v for q, v in zip(PCTS, pct(x, w, PCTS))}})
pt = pd.DataFrame(rows)
pt.to_csv(os.path.join(out_dir, "percentiles.csv"), index=False, float_format="%.4f")
with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
    print(pt[pt.weight == "tokens"].drop(columns="weight").round(3).to_string(index=False))
print(f"saved {out_dir}/percentiles.csv")
