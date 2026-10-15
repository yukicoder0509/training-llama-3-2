"""Score every doc of a packed train.bin with a small LM (train_scorer.py): per-doc mean token loss, on 8 GPUs.

    sbatch --gpus-per-node=8 --cpus-per-task=96 --time=2:00:00 --job-name=score-bin --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python score_bin.py --stage all --model /work/$USER/llama_models/scorer-6b"

Stages: index (doc starts = <|begin_of_text|> right after <|end_of_text|>, or at 0), score (--shard i of
--num_shards: a contiguous range of docs on GPU i), merge (one parquet, each doc with its source / topic / vigintile:
train docs are written in file order, so the i-th doc is the i-th train-rank doc of the source files). `all` runs index,
one score process per GPU, then merge. Output: <data>/scores/<model name>/train_scores.parquet with doc, start,
n_tokens, n_pred, mean_loss, source, topic, vigintile, group.

The scorer may have trained on these docs (a 6B run sees ~76% of train.bin), which lowers their loss a little.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import group_name, parse_dir, score_stream  # noqa: E402

BOS, EOS = 128000, 128001
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["all", "index", "score", "merge"], default="all")
ap.add_argument("--model", required=True)
ap.add_argument("--data", default=os.path.expandvars("/work/$USER/dolma3_llama"))
ap.add_argument("--out_dir", default=None, help="default: <data>/scores/<model name>")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num_shards", type=int, default=8)
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--batch_tokens", type=int, default=65536)
args = ap.parse_args()
out_dir = args.out_dir or os.path.join(args.data, "scores", os.path.basename(args.model.rstrip("/")))
os.makedirs(out_dir, exist_ok=True)
starts_path = os.path.join(out_dir, "train_doc_starts.npy")
bin_path = os.path.join(args.data, "train.bin")


def log(m):
    print(f"[{args.stage} {args.shard}] {m}", flush=True)


def index():
    t = np.memmap(bin_path, dtype=np.uint32, mode="r")
    step, starts = 1 << 28, []
    for i in range(0, len(t), step):
        b = np.flatnonzero(t[i:i + step] == BOS) + i
        starts.append(b[(b == 0) | (t[np.maximum(b - 1, 0)] == EOS)])  # a BOS inside a text is not a boundary
    starts = np.append(np.concatenate(starts), len(t))
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    assert len(starts) - 1 == meta["train_docs"], (len(starts) - 1, meta["train_docs"])
    np.save(starts_path, starts)
    log(f"{len(starts) - 1:,} docs, {len(t):,} tokens")


def score():
    import torch
    from transformers import AutoModelForCausalLM
    starts = np.load(starts_path)
    k = len(starts) - 1
    # contiguous doc ranges with ~equal tokens
    cuts = np.searchsorted(starts, np.linspace(0, starts[-1], args.num_shards + 1))
    cuts[0], cuts[-1] = 0, k
    a, b = int(cuts[args.shard]), int(cuts[args.shard + 1])
    torch.cuda.set_device(args.shard % torch.cuda.device_count())
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="flash_attention_3", dtype=torch.bfloat16).cuda().eval()
    t = np.memmap(bin_path, dtype=np.uint32, mode="r")
    log(f"docs [{a:,}, {b:,}), {starts[b] - starts[a]:,} tokens")
    loss_sum, pred = score_stream(model, t, starts[a:b + 1], args.window, args.batch_tokens, log)
    np.savez(os.path.join(out_dir, f"train_shard{args.shard}.npz"), a=a, b=b, loss_sum=loss_sum, pred=pred)
    log("done")


def merge():
    import pandas as pd
    from prepare_data import NUM_EVAL_DOCS, SEED
    starts = np.load(starts_path)
    k = len(starts) - 1
    loss_sum, pred = np.zeros(k), np.zeros(k)
    for s in range(args.num_shards):
        z = np.load(os.path.join(out_dir, f"train_shard{s}.npz"))
        loss_sum[z["a"]:z["b"]], pred[z["a"]:z["b"]] = z["loss_sum"], z["pred"]
    assert (pred > 0).all()
    # train docs per file, in file order (prepare_data.py: rank < train_cut, written in file order)
    counts = json.load(open(os.path.join(args.data, "doc_counts.json")))
    n = sum(counts["counts"])
    perm = np.random.default_rng(SEED).permutation(n)
    rank = np.empty(n, np.int64)
    rank[perm] = np.arange(n)
    is_train = rank < int(0.05 * (n - NUM_EVAL_DOCS))
    offsets = np.concatenate([[0], np.cumsum(counts["counts"])])
    cs = np.concatenate([[0], np.cumsum(is_train)])
    per_file = cs[offsets[1:]] - cs[offsets[:-1]]
    assert per_file.sum() == k
    cells = [parse_dir(f.split("/")[1]) for f in counts["files"]]
    cell_of_doc = np.repeat(np.arange(len(cells)), per_file)
    df = pd.DataFrame({"doc": np.arange(k), "start": starts[:-1], "n_tokens": np.diff(starts), "n_pred": pred.astype(np.int64),
                       "mean_loss": (loss_sum / pred).astype(np.float32)})
    df["source"] = [cells[c][0] for c in cell_of_doc]
    df["topic"] = [cells[c][1] for c in cell_of_doc]
    df["vigintile"] = [cells[c][2] for c in cell_of_doc]
    df["group"] = [group_name(a, b) for a, b in zip(df.source, df.topic)]
    df.to_parquet(os.path.join(out_dir, "train_scores.parquet"), index=False)
    log(f"{k:,} docs, token-weighted mean loss {np.average(df.mean_loss, weights=df.n_pred):.4f}; saved {out_dir}/train_scores.parquet")


if args.stage == "index":
    index()
elif args.stage == "score":
    score()
elif args.stage == "merge":
    merge()
else:
    index()
    base = [sys.executable, os.path.abspath(__file__), "--model", args.model, "--data", args.data, "--out_dir", out_dir,
            "--num_shards", str(args.num_shards), "--window", str(args.window), "--batch_tokens", str(args.batch_tokens)]
    procs = [subprocess.Popen(base + ["--stage", "score", "--shard", str(s)]) for s in range(args.num_shards)]
    assert all(p.wait() == 0 for p in procs), "a score shard failed"
    merge()
