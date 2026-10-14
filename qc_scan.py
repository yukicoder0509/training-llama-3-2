"""Collect the dolma2_qc quality score of allenai/dolma3_mix-150B-1025 docs, for picking a filter threshold.

dolma2_qc is a fastText classifier output {"0": p_low, "1": p_high}; we keep p_high = ["1"]. Only the common_crawl-*
sources carry it (wiki, arxiv, stack_edu, finemath and olmocr PDFs do not). Reads every --every-th shard of each source
directory (a uniform sample of the shards), and writes, for the CC docs: p_high, character count, WebOrganizer topic;
plus per-directory doc / char totals for all sources (a topic's -00NN bucket directories differ ~1000x in size,
so plot_qc.py weights each sampled doc by its directory's sampling rate), and a tokens-per-char calibration (every 200th doc tokenized).
Plot with plot_qc.py. Runs on a compute node (qc_scan.sbatch): it decompresses and parses ~10% of the 104 GB.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from transformers import AutoTokenizer

from prepare_data import iter_lines

ROOT = "/work/{user}/hf_cache/hub/datasets--allenai--dolma3_mix-150B-1025/snapshots/afa92bfb22366821c5e6cd427cdd036b34b713ef/data"
CALIB_EVERY = 200
SNIPPETS_PER_FILE = 4


def source_of(d):
    return d.rsplit("-", 1)[0] if d[-4:].isdigit() else d  # common_crawl-<topic>-0014 -> common_crawl-<topic>


def scan(path):
    src = os.path.basename(os.path.dirname(path))  # per directory: the -00NN buckets of a topic differ in size and score
    cc = src.startswith("common_crawl")
    p, chars, n, nchar, calib, snippets = [], [], 0, 0, [], []
    for i, line in enumerate(iter_lines(path)):
        d = json.loads(line)
        L = len(d["text"])
        n += 1
        nchar += L
        if i % CALIB_EVERY == 0:
            calib.append(d["text"])
        if cc:
            q = (d.get("metadata") or {}).get("dolma2_qc")
            p.append(np.nan if q is None else q.get("1", np.nan))
            chars.append(L)
            if len(snippets) < SNIPPETS_PER_FILE and i % 997 == 0:
                snippets.append((p[-1], d["text"][:600]))
    toks = sum(len(ids) for ids in tok(calib)["input_ids"]) if calib else 0
    calib_chars = sum(len(t) for t in calib)
    return src, np.array(p, np.float32), np.array(chars, np.int64), n, nchar, toks, calib_chars, snippets


def init():
    global tok
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
    tok.model_max_length = int(1e12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10, help="Read every N-th shard of each source dir (10 = ~10%%)")
    ap.add_argument("--out", default="/work/{user}/dolma3_llama/qc_scan.npz")
    ap.add_argument("--num_proc", type=int, default=os.cpu_count())
    args = ap.parse_args()
    root, out = ROOT.format(user=os.environ["USER"]), args.out.format(user=os.environ["USER"])

    jobs, total_files = [], defaultdict(int)
    for d in sorted(os.listdir(root)):
        shards = sorted(glob.glob(os.path.join(root, d, "*.jsonl.zst")))
        total_files[d] += len(shards)
        jobs += shards[:: args.every]
    print(f"{len(jobs)} of {sum(total_files.values())} shards, {args.num_proc} procs", flush=True)

    dirs, P, C, T = [], [], [], []
    per_src = defaultdict(lambda: np.zeros(4, np.int64))  # docs, chars, calib tokens, calib chars
    sampled = defaultdict(int)
    snippets = []
    with Pool(args.num_proc, initializer=init) as pool:
        for k, (src, p, chars, n, nchar, toks, cch, snip) in enumerate(pool.imap_unordered(scan, jobs)):
            per_src[src] += (n, nchar, toks, cch)
            sampled[src] += 1
            if len(p):
                dirs.append(src)
                P.append(p), C.append(chars), T.append(np.full(len(p), len(dirs) - 1, np.int16))
            snippets += [(float(q), src, t) for q, t in snip]
            if k % 50 == 0:
                print(f"{k + 1}/{len(jobs)} shards", flush=True)

    # one id per CC directory (the list above has one entry per shard)
    names = sorted(set(dirs))
    remap = np.array([names.index(t) for t in dirs], np.int16)
    srcs = sorted(per_src)
    np.savez_compressed(
        out, p_high=np.concatenate(P), chars=np.concatenate(C), dir=remap[np.concatenate(T)], dir_names=np.array(names),
        sources=np.array(srcs), src_stats=np.stack([per_src[s] for s in srcs]),
        src_sampled_files=np.array([sampled[s] for s in srcs]), src_total_files=np.array([total_files[s] for s in srcs]),
    )
    json.dump(sorted(snippets), open(out.replace(".npz", "_snippets.json"), "w"), indent=0, ensure_ascii=False)
    print(f"wrote {out}: {sum(len(p) for p in P):,} CC docs", flush=True)


if __name__ == "__main__":
    main()
