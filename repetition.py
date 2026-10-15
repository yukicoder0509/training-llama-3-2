"""Measure exact-text repetition inside a packed train.bin, over its first N tokens and over the whole file (CPU only).

    sbatch --gpus-per-node=1 --cpus-per-task=12 --time=1:00:00 --job-name=repetition --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python repetition.py --prefix_tokens 3e9"

A document = <|begin_of_text|> ... up to the next <|begin_of_text|> (prepare_data.py packs BOS text EOS). Two docs repeat
if their token sequences are identical (64-bit blake2b). prepare_data.py shuffled the docs, so the first N tokens are a
uniform sample like the random blocks a run of N tokens sees. Also counts train docs identical to a val.bin doc.
"""

import argparse
import hashlib
import os
from multiprocessing import Pool

import numpy as np

BOS = 128000
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.expandvars("/work/$USER/dolma3_llama"))
ap.add_argument("--prefix_tokens", type=float, default=3e9)
ap.add_argument("--num_proc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()


def bos_positions(path):
    t = np.memmap(path, dtype=np.uint32, mode="r")
    step = 1 << 28
    return np.concatenate([np.flatnonzero(t[i:i + step] == BOS) + i for i in range(0, len(t), step)]), len(t)


def hash_range(job):
    path, starts, ends = job
    t = np.memmap(path, dtype=np.uint32, mode="r")
    return np.array([int.from_bytes(hashlib.blake2b(t[s:e].tobytes(), digest_size=8).digest(), "little")
                     for s, e in zip(starts, ends)], dtype=np.uint64)


def doc_hashes(path):
    starts, n = bos_positions(path)
    ends = np.append(starts[1:], n)
    chunks = np.array_split(np.arange(len(starts)), args.num_proc * 8)
    with Pool(args.num_proc) as pool:
        h = np.concatenate(pool.map(hash_range, [(path, starts[c], ends[c]) for c in chunks]))
    return h, starts, ends - starts


def report(name, h, lens, val_set):
    uniq, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    k = cnt[inv]  # copies of each doc's text inside this range
    total = lens.sum()
    first = np.zeros(len(h), bool)
    first[np.unique(inv, return_index=True)[1]] = True
    excess = lens[~first].sum()
    print(f"\n=== {name}: {len(h):,} docs, {total / 1e9:.3f}B tokens ===")
    print(f"distinct texts {len(uniq):,} ({100 * len(uniq) / len(h):.1f}% of docs); "
          f"repeated copies (beyond each text's first) {(~first).sum():,} docs = {excess / 1e9:.3f}B tokens "
          f"= {100 * excess / total:.1f}% of tokens")
    print("copies k of a text | docs | tokens (% of range)")
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 16), (17, 100), (101, 1000), (1001, 10**12)):
        m = (k >= lo) & (k <= hi)
        print(f"  {lo:>5}-{hi if hi < 10**12 else 'inf':<5} | {m.sum():>10,} | {lens[m].sum() / 1e9:7.3f}B ({100 * lens[m].sum() / total:5.1f}%)")
    print(f"token-weighted mean copies: {np.average(k, weights=lens):.2f}; max copies {cnt.max():,}")
    o = np.argsort(-cnt * np.bincount(inv, weights=lens) / cnt)[:args.top]  # texts by total tokens taken
    tok_per_text = np.bincount(inv, weights=lens) / cnt
    print(f"top {args.top} texts by tokens taken: copies x length = tokens")
    for i in o:
        print(f"  {cnt[i]:>8,} x {int(tok_per_text[i]):>6} = {cnt[i] * tok_per_text[i] / 1e6:7.2f}M ({100 * cnt[i] * tok_per_text[i] / total:.2f}%)")
    inval = np.isin(h, val_set)
    print(f"train docs identical to a val.bin doc: {inval.sum():,} docs, {lens[inval].sum() / 1e6:.2f}M tokens; "
          f"distinct val texts hit: {np.isin(val_set, h).sum():,} of {len(val_set):,}")
    return uniq


vh, _, vlens = doc_hashes(os.path.join(args.data, "val.bin"))
val_set = np.unique(vh)
print(f"val.bin: {len(vh):,} docs, {len(val_set):,} distinct texts")
h, starts, lens = doc_hashes(os.path.join(args.data, "train.bin"))
p = starts < args.prefix_tokens
report(f"first {args.prefix_tokens / 1e9:g}B tokens", h[p], lens[p], val_set)
report("whole train.bin", h, lens, val_set)
