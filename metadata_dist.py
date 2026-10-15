"""Is train.bin a representative sample of the Dolma 3 mix? Token counts per (source x topic x quality vigintile x length bin).

    sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=128G --time=1:00:00 --job-name=metadata-dist --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python metadata_dist.py"

Cells come from each doc's directory: data/<source>-<topic>[-<vigintile>]/shard.jsonl.zst. Source: common_crawl,
olmocr_science_pdfs, stack_edu, ...; topic: the WebOrganizer topic (stack_edu: the language; others "-"); vigintile:
the quality bucket 0000-0019 of common_crawl (others "-"). Length bin: tokens per doc.

Population = every train-eligible doc of the mix (all ranks < N - 50k). Its chars per doc come from dedup_keys.npz
(prepare_data.py --dedup); its tokens are estimated per cell as chars x (tokens / chars of that cell's train docs).
Train sets: original train.bin (shuffled ranks < train_cut) and the dedup train.bin (selection replayed from
prepare_data.py); both are written in file order, so their docs line up with the BOS-after-EOS boundaries of train.bin
and get exact token counts. Sampling noise per cell: original train is a uniform doc sample (fraction f), so its doc
count is ~Binomial(N_cell, f).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BOS, EOS, NUM_EVAL_DOCS, SEED = 128000, 128001, 50_000, 42
LEN_EDGES = [0, 128, 512, 2048, 8192, 32768, np.inf]
LEN_NAMES = ["<128", "128-512", "512-2k", "2k-8k", "8k-32k", ">=32k"]

ap = argparse.ArgumentParser()
ap.add_argument("--orig", default=os.path.expandvars("/work/$USER/dolma3_llama"))
ap.add_argument("--dedup", default=os.path.expandvars("/work/$USER/dolma3_llama_dedup"))
ap.add_argument("--out", default="plots/metadata_dist.png")
args = ap.parse_args()

# --- per-doc metadata in global file order
counts = json.load(open(os.path.join(args.orig, "doc_counts.json")))
files, per_file = counts["files"], np.array(counts["counts"])
n = int(per_file.sum())
dirs = sorted({f.split("/")[1] for f in files})


def parse(d):
    head, _, tail = d.rpartition("-")
    if tail.isdigit():  # common_crawl-<topic>-00NN
        src, _, topic = head.partition("-")
        return src, topic, tail[-2:]
    for src in ("common_crawl", "olmocr_science_pdfs", "stack_edu"):
        if d.startswith(src + "-"):
            return src, d[len(src) + 1:], "-"
    return d, "-", "-"


dir_meta = [parse(d) for d in dirs]
doc_dir = np.repeat(np.array([dirs.index(f.split("/")[1]) for f in files], np.int16), per_file)
z = np.load(os.path.join(args.dedup, "dedup_keys.npz"))
keys, chars = z["keys"], z["chars"]
assert len(chars) == n
perm = np.random.default_rng(SEED).permutation(n)
rank = np.empty(n, np.int64)
rank[perm] = np.arange(n)
del perm
eval_cut = n - NUM_EVAL_DOCS
pop = rank < eval_cut
orig_sel = rank < int(0.05 * eval_cut)

# dedup selection, as in prepare_data.py --dedup --min_train_tokens
order = np.lexsort((rank, keys))
first = np.ones(n, bool)
first[order[1:]] = keys[order[1:]] != keys[order[:-1]]
del order
keep = first & ~np.isin(keys, keys[~pop])
del first, keys
tr_ids = np.flatnonzero(keep)
tr_ids = tr_ids[np.random.default_rng(SEED + 1).permutation(len(tr_ids))]
m = json.load(open(os.path.join(args.dedup, "meta.json")))["train_docs"]
dedup_sel = np.zeros(n, bool)
dedup_sel[tr_ids[:m]] = True
del tr_ids, keep


def doc_tokens(path, n_docs):
    t = np.memmap(path, dtype=np.uint32, mode="r")
    step, starts = 1 << 28, []
    for i in range(0, len(t), step):
        b = np.flatnonzero(t[i:i + step] == BOS) + i
        starts.append(b[(b == 0) | (t[np.maximum(b - 1, 0)] == EOS)])  # a BOS inside a text is not a boundary
    starts = np.concatenate(starts)
    assert len(starts) == n_docs, (path, len(starts), n_docs)
    return np.diff(np.append(starts, len(t)))


tok = {"orig": doc_tokens(os.path.join(args.orig, "train.bin"), int(orig_sel.sum())),
       "dedup": doc_tokens(os.path.join(args.dedup, "train.bin"), m)}
sel = {"orig": orig_sel, "dedup": dedup_sel}

# --- cells: dir x length bin. Length of population docs: chars / (tokens per char of its dir, from original train)
tc = np.bincount(doc_dir[orig_sel], weights=tok["orig"], minlength=len(dirs)), np.bincount(doc_dir[orig_sel], weights=chars[orig_sel], minlength=len(dirs))
tpc_dir = np.where(tc[1] > 0, tc[0] / np.maximum(tc[1], 1), tok["orig"].sum() / chars[orig_sel].sum())  # dirs without a train doc: global ratio
print("dirs with no original train doc:", [(dirs[i], int((doc_dir[pop] == i).sum())) for i in np.flatnonzero(tc[1] == 0)])
est_len = chars * tpc_dir[doc_dir]
lbin = np.digitize(est_len, LEN_EDGES[1:-1])  # same (estimated) bin for every doc in every set
cell = doc_dir.astype(np.int64) * len(LEN_NAMES) + lbin
C = len(dirs) * len(LEN_NAMES)
pop_docs = np.bincount(cell[pop], minlength=C)
pop_chars = np.bincount(cell[pop], weights=chars[pop], minlength=C)
res = {}
for k in ("orig", "dedup"):
    s = sel[k]
    res[k] = dict(docs=np.bincount(cell[s], minlength=C), chars=np.bincount(cell[s], weights=chars[s], minlength=C),
                  tokens=np.bincount(cell[s], weights=tok[k], minlength=C))
tpc_cell = np.where(res["orig"]["chars"] > 0, res["orig"]["tokens"] / np.maximum(res["orig"]["chars"], 1), tpc_dir.repeat(len(LEN_NAMES)))
pop_tokens = pop_chars * tpc_cell
print(f"population: {pop.sum():,} docs, ~{pop_tokens.sum() / 1e9:.1f}B tokens (estimated); "
      f"orig train {orig_sel.sum():,} docs {tok['orig'].sum() / 1e9:.3f}B; dedup train {m:,} docs {tok['dedup'].sum() / 1e9:.3f}B")

cell_dir, cell_len = np.arange(C) // len(LEN_NAMES), np.arange(C) % len(LEN_NAMES)
FACTORS = {"source": lambda c: dir_meta[cell_dir[c]][0],
           "topic": lambda c: f"{dir_meta[cell_dir[c]][0][:5]}:{dir_meta[cell_dir[c]][1]}",
           "quality vigintile (common_crawl)": lambda c: dir_meta[cell_dir[c]][2] if dir_meta[cell_dir[c]][0] == "common_crawl" else None,
           "length bin (tokens)": lambda c: LEN_NAMES[cell_len[c]]}


def shares(x):
    return x / x.sum()


p = shares(pop_tokens)
f = orig_sel.sum() / pop.sum()
for k in ("orig", "dedup"):
    q = shares(res[k]["tokens"])
    print(f"\n### {k} train vs population, {int((pop_docs > 0).sum())} non-empty cells")
    print(f"total variation distance of token shares (0 = identical, 1 = disjoint): {0.5 * np.abs(q - p).sum():.4f}")
    if k == "orig":
        exp = pop_docs * f
        zc = (res[k]["docs"] - exp) / np.sqrt(np.maximum(exp * (1 - f), 1e-9))
        big = pop_docs >= 1000
        print(f"doc-count z per cell (cells with >= 1000 population docs: {big.sum()}): |z|>3 in {(np.abs(zc[big]) > 3).sum()} "
              f"(expected ~{0.0027 * big.sum():.1f} if uniform), max |z| {np.abs(zc[big]).max():.2f}; "
              f"chi2/dof {np.sum(zc[big] ** 2) / big.sum():.3f} (expected ~1)")
        # TV expected from sampling alone: resample cells binomially
        rng = np.random.default_rng(0)
        tpd = np.where(res[k]["docs"] > 0, res[k]["tokens"] / np.maximum(res[k]["docs"], 1), pop_tokens / np.maximum(pop_docs, 1))
        tv_null = [0.5 * np.abs(shares(rng.binomial(pop_docs, f) * tpd) - p).sum() for _ in range(20)]
        print(f"TV from sampling noise alone (binomial resample, same tokens/doc per cell): {np.mean(tv_null):.4f} ± {np.std(tv_null):.4f}")
    for name, fn in FACTORS.items():
        lab = np.array([fn(c) for c in range(C)], object)
        groups = LEN_NAMES if name.startswith("length") else sorted({x for x in lab if x is not None})
        rows = [(g, p[lab == g].sum(), q[lab == g].sum()) for g in groups]
        tv = 0.5 * sum(abs(a - b) for _, a, b in rows)
        print(f"\n{name}: TV {tv:.4f}")
        print(f"  {'':40s} {'population':>10s} {k + ' train':>11s}  ratio")
        for g, a, b in (rows if name.startswith("length") else sorted(rows, key=lambda r: -r[1])):
            print(f"  {g:40s} {100 * a:9.3f}% {100 * b:10.3f}%  {b / a if a else np.nan:5.2f}")

np.savez(os.path.join(args.orig, "metadata_dist.npz"), dirs=np.array(dirs), len_names=np.array(LEN_NAMES), pop_docs=pop_docs,
         pop_chars=pop_chars, pop_tokens=pop_tokens, **{f"{k}_{c}": v for k in res for c, v in res[k].items()})

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for a, k in zip(ax, ("orig", "dedup")):
    q = shares(res[k]["tokens"])
    nz = (p > 0) | (q > 0)
    a.loglog(p[nz] + 1e-7, q[nz] + 1e-7, ".", ms=3, alpha=0.6)
    a.plot([1e-7, 1], [1e-7, 1], "k--", lw=0.8)
    a.set_xlim(1e-7, 0.1), a.set_ylim(1e-7, 0.1)
    a.set_xlabel("population token share per cell"), a.set_ylabel(f"{k} train token share")
    a.set_title(f"{k} train vs mix: {nz.sum()} cells (dir x length), TV {0.5 * np.abs(q - p).sum():.3f}")
    a.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=110)
print(f"saved {args.out}")
