"""Split allenai/dolma3_mix-150B-1025 as the OJ does and tokenize it with the Llama 3.2 tokenizer.

Split (DESCRIPTION.md): shuffle the `train` split with seed 42 and reserve the last 50,000 documents for evaluation,
before tokenizing or packing. `datasets` would do this as
    load_dataset(REPO, split="train").shuffle(seed=42)  ->  row order np.random.default_rng(42).permutation(N)
over the data files in the order `datasets` resolves them (sorted paths). Building that arrow dataset would take
~400 GB, so we reproduce the permutation instead: count the docs per file (pass 1), compute the same permutation, and
give every doc its position ("rank") in the shuffled order. Then (pass 2):
  - eval: rank >= N - 50,000 (the OJ's held-out set). All 50k texts go to eval_docs.jsonl (in shuffled order);
    the first --num_val_docs of them are tokenized into val.bin for in-training eval (up to 20% = 10k allowed).
  - train: rank < --train_frac * (N - 50,000), i.e. the first docs of the shuffled train part: a uniform sample of the
    mixture. 5% of ~150B tokens ~= 7.5B tokens, more than the 6B budget, so no token is seen twice.
Train docs are written in file order (blocks are sampled randomly during training anyway).

Each document is tokenized as <|begin_of_text|> text <|end_of_text|> (the tokenizer adds BOS, as at OJ eval time) and
packed into flat uint32 token files (the 128,256 vocab doesn't fit in uint16). The data is not cleaned or deduped.
"""

import hashlib
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Pool

import numpy as np
import zstandard
from datasets import load_dataset_builder
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, HfArgumentParser

REPO = "allenai/dolma3_mix-150B-1025"
NUM_EVAL_DOCS = 50_000
SEED = 42


@dataclass
class PrepareArguments:
    out_dir: str = field(default=os.path.expandvars("/work/$USER/dolma3_llama"))
    tokenizer: str = field(default="NousResearch/Llama-3.2-1B", metadata={"help": "tokenizer.json byte-identical to the gated meta-llama/Llama-3.2-1B (unsloth/Llama-3.2-1B: same ids, other format)"})
    train_frac: float = field(default=0.05, metadata={"help": "Fraction of the shuffled train docs to tokenize (5% ~= 7.5B tokens)"})
    num_val_docs: int = field(default=10_000, metadata={"help": "First N of the 50k eval docs tokenized into val.bin (<= 20% allowed)"})
    num_proc: int = field(default=len(os.sched_getaffinity(0)))
    dedup: bool = field(default=False, metadata={"help": "Exact dedup of the train docs: one copy per text (the lowest shuffled rank), and none of a text that is also an eval doc; train docs are then sampled uniformly over the unique texts. Key: metadata.text_hash (common_crawl), else a 64-bit blake2b of the text. Eval docs are never touched"})
    min_train_tokens: Optional[float] = field(default=None, metadata={"help": "With --dedup: instead of --train_frac (of the unique train texts), take random unique texts until they hold ~this many tokens (chars / measured chars-per-token, +2%%)"})
    qc_max: Optional[float] = field(default=None, metadata={"help": 'Drop train docs with dolma2_qc["1"] >= this (high "1" = spam / ads / short pages, see experiments.md). Docs without the score (non-CC sources) are kept; eval docs are never filtered'})


# Set in main() before the worker pool forks.
tokenizer = None
file_rank = None  # per file: ranks of its docs in the shuffled order
train_cut = eval_cut = None
qc_max = None
file_keep = None  # per file: --dedup mask of train docs to keep (None = keep all)
KEY_MASK = (1 << 64) - 1
QC_RE = re.compile(rb'"dolma2_qc":\s*(\{[^}]*\})')  # quotes inside the text are escaped, so this only matches the metadata


def qc_score(line):
    """dolma2_qc["1"] of a raw JSON line, or None if the doc has no score. Regex instead of json.loads: ~all docs."""
    m = QC_RE.findall(line)
    return json.loads(m[-1])["1"] if m else None


def iter_lines(path):
    """Non-blank lines (raw bytes) of a .jsonl.zst file, i.e. the rows `datasets` reads from it."""
    with open(path, "rb") as f:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), buffer_size=1 << 20)
        for line in reader:
            if not line.isspace():
                yield line


def count_docs(path):
    return sum(1 for _ in iter_lines(path))


def doc_keys(path):
    """Dedup pass for one file: (64-bit key, chars) per doc, plus (tokens, chars) of every 1000th doc for calibration."""
    keys, chars, ctok, cch = [], [], 0, 0
    for i, line in enumerate(iter_lines(path)):
        d = json.loads(line)
        text, th = d["text"], (d.get("metadata") or {}).get("text_hash")
        # text_hash: a 128-bit int (decimal string) on common_crawl docs; truncating to 64 bits: ~3e-4 expected collisions at 1e8 docs
        keys.append(int(th) & KEY_MASK if th is not None else int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"))
        chars.append(len(text))
        if i % 1000 == 0:
            ctok += len(tokenizer(text)["input_ids"]) + 1
            cch += len(text)
    return np.array(keys, np.uint64), np.array(chars, np.int64), ctok, cch


def tokenize(texts, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        for ids in tokenizer(texts[i : i + batch_size])["input_ids"]:  # starts with BOS
            out.append(np.asarray(ids + [tokenizer.eos_token_id], dtype=np.uint32))
    return out


def process_file(job):
    """Pass 2 for one file: (train tokens, #train docs, [(rank, text) of its eval docs], qc stats).

    qc stats over all docs of the file (not just train): docs, bytes, scored docs, and docs / bytes that pass --qc_max.
    """
    i, path = job
    ranks = file_rank[i]
    dkeep = file_keep[i] if file_keep is not None else np.ones(len(ranks), bool)
    train_texts, eval_docs = [], []
    stats = np.zeros(5, np.int64)
    for line, r, dk in zip(iter_lines(path), ranks, dkeep, strict=True):
        keep = bool(dk)
        if qc_max is not None:
            q = qc_score(line)
            keep = keep and (q is None or q < qc_max)
            stats += (1, len(line), q is not None, keep, len(line) * keep)
        if r < train_cut:
            if keep:
                train_texts.append(json.loads(line)["text"])
        elif r >= eval_cut:
            eval_docs.append((int(r), json.loads(line)["text"]))
    ids = tokenize(train_texts)
    return (np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint32)), len(ids), eval_docs, stats


def write_tokens(path, chunks):
    # Sequential writes, not np.memmap: memmap writes to the /work network FS were silently lost before.
    written = 0
    with open(path + ".tmp", "wb") as f:
        for ids in chunks:
            f.write(ids.tobytes())
            written += len(ids)
        f.flush()
        os.fsync(f.fileno())
    check = np.memmap(path + ".tmp", dtype=np.uint32, mode="r")
    assert len(check) == written and check[-4096:].any(), f"{path}.tmp failed verification"
    os.replace(path + ".tmp", path)  # only a complete file ever appears under the final name
    print(f"wrote {written:,} tokens to {path}", flush=True)
    return written


def main():
    global tokenizer, file_rank, train_cut, eval_cut, qc_max, file_keep
    (args,) = HfArgumentParser(PrepareArguments).parse_args_into_dataclasses()
    os.makedirs(args.out_dir, exist_ok=True)
    train_path, val_path = os.path.join(args.out_dir, "train.bin"), os.path.join(args.out_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("train.bin and val.bin already exist, nothing to do")
        return

    # The data files in the order `datasets` loads them, at a pinned revision
    data_files = load_dataset_builder(REPO).config.data_files["train"]
    revision = data_files[0].split("@", 1)[1].split("/", 1)[0]
    rel_paths = [f.split(f"@{revision}/", 1)[1] for f in data_files]
    print(f"{len(rel_paths)} files at revision {revision}", flush=True)
    root = snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns=["data/**"], max_workers=16)
    paths = [os.path.join(root, p) for p in rel_paths]

    # Pass 1: docs per file (cached)
    counts_path = os.path.join(args.out_dir, "doc_counts.json")
    if os.path.exists(counts_path):
        counts = json.load(open(counts_path))
        assert counts["revision"] == revision and counts["files"] == rel_paths, "stale doc_counts.json"
        counts = counts["counts"]
    else:
        with Pool(args.num_proc) as pool:
            counts = pool.map(count_docs, paths, chunksize=4)
        json.dump({"revision": revision, "files": rel_paths, "counts": counts}, open(counts_path, "w"))
    n = sum(counts)
    print(f"{n:,} docs", flush=True)

    # Same permutation as Dataset.shuffle(seed=42): shuffled row k is original row perm[k]
    perm = np.random.default_rng(SEED).permutation(n)
    rank = np.empty(n, dtype=np.int64)
    rank[perm] = np.arange(n)
    del perm
    offsets = np.concatenate([[0], np.cumsum(counts)])
    file_rank = [rank[offsets[i] : offsets[i + 1]] for i in range(len(paths))]
    eval_cut = n - NUM_EVAL_DOCS
    train_cut = int(args.train_frac * eval_cut)
    qc_max = args.qc_max
    if qc_max is not None:
        print(f'qc filter: train docs with dolma2_qc["1"] >= {qc_max} are dropped (unscored docs kept)', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e9)  # silence the length warning; we pack ourselves
    assert tokenizer("x")["input_ids"][0] == tokenizer.bos_token_id

    dedup_stats = None
    assert args.min_train_tokens is None or (args.dedup and qc_max is None), "--min_train_tokens needs --dedup, without --qc_max"
    if args.dedup:
        # Pass 1b: a key per doc (cached), then keep the lowest-rank copy of each text, unless the text is in eval
        keys_path = os.path.join(args.out_dir, "dedup_keys.npz")
        if os.path.exists(keys_path):
            z = np.load(keys_path)
            keys, chars, ctok, cch = z["keys"], z["chars"], int(z["ctok"]), int(z["cch"])
        else:
            with Pool(args.num_proc) as pool:
                res = []
                for i, r in enumerate(pool.imap(doc_keys, paths, chunksize=2)):
                    res.append(r)
                    if i % 500 == 0:
                        print(f"keys: file {i + 1}/{len(paths)}", flush=True)
            keys, chars = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
            ctok, cch = sum(r[2] for r in res), sum(r[3] for r in res)
            del res
            np.savez(keys_path, keys=keys, chars=chars, ctok=ctok, cch=cch)
        assert len(keys) == n
        order = np.lexsort((rank, keys))  # by key, then shuffled rank
        first = np.ones(n, bool)
        first[order[1:]] = keys[order[1:]] != keys[order[:-1]]
        del order
        is_eval = rank >= eval_cut
        in_eval = np.isin(keys, keys[is_eval])
        keep = first & ~in_eval
        chars_per_token = cch / ctok
        # Pick the train docs uniformly over unique texts: a fresh random order of the kept copies. (A prefix of the
        # seed-42 shuffle would favour texts with many copies: with k copies, the chance that one lands in a 5% prefix is
        # 1 - 0.95^k, so the upsampled texts would stay over-represented.)
        tr_ids = np.flatnonzero(keep)
        tr_ids = tr_ids[np.random.default_rng(SEED + 1).permutation(len(tr_ids))]
        if args.min_train_tokens is not None:  # smallest prefix of that order holding min_train_tokens
            m = int(np.searchsorted(np.cumsum(chars[tr_ids]), 1.02 * args.min_train_tokens * chars_per_token)) + 1
            assert m <= len(tr_ids), "not enough unique train docs for --min_train_tokens"
        else:
            m = int(args.train_frac * len(tr_ids))
        selected = np.zeros(n, bool)
        selected[tr_ids[:m]] = True
        train_cut = eval_cut  # the selection is in `selected`; every non-eval rank is eligible
        del tr_ids
        tr = ~is_eval
        dedup_stats = {
            "docs": int(n), "unique_texts": int(first.sum()), "train_docs": int(tr.sum()),
            "train_docs_kept": int((keep & tr).sum()), "train_docs_in_eval": int((in_eval & tr).sum()),
            "train_docs_selected": int(selected.sum()), "train_chars_selected": int(chars[selected].sum()),
            "train_chars": int(chars[tr].sum()), "train_chars_kept": int(chars[keep & tr].sum()),
            "est_unique_train_tokens": int(chars[keep & tr].sum() / chars_per_token), "chars_per_token": chars_per_token,
            "max_copies": int(np.unique(keys, return_counts=True)[1].max()),
        }
        print("dedup:", json.dumps(dedup_stats), flush=True)
        file_keep = [selected[offsets[i] : offsets[i + 1]] for i in range(len(paths))]
        del keys, chars, first, in_eval, is_eval, keep, tr, selected
    print(f"train: shuffled docs [0, {train_cut:,}), eval: [{eval_cut:,}, {n:,})", flush=True)

    # Pass 2: tokenize the train docs (in file order), collect the eval docs
    eval_docs, train_docs = [], 0
    qc_stats = defaultdict(lambda: np.zeros(5, np.int64))

    def train_chunks(pool):
        nonlocal train_docs
        for i, (ids, k, ev, st) in enumerate(pool.imap(process_file, enumerate(paths))):
            train_docs += k
            eval_docs.extend(ev)
            d = rel_paths[i].split("/")[1]
            qc_stats[d.rsplit("-", 1)[0] if d[-4:].isdigit() else d] += st  # common_crawl-<topic>-0014 -> common_crawl-<topic>
            if i % 100 == 0 or i == len(paths) - 1:
                print(f"file {i + 1}/{len(paths)}: {train_docs:,} train docs, {len(eval_docs):,} eval docs", flush=True)
            yield ids

    with Pool(args.num_proc) as pool:
        train_tokens = write_tokens(train_path, train_chunks(pool))
    assert len(eval_docs) == NUM_EVAL_DOCS, len(eval_docs)

    eval_docs.sort()  # shuffled order, as the OJ's held-out set
    with open(os.path.join(args.out_dir, "eval_docs.jsonl"), "w") as f:
        for r, text in eval_docs:
            f.write(json.dumps({"rank": r, "text": text}) + "\n")
    val_tokens = write_tokens(val_path, tokenize([text for _, text in eval_docs[: args.num_val_docs]]))
    json.dump(
        {"revision": revision, "num_docs": n, "train_docs": train_docs, "train_tokens": train_tokens,
         "num_val_docs": args.num_val_docs, "val_tokens": val_tokens, "tokenizer": args.tokenizer, "dtype": "uint32",
         "train_frac": (dedup_stats["train_docs_selected"] / dedup_stats["train_docs_kept"]) if dedup_stats else args.train_frac, "qc_max": args.qc_max, "dedup": dedup_stats},
        open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1,
    )


    if qc_max is not None:  # whole-dataset filter stats per source dir (all docs, train and eval)
        cols = ["docs", "bytes", "scored_docs", "kept_docs", "kept_bytes"]
        per = {src: dict(zip(cols, map(int, v))) for src, v in sorted(qc_stats.items())}
        tot = dict(zip(cols, map(int, sum(qc_stats.values()))))
        json.dump({"qc_max": qc_max, "total": tot, "per_source": per}, open(os.path.join(args.out_dir, "qc_stats.json"), "w"), indent=1)
        print(f"qc filter over the whole dataset: kept {tot['kept_docs']:,} / {tot['docs']:,} docs, "
              f"{tot['kept_bytes'] / tot['bytes']:.1%} of bytes", flush=True)


if __name__ == "__main__":
    main()
