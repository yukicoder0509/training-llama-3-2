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

import io
import json
import os
from dataclasses import dataclass, field
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
    tokenizer: str = field(default="unsloth/Llama-3.2-1B", metadata={"help": "Ungated copy of meta-llama/Llama-3.2-1B's tokenizer"})
    train_frac: float = field(default=0.05, metadata={"help": "Fraction of the shuffled train docs to tokenize (5% ~= 7.5B tokens)"})
    num_val_docs: int = field(default=10_000, metadata={"help": "First N of the 50k eval docs tokenized into val.bin (<= 20% allowed)"})
    num_proc: int = field(default=len(os.sched_getaffinity(0)))


# Set in main() before the worker pool forks.
tokenizer = None
file_rank = None  # per file: ranks of its docs in the shuffled order
train_cut = eval_cut = None


def iter_lines(path):
    """Non-blank lines (raw bytes) of a .jsonl.zst file, i.e. the rows `datasets` reads from it."""
    with open(path, "rb") as f:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), buffer_size=1 << 20)
        for line in reader:
            if not line.isspace():
                yield line


def count_docs(path):
    return sum(1 for _ in iter_lines(path))


def tokenize(texts, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        for ids in tokenizer(texts[i : i + batch_size])["input_ids"]:  # starts with BOS
            out.append(np.asarray(ids + [tokenizer.eos_token_id], dtype=np.uint32))
    return out


def process_file(job):
    """Pass 2 for one file: (train tokens, #train docs, [(rank, text) of its eval docs])."""
    i, path = job
    ranks = file_rank[i]
    train_texts, eval_docs = [], []
    for line, r in zip(iter_lines(path), ranks, strict=True):
        if r < train_cut:
            train_texts.append(json.loads(line)["text"])
        elif r >= eval_cut:
            eval_docs.append((int(r), json.loads(line)["text"]))
    ids = tokenize(train_texts)
    return (np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint32)), len(ids), eval_docs


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
    global tokenizer, file_rank, train_cut, eval_cut
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
    print(f"train: shuffled docs [0, {train_cut:,}), eval: [{eval_cut:,}, {n:,})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e9)  # silence the length warning; we pack ourselves
    assert tokenizer("x")["input_ids"][0] == tokenizer.bos_token_id

    # Pass 2: tokenize the train docs (in file order), collect the eval docs
    eval_docs, train_docs = [], 0

    def train_chunks(pool):
        nonlocal train_docs
        for i, (ids, k, ev) in enumerate(pool.imap(process_file, enumerate(paths))):
            train_docs += k
            eval_docs.extend(ev)
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
         "num_val_docs": args.num_val_docs, "val_tokens": val_tokens, "tokenizer": args.tokenizer, "dtype": "uint32"},
        open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1,
    )


if __name__ == "__main__":
    main()
