"""Tokenize C4 `en` with the GPT-2 tokenizer into flat uint16 token files.

Writes `train.bin` and `val.bin` under --out_dir. Each document is followed by
an EOS token so documents can be packed into fixed-length training blocks.

Train docs come from either the first --num_train_shards raw C4 `en` train shards (~150M GPT-2
tokens each; 20 shards ~= 3B tokens), or the *.jsonl.gz files in --input_dir written by
dedup_filter.py (near-deduped and/or filtered C4). Files are streamed straight from the .json.gz
files instead of `load_dataset`, whose arrow cache would not fit in the /work quota.
The val split is never deduped or filtered.

(The DCLM fastText quality ranking that used to be here was removed: keeping the top ~10% hurt C4
validation perplexity, see experiments.md 2026-10-04.)
"""

import glob
import gzip
import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class PrepareArguments:
    out_dir: str = field(default=os.path.expandvars("/work/$USER/c4_gpt2"))
    num_train_shards: int = field(default=20, metadata={"help": "Out of 1024 C4 en train shards (ignored with --input_dir)."})
    input_dir: str = field(
        default=None, metadata={"help": "Tokenize the *.jsonl.gz files here (dedup_filter.py output) instead of raw C4 shards."}
    )
    num_val_docs: int = field(
        default=5000,
        metadata={"help": "Docs from the validation split (shuffled with seed 42, like the OJ) used for in-training eval."},
    )
    num_proc: int = field(default=len(os.sched_getaffinity(0)))


# Set in main() before the worker pool forks.
tokenizer = None


def tokenize(texts, batch_size=1000):
    # Batched: a whole shard of token lists as Python ints would take ~6GB per worker.
    out = []
    for i in range(0, len(texts), batch_size):
        ids = tokenizer(texts[i : i + batch_size])["input_ids"]
        out.extend(np.asarray(x + [tokenizer.eos_token_id], dtype=np.uint16) for x in ids)
    return out


def read_shard(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line)["text"] for line in f]


def tokenize_file(path):
    ids = tokenize(read_shard(path))
    return (np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint16)), len(ids)


def write_bin(ds, path):
    # Sequential writes, not np.memmap: memmap writes to the /work network FS were silently
    # lost (train.bin came out with its last ~28% all zeros).
    total = int(np.sum(ds["len"], dtype=np.uint64))
    num_chunks = max(1, min(1024, len(ds)))
    written = 0
    with open(path + ".tmp", "wb") as f:
        for i in range(num_chunks):
            chunk = ds.shard(num_shards=num_chunks, index=i, contiguous=True).with_format("numpy")
            if len(chunk):
                ids = np.concatenate(chunk["ids"]).astype(np.uint16)
                f.write(ids.tobytes())
                written += len(ids)
        f.flush()
        os.fsync(f.fileno())
    verify_and_publish(path, written, total)


def verify_and_publish(path, written, total):
    assert written == total, f"wrote {written:,} tokens, expected {total:,}"
    check = np.memmap(path + ".tmp", dtype=np.uint16, mode="r")
    assert len(check) == total and check[-4096:].any(), f"{path}.tmp failed verification"
    os.replace(path + ".tmp", path)  # only a complete file ever appears under the final name
    print(f"wrote {written:,} tokens to {path}")


def write_train_bin(train_files, args, path):
    written = docs = 0
    with Pool(args.num_proc) as pool, open(path + ".tmp", "wb") as f:
        for i, (ids, n) in enumerate(pool.imap(tokenize_file, train_files)):  # in file order
            f.write(ids.tobytes())
            written += len(ids)
            docs += n
            print(f"tokenized file {i + 1}/{len(train_files)}: {docs:,} docs / {written:,} tokens so far", flush=True)
        f.flush()
        os.fsync(f.fileno())
    verify_and_publish(path, written, written)  # total = what the workers returned; checks the file itself


def main():
    global tokenizer
    (args,) = HfArgumentParser(PrepareArguments).parse_args_into_dataclasses()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)  # datasets resolves paths against cwd; don't depend on the (NFS) submit dir

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)  # silence the >1024 length warning; we pack ourselves
    eos = tokenizer.eos_token_id

    def tokenize_batch(batch):
        ids = [x + [eos] for x in tokenizer(batch["text"])["input_ids"]]
        return {"ids": [np.asarray(x, dtype=np.uint16) for x in ids], "len": [len(x) for x in ids]}

    val_path, train_path = os.path.join(args.out_dir, "val.bin"), os.path.join(args.out_dir, "train.bin")
    if os.path.exists(val_path) and os.path.exists(train_path):
        print("val.bin and train.bin already exist, nothing to do")
        return

    if not os.path.exists(val_path):
        val_files = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
        val = load_dataset("allenai/c4", data_files={"validation": val_files}, split="validation")
        val = val.shuffle(seed=42).select(range(args.num_val_docs))
        val = val.map(tokenize_batch, batched=True, num_proc=args.num_proc, remove_columns=val.column_names)
        write_bin(val, val_path)

    if args.input_dir:
        train_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl.gz")))
        assert train_files, f"no *.jsonl.gz in {args.input_dir}"
    else:
        # Resolve (and if needed download) shards here: hf_hub_download inside forked workers deadlocks.
        train_files = [
            hf_hub_download("allenai/c4", f"en/c4-train.{i:05d}-of-01024.json.gz", repo_type="dataset")
            for i in range(args.num_train_shards)
        ]
    write_train_bin(train_files, args, train_path)


if __name__ == "__main__":
    main()
