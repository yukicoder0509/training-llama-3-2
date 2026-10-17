"""Dolma 3 token files (prepare_data.py): flat uint32 <|begin_of_text|> doc <|end_of_text|>, packed into blocks."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlocks(Dataset):
    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")  # lazy, no RAM blowup
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i * self.block_size : (i + 1) * self.block_size].astype(np.int64))
        return {"input_ids": x, "labels": x}  # model shifts labels internally


class DocPackingCollate:
    """--packing doc: each block is split at its <|begin_of_text|> tokens (the block's first segment may be the tail of a
    document cut by the previous block). position_ids restart at 0 per segment (RoPE), and cu_seq_lens (segment starts in
    the micro-batch flattened to one sequence, plus the end) make HF's FlashAttention call the varlen kernel, so attention
    stays inside each document. max_length is the block-size upper bound (a Python int: no per-batch recompile / host
    sync). Labels are unchanged (an <|end_of_text|> still predicts the next <|begin_of_text|>), so the loss counts the same
    tokens."""

    def __init__(self, bos_id, block_size):
        self.bos_id, self.block_size = bos_id, block_size

    def __call__(self, features):
        ids = torch.stack([f["input_ids"] for f in features])
        starts = ids == self.bos_id
        starts[:, 0] = True
        first = torch.where(starts, torch.arange(ids.size(1)), 0).cummax(1).values  # index of the segment's first token
        flat_starts = starts.reshape(-1).nonzero().view(-1)
        cu = torch.cat([flat_starts, torch.tensor([ids.numel()])]).to(torch.int32)
        return {"input_ids": ids, "labels": ids.clone(), "position_ids": torch.arange(ids.size(1)) - first,
                "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": self.block_size, "max_length_k": self.block_size}


def load_datasets(data_dir, block_size):
    """train.bin, val.bin, and dev.bin if present (our own dev set, select_rows.py dirs: evaluated next to val.bin,
    logged as eval/dev_loss, eval/dev_perplexity)."""
    train_ds = TokenBlocks(os.path.join(data_dir, "train.bin"), block_size)
    val_ds = TokenBlocks(os.path.join(data_dir, "val.bin"), block_size)
    dev_ds = TokenBlocks(os.path.join(data_dir, "dev.bin"), block_size) if os.path.exists(os.path.join(data_dir, "dev.bin")) else None
    print(f"Train: {len(train_ds):,} blocks ({len(train_ds) * block_size / 1e9:.2f}B tokens), val: {len(val_ds):,} blocks"
          + (f", dev: {len(dev_ds):,} blocks" if dev_ds is not None else ""))
    return train_ds, val_ds, dev_ds
