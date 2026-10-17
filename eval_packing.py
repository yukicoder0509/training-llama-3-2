"""Eval loss of saved models on val.bin under both attention protocols, for comparing --packing runs (1 GPU).

    sbatch --gpus-per-node=1 --cpus-per-task=12 --time=0:30:00 --wrap="source ./.venv/bin/activate && python eval_packing.py /work/$USER/llama_models/run-a /work/$USER/llama_models/run-b"

concat: each 8192 block attends across document boundaries (train.py --packing concat, the in-training eval of those runs).
doc: each document attends only to itself, positions restart at <|begin_of_text|> (--packing doc; closer to scoring
documents one by one). Same tokens and targets in both: the mean CE over every next-token prediction in the blocks.
"""

import os
import sys

import numpy as np
import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers import AutoModelForCausalLM

from llama_train.data import DocPackingCollate

BLOCK_SIZE = 8192
collate = DocPackingCollate(128000, BLOCK_SIZE)

tokens = np.memmap(os.path.expandvars("/work/$USER/dolma3_llama/val.bin"), dtype=np.uint32, mode="r")
n_blocks = len(tokens) // BLOCK_SIZE
ce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
BATCH = 4

for path in sys.argv[1:]:
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="flash_attention_3", dtype=torch.bfloat16).cuda().eval()
    res = {}
    for mode in ("concat", "doc"):
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, n_blocks, BATCH):
                feats = [{"input_ids": torch.from_numpy(tokens[j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE].astype(np.int64))}
                         for j in range(i, min(i + BATCH, n_blocks))]
                b = collate(feats)
                kw = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items() if k != "labels"}
                if mode == "concat":
                    kw = {"input_ids": kw["input_ids"]}
                h = model.model(**kw).last_hidden_state
                ids = kw["input_ids"]
                total += ce(model.lm_head.weight, h[:, :-1].reshape(-1, h.size(-1)), ids[:, 1:].reshape(-1)).item()
                count += ids[:, 1:].numel()
        res[mode] = total / count
    print(f"{os.path.basename(path.rstrip('/')):32s} {n_blocks} blocks  concat {res['concat']:.4f} (ppl {np.exp(res['concat']):.2f})  "
          f"doc {res['doc']:.4f} (ppl {np.exp(res['doc']):.2f})", flush=True)
