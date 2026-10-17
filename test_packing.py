"""Check train.py --packing doc on 1 GPU: a packed doc-mode forward must equal running each document on its own.

    sbatch --gpus-per-node=1 --cpus-per-task=12 --time=0:15:00 --wrap="source .venv/bin/activate && python test_packing.py"

Runs train.py's doc packing collate (llama_train.data.DocPackingCollate) on val.bin blocks, through a random-init Llama 3.2 1B
with FlashAttention 3, and compares per-token losses with separate per-segment forwards (and with --packing concat,
which must differ after the first document).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig

from llama_train.data import DocPackingCollate

BLOCK_SIZE = 8192
config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=4, num_attention_heads=32,
    num_key_value_heads=8, max_position_embeddings=131072, rope_theta=500000.0, bos_token_id=128000, eos_token_id=128001,
    rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192}, use_cache=False)
collate = DocPackingCollate(config.bos_token_id, BLOCK_SIZE)

torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_3", torch_dtype=torch.bfloat16).cuda().eval()
tokens = np.memmap(os.path.expandvars("/work/$USER/dolma3_llama/val.bin"), dtype=np.uint32, mode="r")
feats = [{"input_ids": torch.from_numpy(tokens[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].astype(np.int64))} for i in (3, 7)]


def token_losses(logits, ids):
    return F.cross_entropy(logits[:-1].float(), ids[1:], reduction="none")


with torch.no_grad():
    b = collate(feats)
    print("segments per block:", [(f["input_ids"] == config.bos_token_id).sum().item() for f in feats],
          "cu_seq_lens:", b["cu_seq_lens_q"][:6].tolist(), "...", b["cu_seq_lens_q"][-3:].tolist())
    gpu = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items() if k != "labels"}
    packed = model(**gpu).logits
    concat = model(input_ids=gpu["input_ids"]).logits
    worst = 0.0
    for r in range(len(feats)):
        ids = gpu["input_ids"][r]
        starts = torch.cat([torch.tensor([0], device="cuda"), (ids == config.bos_token_id).nonzero().view(-1)]).unique().tolist()
        bounds = starts + [BLOCK_SIZE]
        lp, lc = token_losses(packed[r], ids), token_losses(concat[r], ids)
        for a, e in zip(bounds[:-1], bounds[1:]):
            if e - a < 2:
                continue
            alone = token_losses(model(input_ids=ids[a:e][None]).logits[0], ids[a:e])
            worst = max(worst, (lp[a:e - 1] - alone).abs().max().item())
        print(f"row {r}: {len(starts)} segments, packed mean loss {lp.mean():.4f}, concat {lc.mean():.4f}, "
              f"concat vs packed max diff after segment 1: {(lc[bounds[1]:] - lp[bounds[1]:]).abs().max().item() if len(starts) > 1 else 0:.3f}")
    print(f"max |packed - per-document| token loss: {worst:.4f}")
    assert worst < 0.05, "doc packing does not match per-document forwards"
    print("OK")
