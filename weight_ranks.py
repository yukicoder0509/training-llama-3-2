"""Weight-spectrum summary of saved models, per block type (mean over layers): stable rank ||W||_F^2 / sigma_1^2 and
effective rank exp(entropy of sigma_i / sum sigma) as a fraction of min(rows, cols). Same block types as
llama_train/optim_dynamics.py.

    python weight_ranks.py NAME=/work/$USER/llama_models/run-a NAME2=... > ranks.md
"""

import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file

TYPES = {"embed_tokens": "embed", "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_o",
         "gate_proj": "mlp_gate", "up_proj": "mlp_up", "down_proj": "mlp_down"}
ORDER = list(TYPES.values())


def block_type(name):
    return next((t for k, t in TYPES.items() if k in name), None)


def ranks(w):
    s = torch.linalg.svdvals(w.float().cuda() if torch.cuda.is_available() else w.float())
    p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item()
    return (s.square().sum() / s[0] ** 2).item(), erank / min(w.shape)


rows = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    acc = defaultdict(list)
    for n, w in load_file(f"{path}/model.safetensors").items():
        t = block_type(n)
        if t and w.ndim == 2:
            acc[t].append(ranks(w))
    rows[name] = {t: [sum(v) / len(v) for v in zip(*acc[t])] for t in ORDER if acc[t]}

for i, metric in enumerate(["stable rank", "effective rank (fraction)"]):
    print(f"\n### {metric}\n\n| block | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
    for t in ORDER:
        print(f"| {t} | " + " | ".join(f"{rows[r][t][i]:.3g}" if t in rows[r] else "" for r in rows) + " |")
