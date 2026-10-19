"""GradNoiseScale (llama_train/optim_dynamics.py) against an exact noise scale, on CPU in seconds:

    python tests/test_grad_noise.py

Linear regression y = x . w* + s eps, x ~ N(0, I_d), loss = mean over b units of (x . w - y)^2 / 2, weights at
w = w* + delta. Per-unit gradient g = x (x . delta - s eps), so G = delta and (Isserlis) tr(Sigma) = (d + 1)|delta|^2 + d s^2:
B = (d + 1) + d s^2 / |delta|^2. The weight is split into two tensors of different block types to check the per-group sums
add up to the global ones.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llama_train.optim_dynamics import GradNoiseScale  # noqa: E402

torch.manual_seed(0)
d, s, b, trials = 8, 0.5, 64, 20000
w_star = torch.randn(d, dtype=torch.float64)
delta = torch.randn(d, dtype=torch.float64)
exact = (d + 1) + d * s**2 / delta.pow(2).sum().item()


class Model(torch.nn.Module):  # two tensors so the per-type groups are exercised: q_proj gets 3 coordinates, k_proj 5
    def __init__(self):
        super().__init__()
        w = w_star + delta
        self.q_proj = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
        self.k_proj = torch.nn.Linear(5, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.q_proj.weight.copy_(w[:3].reshape(1, 3))
            self.k_proj.weight.copy_(w[3:].reshape(1, 5))


model = Model()


def loss_fn(i):
    x = torch.randn(b, d, dtype=torch.float64)
    y = x @ w_star + s * torch.randn(b, dtype=torch.float64)
    w = torch.cat([model.q_proj.weight, model.k_proj.weight], 1).reshape(-1)
    return ((x @ w - y).pow(2) / 2).mean(), b


gns = GradNoiseScale(model, loss_fn)
recs = [gns.measure() for _ in range(trials)]
mean = {k: sum(r[k] for r in recs) / trials for k in recs[0] if k.endswith(("G2", "trS"))}
B = mean["global/gns_trS"] / mean["global/gns_G2"]
print(f"B estimate {B:.3f} (window of {trials} measurements) vs exact {exact:.3f}; "
      f"|G|^2 {mean['global/gns_G2']:.4f} vs {delta.pow(2).sum().item():.4f}")
assert abs(B / exact - 1) < 0.03, (B, exact)
assert abs(mean["global/gns_G2"] / delta.pow(2).sum().item() - 1) < 0.03
for k in ("gns_G2", "gns_trS"):  # groups are additive
    tot = recs[0][f"type/attn_q/{k}"] + recs[0][f"type/attn_k/{k}"]
    assert abs(tot - recs[0][f"global/{k}"]) < 1e-9 * max(1, abs(tot)), k
assert recs[0]["global/gns_b"] == b
# per-type exact |G|^2: the coordinates of delta in that tensor
assert abs(mean["type/attn_q/gns_G2"] / delta[:3].pow(2).sum().item() - 1) < 0.05
print("OK")
