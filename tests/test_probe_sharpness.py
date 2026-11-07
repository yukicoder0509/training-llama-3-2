"""ProbeSharpness (llama_train/optim_dynamics.py) against exact values, on CPU in seconds:

    python tests/test_probe_sharpness.py

Quadratic probe loss L(w) = (w - c)^T A (w - c) / 2 with a fixed SPD A, so g(w) = A (w - c) and the finite difference
along any step v is exact: probe_sharpness = v^T A v / |v|^2, probe_slope = g(w) . v / |v|, probe_D1 = g(w + v) . v / |v|,
probe_dloss = g(w) . v + v^T A v / 2. The weight is split into two tensors of different block types (q_proj, k_proj) to
check the per-type values against the same formulas restricted to each tensor's coordinates.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llama_train.optim_dynamics import ProbeSharpness  # noqa: E402

torch.manual_seed(0)
d = 8
Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64)).Q
A = Q @ torch.diag(torch.linspace(0.1, 5.0, d, dtype=torch.float64)) @ Q.T
c = torch.randn(d, dtype=torch.float64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
        self.k_proj = torch.nn.Linear(5, 1, bias=False, dtype=torch.float64)

    def w(self):
        return torch.cat([self.q_proj.weight, self.k_proj.weight], 1).reshape(-1)


model = Model()


def loss_fn(j):
    r = model.w() - c
    return r @ A @ r / 2


probe = ProbeSharpness(model, loss_fn)
w0 = model.w().detach().clone()
v = 0.1 * torch.randn(d, dtype=torch.float64)
probe.before_step()
with torch.no_grad():  # the "optimizer step"
    model.q_proj.weight.add_(v[:3].reshape(1, 3))
    model.k_proj.weight.add_(v[3:].reshape(1, 5))
rec = probe.after_step()

g0, g1 = A @ (w0 - c), A @ (w0 + v - c)
vv = v @ v
exact = {
    "global/probe_sharpness": (v @ A @ v / vv).item(),
    "global/probe_slope": (g0 @ v / vv.sqrt()).item(),
    "global/probe_D1": (g1 @ v / vv.sqrt()).item(),
    "global/probe_dloss": (g0 @ v + v @ A @ v / 2).item(),
    "global/probe_loss": ((w0 - c) @ A @ (w0 - c) / 2).item(),
    # per type: (g1 - g0) restricted to q_proj's coordinates, dotted with v there, over |v_q|^2
    "type/attn_q/probe_sharpness": ((g1 - g0)[:3] @ v[:3] / (v[:3] @ v[:3])).item(),
}
for k, x in exact.items():
    print(f"{k}: {rec[k]:.10f} vs exact {x:.10f}")
    assert abs(rec[k] - x) < 1e-5 * max(1.0, abs(x)), k  # the probe keeps fp32 copies
print("OK")
