"""SOAP with its per-parameter work split across the DDP ranks.

The official soap.py runs the full optimizer step on every GPU. For Llama 3.2 1B that means ~30 GB of fp32
preconditioner state per GPU (8192 x 8192 for each MLP matrix) and ~90 TFLOP of fp32 matmuls per step, repeated
identically on all ranks. After DDP's all-reduce every rank holds the same gradients, so each parameter can instead be
updated by one owner rank, which then broadcasts the new values. The result is the same as the replicated
optimizer, and each rank has ~1/world_size of the state and work.

Owners are assigned once, greedily by estimated cost (numel x sum of preconditioned dims), in the same order on every rank.
Non-owned parameters get p.grad = None for the step, which soap.py skips. The Trainer zeroes the grads right after
the step and computes grad norm / clipping before it.
"""

import torch
import torch.distributed as dist

from soap import SOAP


class DistributedSOAP(SOAP):
    owners = None

    def _assign_owners(self):
        ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        params = [(p, g["max_precond_dim"], g["precondition_1d"]) for g in self.param_groups for p in g["params"]]

        def cost(p, max_dim, pre_1d):
            dims = [d for d in p.shape if d <= max_dim] if p.dim() > 1 or pre_1d else []
            return p.numel() * (1 + sum(dims))

        load = [0.0] * ws
        self.owners = [None] * len(params)
        for i in sorted(range(len(params)), key=lambda i: -cost(*params[i])):
            r = min(range(ws), key=load.__getitem__)
            self.owners[i] = r
            load[r] += cost(*params[i])
        self.params_in_order = [p for p, _, _ in params]
        self.world_size, self.rank = ws, (dist.get_rank() if ws > 1 else 0)
        if self.rank == 0:
            print(f"DistributedSOAP: {len(params)} tensors over {ws} ranks, relative load per rank "
                  f"{[round(l / max(load), 2) for l in load]}")

    @torch.no_grad()
    def step(self, closure=None):
        if self.owners is None:
            self._assign_owners()
        for p, r in zip(self.params_in_order, self.owners):
            if r != self.rank:
                p.grad = None
        loss = super().step(closure)
        if self.world_size > 1:  # each tensor was updated only on its owner
            handles = [dist.broadcast(p.data, src=r, async_op=True) for p, r in zip(self.params_in_order, self.owners)]
            for h in handles:
                h.wait()
        return loss
