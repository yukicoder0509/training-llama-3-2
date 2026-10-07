"""Muon for the transformer blocks' 2D weight matrices + AdamW for everything else, as one optimizer.

Split as in modded-nanogpt: Muon (torch.optim.Muon: Nesterov momentum, 5 Newton-Schulz steps in bf16) for
h.*.attn.c_attn, h.*.attn.c_proj, h.*.mlp.c_fc, h.*.mlp.c_proj; AdamW for wte (tied to lm_head), wpe, biases and
LayerNorms, with no weight decay on the 1D params (as HF Trainer's AdamW).

HF Trainer takes one optimizer and builds one LR scheduler on it, so MuonWithAdamW exposes both optimizers'
param groups (the same dicts) and steps both: the WSD schedule scales the Muon and AdamW LRs by the same factor.

Muon uses adjust_lr_fn="match_rms_adamw" (update scaled by 0.2 * sqrt(max(rows, cols))): HF GPT-2's Conv1D stores
weights as (in, out), the transpose of nn.Linear, and this scaling is symmetric in the two dims, unlike the default
"original" sqrt(max(1, rows / cols)). It also puts the Muon LR on roughly the AdamW scale (Moonlight).
"""

import torch


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(self, muon, adamw):
        self.muon, self.adamw = muon, adamw
        super().__init__([p for o in (adamw, muon) for g in o.param_groups for p in g["params"]], {})
        # AdamW groups first, so the Trainer's logged learning_rate stays the AdamW LR as in earlier runs
        self.param_groups = adamw.param_groups + muon.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.adamw.step()
        self.muon.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none)
        self.muon.zero_grad(set_to_none)

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        self.adamw.load_state_dict(state_dict["adamw"])
        self.muon.load_state_dict(state_dict["muon"])


def build_muon_adamw(model, muon_lr, muon_momentum, adamw_lr, beta2, weight_decay):
    """Returns (optimizer, names of the Muon params, names of the AdamW params)."""
    muon_params, decay, no_decay, muon_names, adamw_names = [], [], [], [], []
    for name, p in model.named_parameters():  # tied lm_head.weight is listed once, as transformer.wte.weight
        if p.ndim == 2 and name.startswith("transformer.h."):
            muon_params.append(p)
            muon_names.append(name)
        else:
            (decay if p.ndim >= 2 else no_decay).append(p)
            adamw_names.append(name)
    muon = torch.optim.Muon(muon_params, lr=muon_lr, momentum=muon_momentum, nesterov=True,
                            weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=adamw_lr, betas=(0.9, beta2), eps=1e-8)
    return MuonWithAdamW(muon, adamw), muon_names, adamw_names
