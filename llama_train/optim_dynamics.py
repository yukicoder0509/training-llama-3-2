"""
optim_dynamics.py - log per-block optimizer / training dynamics.

Every `log_every` optimizer steps the logger opens a 2-step window (steps a=b-1 and b)
and measures, for every parameter tensor, then aggregates per block type
(attn_q, mlp_down, embed, lm_head, norm, ...), per (layer, type), and globally:

  gradient   grad_norm, grad_max, grad_cos_prev       (cosine of consecutive gradients)
  update     upd_rel = ||dW||/||W||, upd_rms, upd_max, upd_cos_prev (consecutive updates),
             upd_grad_align = cos(dW, -g)              (1.0 = plain SGD direction)
  curvature  D1 = g_b . v_a / ||v_a||                  (MiniCPM, sec. 4.4)
             dir_sharpness = (g_b - g_a) . v_a / ||v_a||^2   (~ v^T H v / ||v||^2, MiniCPM's D2)
             curvature K = |D2| / (1 + D1^2)^1.5
  spectral   (2-D tensors, every `spectral_every` steps)
             w_sigma1, w_srank, upd_sigma1, upd_srank, upd_spec_rel = sigma1(dW)/sigma1(W),
             upd_row_cv = std/mean of per-row (per-output-neuron) update norms
  eff. rank  (2-D tensors, on spectral steps, needs a full SVD; erank=True)
             upd_erank, upd_erank_frac  of the one-step update dW
             delta_erank, delta_erank_frac, delta_rel  of the accumulated change
                 W_t - W_{t-delta} over delta = log_every steps (track_delta=True)
             erank(A) = exp(-sum_i p_i log p_i),  p_i = sigma_i^2 / sum_j sigma_j^2,
             i.e. the exponentiated spectral entropy with alpha = 2 from
             Huang et al., "Mechanism-Driven Monitors for Preemptive Detection of LLM
             Training Instability", arXiv:2606.28116 (2026), sec. 3.2.
             *_frac = erank / min(rows, cols), comparable across matrix shapes.
             The paper reports that the spectrum of dW is a more sensitive and
             explainable instability signal than raw-weight statistics, stable rank or
             ||dW||_F. (Its QK-product variant, Delta_2 = dWq Wk^T + Wq dWk^T, is not
             implemented here.)

Group values for norms / cosines / curvature are exact: they are computed from summed
dot products, i.e. as if the group's tensors were flattened into one vector (the
"global" group reproduces MiniCPM's whole-model definitions). Spectral values are
means over the matrices in the group.

Notes
- dW is the update the optimizer actually applied, so it includes weight decay.
- Gradients are read where the optimizer reads them (after clipping in HF Trainer).
- g_a and g_b come from different mini-batches, so dir_sharpness / K contain gradient
  noise of size ~||noise|| / ||v_a||. Look at medians over many log points, and ignore
  them where the LR is near zero (end of decay): there the tiny update makes the noise
  term blow up.
- Memory: during the window the logger holds about two extra parameter-sized buffers
  (use buffer_device="cpu" and/or include= to limit this on big models).
  track_delta=True keeps one more persistent copy of every 2-D parameter.
- Cost: erank needs torch.linalg.svdvals per matrix. On large models raise
  spectral_every, restrict with include=, or set erank=False.
- FSDP2 / DTensor: sums and maxima are all-reduced across shards; spectral metrics
  are skipped for sharded tensors. This path is untested.

Usage (custom loop):
    logger = OptimDynamicsLogger(model, log_every=100, optimizer=opt)
    loss.backward(); clip_grad_norm_(...)
    logger.before_step()          # grads ready, before opt.step()
    opt.step()
    logger.after_step()           # BEFORE opt.zero_grad()
    opt.zero_grad()

Usage (HF Trainer):
    trainer.add_callback(OptimDynamicsCallback(log_every=100, out_path="dyn.jsonl"))
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from typing import Callable, Iterable

import torch

try:  # DTensor support is optional
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover
    DTensor = ()  # type: ignore

# ---------------------------------------------------------------------------
# parameter classification
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[tuple[str, str]] = [
    ("pos_embed", r"(?:^|\.)(wpe|position_embeddings|pos_embed)\.weight$"),
    ("embed", r"(?:^|\.)(embed_tokens|wte|word_embeddings|tok_embeddings|embed)\.weight$"),
    ("lm_head", r"(?:^|\.)(lm_head|unembed|output)\.weight$"),
    ("attn_qkv", r"(?:^|\.)(qkv_proj|c_attn|query_key_value|Wqkv)\."),
    ("attn_q", r"(?:^|\.)(q_proj|query|wq)\."),
    ("attn_k", r"(?:^|\.)(k_proj|key|wk)\."),
    ("attn_v", r"(?:^|\.)(v_proj|value|wv)\."),
    ("attn_o", r"(?:^|\.)(o_proj|out_proj|wo|dense)\.|attn\.c_proj\."),
    ("mlp_gate", r"(?:^|\.)(gate_proj|w1)\."),
    ("mlp_up", r"(?:^|\.)(up_proj|c_fc|w3|fc1|dense_h_to_4h)\."),
    ("mlp_down", r"(?:^|\.)(down_proj|w2|fc2|dense_4h_to_h)\.|mlp\.c_proj\."),
]
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)\.")
_NORM_RE = re.compile(r"norm|(?:^|\.)ln_|layernorm", re.IGNORECASE)


def classify(name: str, p: torch.Tensor, rules=DEFAULT_RULES) -> tuple[int, str]:
    """Return (layer index or -1, block type) for a parameter name."""
    m = _LAYER_RE.search(name)
    layer = int(m.group(1)) if m else -1
    if p.ndim < 2:
        if _NORM_RE.search(name):
            return layer, "norm"
        return layer, "bias" if name.endswith("bias") else "vector"
    for typ, pat in rules:
        if re.search(pat, name):
            return layer, typ
    parts = name.split(".")
    return layer, f"other_{parts[-2] if len(parts) > 1 else parts[0]}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# per-tensor additive statistics (summed across tensors in a group, and across shards)
_SUM_FIELDS = ["n", "ww", "gg_a", "gg_b", "g_ab", "vv_a", "ga_va", "gb_va", "vv_b", "v_ab"]
_MAX_FIELDS = ["gmax_b", "vmax_a"]
_SPEC_FIELDS = ["w_sigma1", "w_srank", "upd_sigma1", "upd_srank", "upd_spec_rel", "upd_row_cv",
                "upd_erank", "upd_erank_frac", "delta_erank", "delta_erank_frac", "delta_rel"]


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _is_sharded(t: torch.Tensor) -> bool:
    return isinstance(t, DTensor) and any(not pl.is_replicate() for pl in t.placements)


def _as_matrix(t: torch.Tensor) -> torch.Tensor:
    return t if t.ndim == 2 else t.reshape(t.shape[0], -1)


def _sigma1(A: torch.Tensor, v0: torch.Tensor | None, iters: int):
    """Largest singular value by warm-started power iteration. Returns (sigma1, v)."""
    n = A.shape[1]
    v = v0 if v0 is not None else torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + 1e-30)
    s = torch.zeros((), device=A.device, dtype=A.dtype)
    for _ in range(iters):
        u = A @ v
        u = u / (u.norm() + 1e-30)
        v = A.T @ u
        s = v.norm()
        v = v / (s + 1e-30)
    return s, v


def effective_rank(sv: torch.Tensor) -> torch.Tensor:
    """exp of the Shannon entropy of p_i = sigma_i^2 / sum_j sigma_j^2 (alpha = 2),
    as in Huang et al., arXiv:2606.28116, sec. 3.2. Takes singular values."""
    s2 = sv.double().pow(2)
    total = s2.sum()
    if total <= 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    p = s2 / total
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum())


def _safe(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------


class OptimDynamicsLogger:
    def __init__(
        self,
        model: torch.nn.Module,
        log_every: int = 100,
        spectral_every: int | None = None,
        out_path: str | None = "optim_dynamics.jsonl",
        optimizer: torch.optim.Optimizer | None = None,
        include: str | None = None,
        exclude: str | None = None,
        rules: list[tuple[str, str]] | None = None,
        per_layer: bool = True,
        power_iters: int = 6,
        exact_spectral: bool = False,
        erank: bool = True,
        track_delta: bool = False,
        buffer_device: str | torch.device | None = None,
        buffer_dtype: torch.dtype = torch.float32,
        log_fn: Callable[[dict, int], None] | None = None,
        rank0_only: bool = True,
    ):
        """
        log_every       measure every N optimizer steps (N >= 2)
        spectral_every  spectral metrics every M steps (multiple of log_every; None = log_every,
                        0 = never)
        include/exclude regex on parameter names
        exact_spectral  use full SVD (torch.linalg.matrix_norm) instead of power iteration
        erank           effective rank of the one-step update (full SVD per matrix)
        track_delta     also effective rank of W_t - W_{t-log_every} (keeps a snapshot)
        buffer_device   where to keep the window buffers (e.g. "cpu" to save GPU memory)
        log_fn          optional callable(flat_metrics_dict, step), e.g. a wandb.log wrapper
        """
        if log_every < 2:
            raise ValueError("log_every must be >= 2 (the window spans two steps)")
        self.log_every = log_every
        self.spectral_every = log_every if spectral_every is None else spectral_every
        self.out_path = out_path
        self.optimizer = optimizer
        self.per_layer = per_layer
        self.power_iters = power_iters
        self.exact_spectral = exact_spectral
        self.erank = erank
        self.track_delta = track_delta
        self._snap: dict[str, torch.Tensor] = {}  # W at the previous window (track_delta)
        self.buffer_device = buffer_device
        self.buffer_dtype = buffer_dtype
        self.log_fn = log_fn
        self.rank0_only = rank0_only
        rules = rules or DEFAULT_RULES

        inc = re.compile(include) if include else None
        exc = re.compile(exclude) if exclude else None
        self.params: list[tuple[str, torch.Tensor, int, str]] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if inc and not inc.search(name):
                continue
            if exc and exc.search(name):
                continue
            layer, typ = classify(name, p, rules)
            self.params.append((name, p, layer, typ))

        self.t = 0  # completed optimizer steps
        self._buf: dict[str, dict[str, torch.Tensor]] = {}
        self._sums: torch.Tensor | None = None
        self._maxs: torch.Tensor | None = None
        self._spec: dict[str, list[float]] = {}
        self._pi_vecs: dict[tuple[str, str], torch.Tensor] = {}
        self.last_record: dict | None = None

    # -- schedule -----------------------------------------------------------
    def _phase(self) -> str | None:
        if (self.t + 2) % self.log_every == 0:
            return "a"
        if (self.t + 1) % self.log_every == 0 and self.t > 0:
            return "b"
        return None

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        return _local(t).detach().to(device=self.buffer_device or _local(t).device,
                                     dtype=self.buffer_dtype, copy=True)

    # -- public API -----------------------------------------------------------
    @torch.no_grad()
    def before_step(self) -> None:
        """Call after backward (and clipping), before optimizer.step()."""
        phase = self._phase()
        if phase == "a":
            self._buf = {}
            for name, p, _, _ in self.params:
                if p.grad is None:
                    continue
                self._buf[name] = {
                    "g_a": self._store(p.grad),
                    # keep W exactly (own dtype) so W_after - W_before is the true update
                    "w_before": _local(p).detach().to(device=self.buffer_device or _local(p).device,
                                                      copy=True),
                }
        elif phase == "b":
            self._measure_before_b()

    @torch.no_grad()
    def after_step(self) -> dict | None:
        """Call after optimizer.step() and BEFORE optimizer.zero_grad()."""
        phase = self._phase()
        record = None
        if phase == "a":
            for name, p, _, _ in self.params:
                b = self._buf.get(name)
                if b is None:
                    continue
                w = _local(p).detach()
                b["v_a"] = (w.float() - b.pop("w_before").to(w.device).float()).to(
                    device=self.buffer_device or w.device, dtype=self.buffer_dtype)
        elif phase == "b":
            record = self._measure_after_b()
        self.t += 1
        return record

    # -- measurement ------------------------------------------------------------
    def _measure_before_b(self) -> None:
        P = len(self.params)
        dev = self.params[0][1].device if P else torch.device("cpu")
        sums = torch.zeros(P, len(_SUM_FIELDS), dtype=torch.float64, device=dev)
        maxs = torch.zeros(P, len(_MAX_FIELDS), dtype=torch.float64, device=dev)
        do_spec = self.spectral_every > 0 and (self.t + 1) % self.spectral_every == 0
        self._spec = {}
        for i, (name, p, _, _) in enumerate(self.params):
            b = self._buf.get(name)
            if b is None or p.grad is None or "v_a" not in b:
                self._buf.pop(name, None)
                continue
            w = _local(p).detach().float()
            g_b = _local(p.grad).detach().float()
            g_a = b.pop("g_a").to(w.device).float()
            v_a = b["v_a"].to(w.device).float()
            row = sums[i]
            row[0] = w.numel()
            row[1] = w.pow(2).sum()
            row[2] = g_a.pow(2).sum()
            row[3] = g_b.pow(2).sum()
            row[4] = (g_a * g_b).sum()
            row[5] = v_a.pow(2).sum()
            row[6] = (g_a * v_a).sum()
            row[7] = (g_b * v_a).sum()
            maxs[i, 0] = g_b.abs().max()
            maxs[i, 1] = v_a.abs().max()
            matrix = p.ndim >= 2 and not _is_sharded(p)
            delta = None
            if matrix and self.track_delta:
                prev = self._snap.get(name)
                if prev is not None and do_spec:
                    # accumulated change over the last log_every steps: W_t - W_{t-delta}
                    delta = w - prev.to(w.device).float()
                # exact copy in the parameter's own dtype, refreshed every window
                self._snap[name] = _local(p).detach().to(
                    device=self.buffer_device or _local(p).device, copy=True)
            if do_spec and matrix:
                self._spec[name] = self._spectral(
                    name, _as_matrix(w), _as_matrix(v_a),
                    _as_matrix(delta) if delta is not None else None)
            del delta
            del g_a, g_b, v_a
            b["w_before"] = _local(p).detach().to(device=self.buffer_device or _local(p).device,
                                                  copy=True)
        self._sums, self._maxs = sums, maxs

    def _spectral(self, name: str, W: torch.Tensor, V: torch.Tensor,
                  D: torch.Tensor | None = None) -> list[float]:
        nan = float("nan")
        min_dim = min(V.shape)
        v_sv = torch.linalg.svdvals(V) if self.erank else None
        if self.exact_spectral:
            ws = torch.linalg.matrix_norm(W, ord=2)
        else:
            ws, self._pi_vecs[(name, "w")] = _sigma1(W, self._pi_vecs.get((name, "w")), self.power_iters)
        if v_sv is not None:  # full SVD already done: exact sigma1 for free
            vs = v_sv[0]
        elif self.exact_spectral:
            vs = torch.linalg.matrix_norm(V, ord=2)
        else:
            vs, self._pi_vecs[(name, "v")] = _sigma1(V, self._pi_vecs.get((name, "v")), self.power_iters)
        wf2, vf2 = W.pow(2).sum(), V.pow(2).sum()
        rows = V.norm(dim=1)
        cv = rows.std() / (rows.mean() + 1e-30)
        vals = torch.stack([ws, wf2 / (ws**2 + 1e-30), vs, vf2 / (vs**2 + 1e-30),
                            vs / (ws + 1e-30), cv]).tolist()
        if v_sv is not None:
            er = effective_rank(v_sv).item()
            vals += [er, er / min_dim]
        else:
            vals += [nan, nan]
        if D is not None:
            der = effective_rank(torch.linalg.svdvals(D)).item() if self.erank else nan
            vals += [der, der / min_dim, (D.norm() / (W.norm() + 1e-30)).item()]
        else:
            vals += [nan, nan, nan]
        return vals

    def _measure_after_b(self) -> dict | None:
        sums, maxs = self._sums, self._maxs
        if sums is None:
            return None
        for i, (name, p, _, _) in enumerate(self.params):
            b = self._buf.get(name)
            if b is None or "w_before" not in b:
                continue
            w = _local(p).detach()
            v_b = w.float() - b.pop("w_before").to(w.device).float()
            v_a = b["v_a"].to(w.device).float()
            sums[i, 8] = v_b.pow(2).sum()
            sums[i, 9] = (v_a * v_b).sum()
        self._buf = {}

        if torch.distributed.is_available() and torch.distributed.is_initialized() and any(
            _is_sharded(p) for _, p, _, _ in self.params
        ):
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(maxs, op=torch.distributed.ReduceOp.MAX)

        S = sums.cpu().tolist()
        M = maxs.cpu().tolist()
        self._sums = self._maxs = None
        step = self.t + 1
        record = self._aggregate(S, M, step)
        self.last_record = record
        is_rank0 = not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
            or torch.distributed.get_rank() == 0
        if is_rank0 or not self.rank0_only:
            if self.out_path:
                with open(self.out_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            if self.log_fn:
                self.log_fn({k: v for k, v in record.items() if v is not None}, step)
        return record

    # -- aggregation ------------------------------------------------------------
    @staticmethod
    def _derive(s: dict, mx: dict) -> dict:
        d1 = _div(s["gb_va"], math.sqrt(s["vv_a"]))
        d2 = _div(s["gb_va"] - s["ga_va"], s["vv_a"])
        return {
            "grad_norm": math.sqrt(s["gg_b"]),
            "grad_max": mx["gmax_b"],
            "grad_cos_prev": _div(s["g_ab"], math.sqrt(s["gg_a"] * s["gg_b"])),
            "w_norm": math.sqrt(s["ww"]),
            "upd_norm": math.sqrt(s["vv_a"]),
            "upd_rms": math.sqrt(_div(s["vv_a"], s["n"])),
            "upd_rel": math.sqrt(_div(s["vv_a"], s["ww"])),
            "upd_max": mx["vmax_a"],
            "upd_cos_prev": _div(s["v_ab"], math.sqrt(s["vv_a"] * s["vv_b"])),
            "upd_grad_align": _div(-s["ga_va"], math.sqrt(s["gg_a"] * s["vv_a"])),
            "D1": d1,
            "dir_sharpness": d2,
            "curvature_K": abs(d2) / (1 + d1 * d1) ** 1.5 if math.isfinite(d1 * d2) else float("nan"),
        }

    def _aggregate(self, S: list, M: list, step: int) -> dict:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, (_, _, layer, typ) in enumerate(self.params):
            if S[i][0] == 0:  # not measured this window
                continue
            groups["global"].append(i)
            groups[f"type/{typ}"].append(i)
            if self.per_layer and layer >= 0:
                groups[f"layer/{layer:03d}/{typ}"].append(i)
        name_of = {i: n for i, (n, _, _, _) in enumerate(self.params)}

        rec: dict = {"step": step, "time": time.time()}
        if self.optimizer is not None:
            for gi, g in enumerate(self.optimizer.param_groups):
                lr = g.get("lr")
                rec[f"lr/group{gi}"] = float(lr) if lr is not None else None
        for gname, idx in groups.items():
            s = {f: sum(S[i][k] for i in idx) for k, f in enumerate(_SUM_FIELDS)}
            mx = {f: max(M[i][k] for i in idx) for k, f in enumerate(_MAX_FIELDS)}
            for k, v in self._derive(s, mx).items():
                rec[f"{gname}/{k}"] = _safe(v)
            spec = [self._spec[name_of[i]] for i in idx if name_of[i] in self._spec]
            if spec:
                for k, f in enumerate(_SPEC_FIELDS):
                    vals = [r[k] for r in spec if math.isfinite(r[k])]
                    if vals:  # mean over the matrices where the metric is available
                        rec[f"{gname}/{f}"] = _safe(sum(vals) / len(vals))
        return rec


# ---------------------------------------------------------------------------
# HF Trainer callback
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback

    class OptimDynamicsCallback(TrainerCallback):
        """Attach with trainer.add_callback(OptimDynamicsCallback(log_every=100, ...)).

        Writes one JSON line per measurement to out_path. To also send metrics to wandb:
            OptimDynamicsCallback(log_every=100, log_fn=make_wandb_log_fn())
        """

        def __init__(self, **logger_kwargs):
            self.logger_kwargs = logger_kwargs
            self.logger: OptimDynamicsLogger | None = None

        def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
            self.logger = OptimDynamicsLogger(model, optimizer=optimizer, **self.logger_kwargs)
            self.logger.t = state.global_step  # resume-safe

        def on_pre_optimizer_step(self, args, state, control, **kw):
            if self.logger is not None:
                self.logger.before_step()

        def on_optimizer_step(self, args, state, control, **kw):
            if self.logger is not None:
                self.logger.after_step()

except ImportError:  # transformers not installed
    pass


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------


def make_wandb_log_fn(
    prefix: str = "dyn",
    per_layer: str = "heatmap",
    heatmap_metrics: Iterable[str] = ("upd_rel", "upd_cos_prev", "grad_cos_prev", "upd_srank"),
    heatmap_every: int = 1,
    step_key: str = "train/global_step",
):
    """Return a log_fn that sends metrics to the active wandb run.

    Scalars are renamed so wandb puts one section per metric with one line per block type:
        global/upd_rel          -> dyn_upd_rel/global
        type/attn_q/upd_rel     -> dyn_upd_rel/attn_q
        lr/group0               -> dyn_lr/group0
    Every key is logged together with `step_key` ("train/global_step" is the x-axis
    HF's WandbCallback defines), so the curves line up with loss and grad_norm.

    per_layer:
        "heatmap"  every `heatmap_every`-th measurement, log a (layer x block type)
                   heatmap image per metric in heatmap_metrics (needs matplotlib);
                   wandb's image panel has a step slider
        "scalars"  log every per-layer value as its own line
                   (dyn_L_upd_rel/attn_q_L003 ...): fine for small models,
                   thousands of keys for big ones
        "none"     per-layer values only go to the JSONL file
    """
    import wandb

    heatmap_metrics = tuple(heatmap_metrics)
    calls = {"n": 0}

    def log_fn(rec: dict, step: int) -> None:
        if wandb.run is None:
            return
        out: dict = {step_key: step}
        layer_vals: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
        for k, v in rec.items():
            parts = k.split("/")
            if parts[0] == "global":
                out[f"{prefix}_{parts[1]}/global"] = v
            elif parts[0] == "type":
                out[f"{prefix}_{parts[2]}/{parts[1]}"] = v
            elif parts[0] == "lr":
                out[f"{prefix}_lr/{parts[1]}"] = v
            elif parts[0] == "layer":
                layer, typ, metric = int(parts[1]), parts[2], parts[3]
                if per_layer == "scalars":
                    out[f"{prefix}_L_{metric}/{typ}_L{layer:03d}"] = v
                elif per_layer == "heatmap" and metric in heatmap_metrics:
                    layer_vals[metric][(layer, typ)] = v
        if per_layer == "heatmap" and calls["n"] % heatmap_every == 0 and layer_vals:
            import matplotlib.pyplot as plt

            for metric, vals in layer_vals.items():
                fig = _heatmap(vals, metric, step)
                out[f"{prefix}_heatmap/{metric}"] = wandb.Image(fig)
                plt.close(fig)  # avoid leaking figures over a long run
        calls["n"] += 1
        wandb.log(out)

    return log_fn


def _heatmap(vals: dict[tuple[int, str], float], metric: str, step: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted({l for l, _ in vals})
    types = sorted({t for _, t in vals})
    grid = np.full((len(layers), len(types)), np.nan)
    for (l, t), v in vals.items():
        grid[layers.index(l), types.index(t)] = v
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(types), 1 + 0.25 * len(layers)))
    cos_like = "cos" in metric or "align" in metric
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r" if cos_like else "viridis",
                   vmin=-1 if cos_like else None, vmax=1 if cos_like else None)
    ax.set_xticks(range(len(types)), types, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)), [str(l) for l in layers])
    ax.set_ylabel("layer")
    ax.set_title(f"{metric} @ step {step}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_layer_heatmap(df, metric: str, block_type: str, ax=None, log: bool = False):
    """Offline: layer x step heatmap of one metric for one block type, from load_dynamics()."""
    import matplotlib.pyplot as plt
    import numpy as np

    sub = df[(df.scope == "layer") & (df.metric == metric) & (df.type == block_type)]
    grid = sub.pivot_table(index="layer", columns="step", values="value")
    data = np.log10(grid.abs().values) if log else grid.values
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                   extent=[grid.columns.min(), grid.columns.max(), -0.5, len(grid.index) - 0.5])
    ax.set_yticks(range(len(grid.index)), [str(int(l)) for l in grid.index])
    ax.set_xlabel("step")
    ax.set_ylabel("layer")
    ax.set_title(f"{'log10 ' if log else ''}{metric} / {block_type}")
    plt.colorbar(im, ax=ax)
    return ax


# ---------------------------------------------------------------------------
# analysis helper
# ---------------------------------------------------------------------------


def load_dynamics(path: str):
    """Load a JSONL log into a long pandas DataFrame:
    columns = step, scope (global|type|layer), layer, type, metric, value."""
    import pandas as pd

    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            step = rec["step"]
            for k, v in rec.items():
                if v is None or k in ("step", "time"):
                    continue
                parts = k.split("/")
                if parts[0] == "global":
                    rows.append((step, "global", None, None, parts[1], v))
                elif parts[0] == "type":
                    rows.append((step, "type", None, parts[1], parts[2], v))
                elif parts[0] == "layer":
                    rows.append((step, "layer", int(parts[1]), parts[2], parts[3], v))
                elif parts[0] == "lr":
                    rows.append((step, "lr", None, parts[1], "lr", v))
    return pd.DataFrame(rows, columns=["step", "scope", "layer", "type", "metric", "value"])
