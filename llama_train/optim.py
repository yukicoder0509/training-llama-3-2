"""Optimizers for --optimizer: adamw (built by the Trainer, or here with --norm_wd), adam_mini, muon, muonium, sophia, soap."""

import math
import os

import torch

from adam_mini import Adam_mini
from muon_adamw import build_muon_adamw
from sophia import SophiaG
from soap_dist import DistributedSOAP


def uses_sophia(args):
    return args.optimizer == "sophia" or (args.optimizer == "muonium" and args.muonium_rest == "sophia")


def norm_wd(args):
    """Weight decay of the 1-D params (RMSNorm weights): --weight_decay with --norm_wd, else 0."""
    return args.weight_decay if args.norm_wd else 0.0


def sophia_bs(plan, step):  # official: step(bs=total_bs * block_size), tokens per step; follows --batch_ramp
    return plan.batch_at(step) * plan.block_size


class ChainedOptimizer(torch.optim.Optimizer):
    """Several optimizers as one for the HF Trainer: exposes all their param groups (the same dicts, so the one LR
    scheduler scales every group by the same factor, each from its own peak LR) and steps them all."""

    def __init__(self, *opts):
        self.opts = opts
        super().__init__([p for o in opts for g in o.param_groups for p in g["params"]], {})
        self.param_groups = [g for o in opts for g in o.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.opts:
            o.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none)

    def state_dict(self):
        return {str(i): o.state_dict() for i, o in enumerate(self.opts)}

    def load_state_dict(self, state_dict):
        for i, o in enumerate(self.opts):
            o.load_state_dict(state_dict[str(i)])
        # torch's load_state_dict gives each optimizer new param-group dicts: relink, or the LR scheduler (and the
        # dynamics logger) would keep writing the old ones. accelerate calls this at prepare() (device placement), so
        # before this fix every chained run trained at a constant LR (no warmup, no decay) while logging the schedule.
        self.param_groups = [g for o in self.opts for g in o.param_groups]


def muon_head_groups(args, config, muon_named):
    """muonium groups for the hidden matrices. Default: one group at --muon_lr. --muon_split_heads kv / qkv: k_proj and
    v_proj (and q_proj) in their own group, orthogonalized per attention head (muonium split_sizes: row blocks of
    head_dim). --muon_lr_mult_kv: k_proj / v_proj in their own group at muon_lr x mult."""
    alg = args.muonium_algorithm
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_own = args.muon_split_heads in ("kv", "qkv") or args.muon_lr_mult_kv != 1
    is_q = lambda n: args.muon_split_heads == "qkv" and "q_proj" in n
    is_kv = lambda n: kv_own and ("k_proj" in n or "v_proj" in n)

    def head_split(ps):
        rows = {p.shape[0] for p in ps}
        assert len(rows) == 1 and next(iter(rows)) % head_dim == 0, f"head split: rows {rows}, head_dim {head_dim}"
        return (head_dim,) * (next(iter(rows)) // head_dim)

    groups = [{"params": [p for n, p in muon_named if not is_q(n) and not is_kv(n)], "algorithm": alg}]
    q_p = [p for n, p in muon_named if is_q(n)]
    if q_p:
        groups.append({"params": q_p, "algorithm": alg, "split_sizes": head_split(q_p)})
    kv_p = [p for n, p in muon_named if is_kv(n)]
    if kv_p:
        kv = {"params": kv_p, "algorithm": alg}
        if args.muon_split_heads != "none":
            kv["split_sizes"] = head_split(kv_p)
        if args.muon_lr_mult_kv != 1:
            kv["lr"] = args.muon_lr * args.muon_lr_mult_kv
        groups.append(kv)
    return groups


def param_group_table(optimizer, named):
    """One row per param group (index as in the dynamics logger's lr/group<i>): algorithm, lr, wd, split_sizes, sizes and
    example names. Printed at startup."""
    names = {id(p): n for n, p in named}
    rows = []
    for i, g in enumerate(optimizer.param_groups):
        ps = g["params"]
        ex = sorted({names[id(p)].split(".")[-2] if names[id(p)].count(".") >= 2 else names[id(p)] for p in ps})
        rows.append({"group": i, "algorithm": g.get("algorithm", "soap" if "precondition_frequency" in g else "?"),
                     "lr": g["lr"], "wd": g.get("wd", g.get("weight_decay")), "split_sizes": g.get("split_sizes"),
                     "tensors": len(ps), "params": sum(p.numel() for p in ps), "examples": ",".join(ex[:4])})
    print("Param groups:\n  group | algorithm | lr | wd | split_sizes | tensors | params | examples")
    for r in rows:
        split = f"({r['split_sizes'][0]},)x{len(r['split_sizes'])}" if r["split_sizes"] else "-"
        print(f"  {r['group']} | {r['algorithm']} | {r['lr']:g} | {r['wd']:g} | {split} | {r['tensors']} | "
              f"{r['params'] / 1e6:.2f}M | {r['examples']}")
    return rows


def build_sophia(args, plan, params):
    class SophiaGTrainer(SophiaG):
        """The HF Trainer calls optimizer.step() without SophiaG's bs argument."""
        t = 0  # optimizer steps taken (1-based step of this call)

        def step(self, closure=None, bs=None):
            self.t += 1
            return super().step(closure, bs=sophia_bs(plan, self.t))

    # Same decay split as the HF Trainer's AdamW (matrices + embedding; the RMSNorm weights only with --norm_wd)
    opt = SophiaGTrainer(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": norm_wd(args)}],
        lr=args.learning_rate, betas=tuple(float(b) for b in args.sophia_betas.split(",")), rho=args.sophia_rho,
        weight_decay=args.weight_decay,
    )
    print(f"SophiaG: lr {args.learning_rate}, rho {args.sophia_rho}, betas {args.sophia_betas}, wd {args.weight_decay} (norms {norm_wd(args)}), "
          f"bs {sophia_bs(plan, 1):,}{f' -> {sophia_bs(plan, plan.max_steps):,}' if plan.ramp else ''} tokens, Hessian every {args.sophia_hess_interval} steps")
    return opt


def build_optimizer(args, model, config, plan, training_args):
    """(optimizer, sophia_opt). optimizer None: the Trainer builds AdamW from training_args. sophia_opt: the SophiaG
    instance the Hessian callback updates (--optimizer sophia, or muonium with --muonium_rest sophia), else None."""
    optimizer, sophia_opt = None, None
    if uses_sophia(args):
        assert args.fused_ce, "SophiaG samples labels inside the fused CE forward; use --fused_ce"

    # Adam-mini recognizes the HF Llama names itself: embed_tokens (tied lm_head) one lr per row,
    # q_proj/k_proj one lr per head, the rest one per output neuron / tensor.
    if args.optimizer == "adam_mini":
        optimizer = Adam_mini(
            named_parameters=model.named_parameters(),
            lr=args.learning_rate,
            betas=(0.9, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,  # Adam-mini skips decay on norm and bias params itself
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            verbose=int(os.environ.get("LOCAL_RANK", 0)) == 0,
        )
    elif args.optimizer == "muon":
        optimizer, muon_names, adamw_names = build_muon_adamw(
            model, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum,
            adamw_lr=args.learning_rate, beta2=args.beta2, weight_decay=args.weight_decay,
        )
        n = dict(model.named_parameters())
        print(f"Muon: {len(muon_names)} tensors, {sum(n[k].numel() for k in muon_names) / 1e6:.1f}M params, lr {args.muon_lr}; "
              f"AdamW: {len(adamw_names)} tensors, {sum(n[k].numel() for k in adamw_names) / 1e6:.1f}M params, lr {args.learning_rate}")

    elif args.optimizer == "muonium":
        # pip muonium (github.com/tonyf/muonium): Muon for 2D weights (momentum 0.95 Nesterov, Polar Express
        # orthogonalization, LR x sqrt(max(1, rows/cols)), cautious decoupled wd) or NorMuon (--muonium_algorithm), and for
        # the rest (as in its README: matrices except the tied embedding -> Muon) its fused AdamW (betas 0.9/0.95, eps 1e-8)
        # or SophiaG (--muonium_rest). Muon LR --muon_lr, wd --muon_wd (default --weight_decay); the rest --learning_rate,
        # --weight_decay. Package defaults: one lr 1e-3, wd 0.1 (the first run, muonium-default-1b, used --learning_rate for both).
        from muonium import Muon
        named = list(model.named_parameters())
        muon_p = [p for n, p in named if p.ndim >= 2 and "embed" not in n]
        rest_p = [p for n, p in named if p.ndim < 2 or "embed" in n]
        muon_wd = args.weight_decay if args.muon_wd is None else args.muon_wd
        muon_groups = muon_head_groups(args, config, [(n, p) for n, p in named if p.ndim >= 2 and "embed" not in n])
        if args.muonium_rest == "adamw":
            adamw = dict(algorithm="adamw", lr=args.learning_rate)
            embed_p = [p for p in rest_p if p.ndim >= 2]
            norms = {"params": [p for p in rest_p if p.ndim < 2], "wd": norm_wd(args), **adamw}
            if args.norm_lr is not None:
                norms["lr"] = args.norm_lr
            if args.embed_optimizer == "soap":  # the tied embedding on SOAP (--learning_rate), the rest in muonium
                optimizer = ChainedOptimizer(Muon([*muon_groups, norms], lr=args.muon_lr, wd=muon_wd), DistributedSOAP(
                    [{"params": embed_p, "weight_decay": args.weight_decay}], lr=args.learning_rate,
                    betas=tuple(float(b) for b in args.soap_betas.split(",")), eps=args.adam_eps, weight_decay=args.weight_decay,
                    precondition_frequency=args.soap_precond_freq, max_precond_dim=args.soap_max_precond_dim))
            else:
                optimizer = Muon([*muon_groups, {"params": embed_p, "wd": args.weight_decay, **adamw}, norms],
                                 lr=args.muon_lr, wd=muon_wd)
        else:
            assert args.norm_lr is None and args.embed_optimizer == "adamw", "--norm_lr / --embed_optimizer need --muonium_rest adamw"
            sophia_opt = build_sophia(args, plan, rest_p)
            optimizer = ChainedOptimizer(Muon(muon_groups, lr=args.muon_lr, wd=muon_wd), sophia_opt)
        print(f"muonium {args.muonium_algorithm}: {len(muon_p)} tensors, {sum(p.numel() for p in muon_p) / 1e6:.1f}M params, "
              f"lr {args.muon_lr}, wd {muon_wd}; {args.muonium_rest}: {len(rest_p)} tensors, "
              f"{sum(p.numel() for p in rest_p) / 1e6:.1f}M params, lr {args.learning_rate}, wd {args.weight_decay} (norms {norm_wd(args)})")
        args.param_groups = param_group_table(optimizer, named)  # in the W&B config with the other args

    elif args.optimizer == "adamw" and args.norm_wd:
        # The Trainer's adamw_torch_fused, but decaying the RMSNorm weights too (the model has no biases)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay, fused=True,
        )
        print(f"AdamW (fused): lr {training_args.learning_rate}, wd {training_args.weight_decay} on all params incl. norms")

    elif args.optimizer == "sophia":
        sophia_opt = optimizer = build_sophia(args, plan, [p for p in model.parameters() if p.requires_grad])

    elif args.optimizer == "soap" and args.soap_impl == "meta":
        import distributed_shampoo as ds
        # DistributedShampoo allocates and shards its state at construction: the model must already be on its GPU
        # (training_args.device also initializes the process group).
        model.to(training_args.device)
        params = [p for p in model.parameters() if p.requires_grad]
        betas = tuple(float(b) for b in args.soap_betas.split(","))
        optimizer = ds.DistributedShampoo(
            [  # matrices + embedding: SOAP (eigenvalue-corrected Shampoo, QR eigenbasis updates as in the SOAP paper)
             {"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
             # RMSNorm weights: plain Adam, as official SOAP (precondition_1d=False), weight decay only with --norm_wd
             {"params": [p for p in params if p.ndim < 2], "weight_decay": norm_wd(args), "start_preconditioning_step": math.inf,
              "grafting_config": ds.AdamPreconditionerConfig(beta2=betas[1], epsilon=args.adam_eps)}],
            lr=args.learning_rate, betas=betas, epsilon=args.adam_eps, weight_decay=args.weight_decay,
            weight_decay_type=ds.WeightDecayType.DECOUPLED, max_preconditioner_dim=args.soap_block,
            precondition_frequency=args.soap_precond_freq, use_bias_correction=True,
            preconditioner_config=ds.DefaultSOAPConfig,
            distributed_config=ds.DDPDistributedConfig(communication_dtype=getattr(torch, args.soap_comm_dtype),
                                                       num_trainers_per_group=-1, communicate_params=False),
        )
        print(f"SOAP (meta distributed_shampoo): lr {args.learning_rate}, betas {args.soap_betas}, eps {args.adam_eps}, "
              f"wd {args.weight_decay}, precondition every {args.soap_precond_freq} steps, block {args.soap_block}, "
              f"AllGather in {args.soap_comm_dtype}")
    elif args.optimizer == "soap":
        # Same decay split as the HF Trainer's AdamW (matrices + embedding; the RMSNorm weights only with --norm_wd)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = DistributedSOAP(
            [{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
             {"params": [p for p in params if p.ndim < 2], "weight_decay": norm_wd(args)}],
            lr=args.learning_rate, betas=tuple(float(b) for b in args.soap_betas.split(",")), eps=args.adam_eps,
            weight_decay=args.weight_decay, precondition_frequency=args.soap_precond_freq,
            max_precond_dim=args.soap_max_precond_dim, precondition_1d=args.soap_precondition_1d,
        )
        print(f"SOAP: lr {args.learning_rate}, betas {args.soap_betas}, eps {args.adam_eps}, wd {args.weight_decay} (norms {norm_wd(args)}), "
              f"precondition every {args.soap_precond_freq} steps, max dim {args.soap_max_precond_dim}, precondition 1-D {args.soap_precondition_1d}")
    return optimizer, sophia_opt
