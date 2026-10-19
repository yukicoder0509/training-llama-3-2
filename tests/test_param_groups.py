"""Stage 0 T1 / T2: muonium param groups for --muon_split_heads, --muon_lr_mult_kv, --norm_lr, --embed_optimizer soap.

    sbatch --account=ACD115198 --partition=dev --gpus-per-node=1 --cpus-per-task=4 --time=0:30:00 \
        --output=logs/%x-%j.out --job-name=test-param-groups --wrap="source ./.venv/bin/activate && python tests/test_param_groups.py"

On the 174M proxy (train_scorer.py config): prints each flag set's group table, checks groups / LRs / split sizes, takes
one optimizer step on random gradients, and checks that a head split orthogonalizes each head's rows separately (plain
Muon, no weight decay: each 64-row block of the k_proj update must match the polar factor of that head's gradient).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_train import cli  # noqa: E402
from llama_train.model import build_model, llama_config  # noqa: E402
from llama_train.optim import build_optimizer  # noqa: E402
from train_scorer import DEFAULTS  # noqa: E402

BASE = [*DEFAULTS, "--muonium_algorithm=normuon", "--learning_rate=3e-3"]  # = proxy-dyn-normuon-1b-bs128 (566231)
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def build(extra, seed=0):
    args = cli.parse_args([*BASE, *extra])
    config = llama_config(args)
    torch.manual_seed(seed)
    model = build_model(args, config).cuda()
    opt, _ = build_optimizer(args, model, config, None, None)
    return args, model, opt


def step(model, opt, grads=None, override=None):
    """One step on fixed random gradients (or override {name: grad}); returns {name: before - after} (and fills grads
    {name: grad} if given)."""
    g = torch.Generator(device="cuda").manual_seed(1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        p.grad = torch.randn(p.shape, generator=g, device="cuda", dtype=p.dtype)
        if override and n in override:
            p.grad = override[n].to(p.dtype)
        if grads is not None:
            grads[n] = p.grad.detach().clone()
    opt.step()
    return {n: before[n] - p.detach() for n, p in model.named_parameters()}


def groups(args):
    return args.param_groups


print("== default (baseline 566231: lr/group0 0.01, group1 = group2 0.003)")
args, model, opt = build([])
gs = groups(args)
check(len(gs) == 3, f"3 groups, got {len(gs)}")
check(gs[0]["algorithm"] == "normuon" and gs[0]["lr"] == 0.01 and gs[0]["tensors"] == 84 and gs[0]["split_sizes"] is None,
      "group0 normuon lr 0.01, 84 matrices, no split")
check(gs[1]["algorithm"] == "adamw" and gs[1]["lr"] == 3e-3 and gs[1]["examples"] == "embed_tokens", "group1 adamw 3e-3 embedding")
check(gs[2]["algorithm"] == "adamw" and gs[2]["lr"] == 3e-3 and gs[2]["wd"] == 0.1 and gs[2]["tensors"] == 25,
      "group2 adamw 3e-3 norms (25), wd 0.1")
step(model, opt)
del model, opt

print("== --muon_split_heads kv")
args, model, opt = build(["--muon_split_heads=kv"])
gs = groups(args)
check(len(gs) == 4 and gs[0]["tensors"] == 60, f"4 groups, group0 60 matrices (got {len(gs)}, {gs[0]['tensors']})")
check(gs[1]["examples"] == "k_proj,v_proj" and gs[1]["tensors"] == 24 and tuple(gs[1]["split_sizes"]) == (64,) * 4
      and gs[1]["lr"] == 0.01, "group1 k_proj + v_proj (24), split (64,)x4, lr 0.01")
step(model, opt)
del model, opt

print("== --muon_split_heads qkv")
args, model, opt = build(["--muon_split_heads=qkv"])
gs = groups(args)
check(len(gs) == 5 and gs[0]["tensors"] == 48, f"5 groups, group0 48 matrices (got {len(gs)}, {gs[0]['tensors']})")
check(gs[1]["examples"] == "q_proj" and tuple(gs[1]["split_sizes"]) == (64,) * 12, "group1 q_proj, split (64,)x12")
check(gs[2]["examples"] == "k_proj,v_proj" and tuple(gs[2]["split_sizes"]) == (64,) * 4, "group2 k_proj + v_proj, split (64,)x4")
step(model, opt)
del model, opt

print("== --muon_lr_mult_kv 0.5")
args, model, opt = build(["--muon_lr_mult_kv=0.5"])
gs = groups(args)
check(len(gs) == 4 and gs[1]["examples"] == "k_proj,v_proj" and gs[1]["lr"] == 0.005 and gs[1]["split_sizes"] is None,
      "group1 k_proj + v_proj at 0.005, no split")
step(model, opt)
del model, opt

print("== --norm_lr 1e-3")
args, model, opt = build(["--norm_lr=1e-3"])
gs = groups(args)
check(len(gs) == 3 and gs[1]["lr"] == 3e-3 and gs[2]["lr"] == 1e-3 and gs[2]["tensors"] == 25, "embedding 3e-3, norms 1e-3")
step(model, opt)
del model, opt

print("== --embed_optimizer soap --learning_rate 2.8e-3 --norm_lr 3e-3")
args, model, opt = build(["--embed_optimizer=soap", "--learning_rate=2.8e-3", "--norm_lr=3e-3"])
gs = groups(args)
check(len(gs) == 3 and gs[0]["algorithm"] == "normuon" and gs[1]["examples"] == "input_layernorm,norm,post_attention_layernorm"
      and gs[1]["lr"] == 3e-3, "group0 normuon, group1 norms adamw 3e-3")
check(gs[2]["algorithm"] == "soap" and gs[2]["lr"] == 2.8e-3 and gs[2]["wd"] == 0.1 and gs[2]["examples"] == "embed_tokens",
      "group2 SOAP 2.8e-3 embedding, wd 0.1")
for _ in range(3):  # SOAP skips its first step (preconditioner init); the third moves the embedding
    upd = step(model, opt)
check(upd["model.embed_tokens.weight"].abs().max().item() > 0, "embedding updated by SOAP")
sd = opt.state_dict()
check(set(sd) == {"0", "1"} and len(sd["1"]["state"]) == 1, "state_dict nests muonium and SOAP (1 SOAP tensor)")
# accelerate's prepare() round-trips the optimizer through load_state_dict (device placement). The LR scheduler then
# writes optimizer.param_groups: those must still be the dicts the inner optimizers read.
from accelerate import Accelerator  # noqa: E402
from accelerate.optimizer import AcceleratedOptimizer  # noqa: E402
Accelerator()  # AcceleratedOptimizer needs the accelerate state
wrapped = AcceleratedOptimizer(opt)
sched = torch.optim.lr_scheduler.LambdaLR(wrapped, lambda s: 0.5)
inner = [g for o in opt.opts for g in o.param_groups]
check(all(a is b for a, b in zip(wrapped.param_groups, inner)) and len(wrapped.param_groups) == len(inner),
      "after accelerate's load_state_dict, the chained param_groups are the inner optimizers' dicts")
check(abs(opt.opts[1].param_groups[0]["lr"] - 1.4e-3) < 1e-12 and abs(opt.opts[0].param_groups[0]["lr"] - 5e-3) < 1e-12,
      "a scheduler factor 0.5 reaches SOAP (1.4e-3) and muonium (5e-3)")
del model, opt

print("== head split orthogonalizes per head (plain Muon, no weight decay; k_proj of layer 0)")
# First step: the update is the orthogonalized (Polar Express) Nesterov momentum, a multiple of the gradient. With the
# split, each head's 64 rows must be the polar factor U V^T of that head's own gradient block; without it, not (the
# whole 256-row matrix is orthogonalized jointly). The heads get correlated gradients (a shared 64 x 768 component plus
# noise): joint orthogonalization then decorrelates the heads, per-head does not. (With i.i.d. random gradients the rows
# are already near-orthogonal and both agree to ~0.96.) Cosine of each head block with the exact polar factor:
def polar(m):
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    return u @ vh


cos = {}
for split in ("none", "kv"):
    args, model, opt = build(["--muonium_algorithm=muon", "--muon_wd=0", f"--muon_split_heads={split}"])
    grads = {}
    name = "model.layers.0.self_attn.k_proj.weight"
    gen = torch.Generator(device="cuda").manual_seed(2)
    shared = torch.randn(64, 768, generator=gen, device="cuda")
    corr = torch.cat([shared + 0.5 * torch.randn(64, 768, generator=gen, device="cuda") for _ in range(4)])
    upd = step(model, opt, grads, override={name: corr})[name].float()
    blocks = zip(upd.split(64, dim=0), grads[name].float().split(64, dim=0))
    cos[split] = [torch.nn.functional.cosine_similarity(u.flatten(), polar(g).flatten(), dim=0).item() for u, g in blocks]
    print(f"  {split}: cos(update head block, polar(grad head block)) = " + ", ".join(f"{c:.3f}" for c in cos[split]))
    del model, opt
check(min(cos["kv"]) > 0.95 and max(cos["none"]) < min(cos["kv"]) - 0.05,
      "split: each head's update is its own gradient block's polar factor; unsplit is not")

print("FAILED: " + "; ".join(failures) if failures else "ALL OK")
sys.exit(1 if failures else 0)
