"""Dump train.py's run plan (batch ramp, steps, LR schedule, checkpoint steps) for several flag sets, CPU only.

    sbatch --partition=dev --gpus-per-node=1 --cpus-per-task=4 --time=0:15:00 --output=logs/%x-%j.out --job-name=plan-dump \
        --wrap="source ./.venv/bin/activate && python tests/plan_dump.py --out plan_new.txt"

Per flag set: the startup lines the plan prints, the derived numbers, and per optimizer step the global batch, tokens
seen and LR factor (the scheduler the Trainer builds: its own WSD / cosine, or train.py's LambdaLR for wsd2 and
--schedule_unit tokens). Two runs diff byte-identical iff the plans agree.
--impl old --src <train.py of an earlier commit> runs that file's plan section (the code before the model is built, plus
the --save_at_tokens block) instead, for checking a refactor against the code it replaced.
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["WORLD_SIZE"] = "8"

import torch  # noqa: E402
from transformers import get_scheduler  # noqa: E402

COMBOS = {
    "default_6b": [],
    "default_3b": ["--token_budget=3e9"],
    "tokens_trunk": ["--token_budget=3e9", "--schedule_unit=tokens", "--batch_ramp=32:0.5,32", "--per_device_batch=2",
                     "--save_at_tokens=500M,1B,1.5B"],
    "tokens_branch": ["--token_budget=3e9", "--schedule_unit=tokens", "--batch_ramp=32:1B,64"],
    "wsd2": ["--token_budget=1e9", "--lr_schedule=wsd2", "--global_batch_size=64"],
    "fixed_save": ["--token_budget=1e9", "--global_batch_size=64", "--save_at_tokens=100M,0.5"],
    "cosine": ["--token_budget=1e9", "--lr_schedule=cosine", "--batch_ramp=none"],
    "max_steps_ramp": ["--max_steps=100", "--batch_ramp=32:0.25,64:0.25,128"],
    "scorer": ["--block_size=2048", "--global_batch_size=256", "--per_device_batch=32", "--token_budget=1e9", "--decay_frac=0"],
}


def old_plan(src_path, flags):
    """The plan of an earlier train.py: exec its code up to the model, and its --save_at_tokens block."""
    src = open(src_path).read()
    head = src[:src.index("# Prepare tokenizer and model")]
    ckpt = src[src.index("CKPT_DIR = "):src.index("# Training arg")]
    sys.argv = ["train.py", "--model_name=x", *flags]
    ns = {"__name__": "plan_dump"}
    exec(compile(head, src_path, "exec"), ns)
    exec(compile(ckpt, src_path, "exec"), ns)
    a = ns["args"]
    decay = ns["DECAY_STEPS"]
    sched = ("cosine_with_min_lr" if a.lr_schedule == "cosine" else "warmup_stable_decay" if decay else "constant_with_warmup",
             {"min_lr_rate": a.min_lr_ratio} if a.lr_schedule == "cosine"
             else {"num_decay_steps": decay, "decay_type": "linear"} if decay else {})
    lam = ns.get("wsd2_lambda") if a.lr_schedule == "wsd2" else ns.get("wsd_tokens_lambda") if a.schedule_unit == "tokens" else None
    return dict(max_steps=ns["MAX_STEPS"], eval_steps=ns["EVAL_STEPS"], per_device_batch=ns["PER_DEVICE_BATCH"],
                grad_accum=ns["GRAD_ACCUM"], ramp=ns["RAMP"], warmup_steps=ns["WARMUP_STEPS"], decay_steps=decay,
                save_steps=ns["SAVE_STEPS"], sched=sched, lam=lam, batch_at=ns["batch_at"], tokens_seen=ns["tokens_seen"])


def new_plan(flags):
    """The same through llama_train, called as train.main() calls it."""
    from llama_train import cli, plan as P
    a = cli.parse_args(["--model_name=x", *flags])
    cli.print_summary(a)
    p = P.make_plan(a, int(os.environ["WORLD_SIZE"]))
    lam = P.lr_lambda(a, p)
    save_steps = P.save_steps(a, p, P.ckpt_dir(a))
    sched_type, sched_kwargs, _ = P.hf_schedule(a, p)
    return dict(max_steps=p.max_steps, eval_steps=p.eval_steps, per_device_batch=p.per_device_batch,
                grad_accum=p.grad_accum, ramp=p.ramp, warmup_steps=p.warmup_steps, decay_steps=p.decay_steps,
                save_steps=save_steps, sched=(sched_type, sched_kwargs), lam=lam, batch_at=p.batch_at, tokens_seen=p.tokens_seen)


ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["old", "new"], default="new")
ap.add_argument("--src", default=None, help="--impl old: the train.py to run")
ap.add_argument("--out", required=True)
cli_args = ap.parse_args()

with open(cli_args.out, "w") as f:
    for name, flags in COMBOS.items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = old_plan(cli_args.src, flags) if cli_args.impl == "old" else new_plan(flags)
        lines = [l for l in buf.getvalue().splitlines() if "Cuda:" not in l]  # torch version line: printed by main, not the plan
        f.write(f"##### {name}: {' '.join(flags)}\n" + "\n".join(lines) + "\n")
        for k in ("max_steps", "eval_steps", "per_device_batch", "grad_accum", "ramp", "warmup_steps", "decay_steps", "save_steps", "sched"):
            f.write(f"{k}: {r[k]}\n")
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        if r["lam"] is not None:
            s = torch.optim.lr_scheduler.LambdaLR(opt, r["lam"])
        else:  # as Trainer.create_scheduler
            s = get_scheduler(r["sched"][0], opt, num_warmup_steps=r["warmup_steps"], num_training_steps=r["max_steps"],
                              scheduler_specific_kwargs=r["sched"][1])
        f.write("step batch tokens_seen lr_factor\n")
        for step in range(r["max_steps"] + 1):
            f.write(f"{step} {r['batch_at'](step + 1)} {r['tokens_seen'](step)} {s.lr_lambdas[0](step)!r}\n")
        print(name, r["max_steps"], "steps", flush=True)
print("wrote", cli_args.out)
