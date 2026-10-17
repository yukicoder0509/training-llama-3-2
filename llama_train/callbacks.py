"""Trainer callbacks: checkpoints at token marks, the Slurm deadline, W&B token axis, step timing / MFU, Sophia's Hessian
pass, torch.profiler."""

import contextlib
import os
import subprocess
import time

import torch
from torch.autograd import DeviceType
from transformers import TrainerCallback

from .optim import sophia_bs

PEAK_FLOPS_PER_GPU = 989e12  # H200 dense bf16 ~989 TFLOPS per GPU


def slurm_time_left():
    """Seconds left in this Slurm job (squeue %L: [D-]HH:MM:SS, MM:SS), or None outside Slurm."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True, timeout=30).stdout.strip()
        days, _, hms = left.rpartition("-")
        parts = [int(x) for x in hms.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception as e:  # UNLIMITED, NOT_SET, squeue unavailable, ...
        print("Could not read the Slurm time limit:", e)
        return None


class SaveAtTokensCallback(TrainerCallback):
    """--save_at_tokens: a full checkpoint (Trainer._save_checkpoint: model, optimizer, scheduler, RNG, trainer state) at
    save_steps, then a tokens-<mark> link to it. save_strategy stays "no"; this sets should_save itself."""

    def __init__(self, save_steps, ckpt_dir, plan):
        self.save_steps, self.ckpt_dir, self.plan = save_steps, ckpt_dir, plan

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        mark = self.save_steps.get(state.global_step)
        if mark and state.is_world_process_zero:
            link = os.path.join(self.ckpt_dir, f"tokens-{mark}")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(f"checkpoint-{state.global_step}", link)
            print(f"=== checkpoint at step {state.global_step} ({self.plan.tokens_seen(state.global_step):,} tokens) -> {link}")
        return control


class DeadlineCallback(TrainerCallback):
    """Runs are bounded by the token budget: max_steps = budget / tokens per step, and training ends after max_steps. The
    job time limit is only a safety net: stop early so the model is still saved before Slurm kills the job (slow node, or
    a --time too short for the budget). Such a run saw fewer tokens and its LR decay was cut short, so it is not
    comparable to others. It warns as early as step 50 if the projected finish is past the deadline, and train.py
    records budget_completed in the W&B summary."""
    PROJECT_AT = 50  # compile and warm-up are done by then

    def __init__(self, deadline, start_time):
        self.deadline, self.start_time = deadline, start_time
        self.t_project = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.PROJECT_AT // 2:
            self.t_project = time.time()
        elif state.global_step == self.PROJECT_AT and self.t_project and state.is_world_process_zero:
            per_step = (time.time() - self.t_project) / (self.PROJECT_AT - self.PROJECT_AT // 2)  # includes evals in that window
            finish = time.time() + per_step * (state.max_steps - state.global_step)
            if finish > self.deadline:
                print(f"=== WARNING: projected finish {(finish - self.start_time) / 60:.0f} min after start is past the deadline "
                      f"({(self.deadline - self.start_time) / 60:.0f} min): the run will stop before the token budget. Raise --time.")
            else:
                print(f"=== Projected finish {(finish - self.start_time) / 60:.0f} min after start (deadline {(self.deadline - self.start_time) / 60:.0f} min), plus evals")
        stop = torch.tensor(float(time.time() > self.deadline), device=args.device)
        if torch.distributed.is_initialized():  # all ranks must stop at the same step or the next collective hangs
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            print(f"=== WARNING: deadline reached at step {state.global_step}/{state.max_steps}, stopping to save the model. "
                  f"The token budget was NOT completed; this run is not comparable to full-budget runs.")
            control.should_training_stop = True
        return control


# W&B x-axis: tokens seen instead of the Trainer's global step, for train and eval charts alike. Eval rows carry the
# same key: CustomTrainer.log adds total_tokens_seen to eval logs, and WandbCallback prefixes it "train/" (no eval_ prefix).
# Runs after WandbCallback.on_train_begin (user callbacks come after the reporting ones), i.e. after wandb.init.
TOKEN_AXIS_METRICS = ["train/loss", "train/ppl", "train/grad_norm", "train/learning_rate", "train/tokens_per_second",
                      "train/mfu", "train/sophia_win_rate", "eval/loss", "eval/perplexity", "eval/dev_loss", "eval/dev_perplexity"]


class WandbTokenAxisCallback(TrainerCallback):
    def __init__(self, cli_args, plan):
        self.cli_args, self.plan = cli_args, plan

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        p = self.plan
        if state.is_world_process_zero and wandb.run is not None:
            wandb.define_metric("train/total_tokens_seen")
            # WandbCallback logs TrainingArguments and the model config; add every train.py flag (W&B config "args.*")
            # and the derived run shape ("derived.*"), so runs can be filtered / compared by them
            wandb.config.update({"args": vars(self.cli_args), "derived": {
                "max_steps": p.max_steps, "warmup_steps": p.warmup_steps, "decay_steps": p.decay_steps, "eval_steps": p.eval_steps,
                "tokens": p.tokens_seen(p.max_steps), "tokens_per_step": None if p.ramp else p.batch_at(1) * p.block_size,
                "batch_ramp_steps": p.ramp, "world_size": p.world_size, "per_device_batch": p.per_device_batch, "grad_accum": p.grad_accum,
                "block_size": p.block_size, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")}},
                allow_val_change=True)
            for key in TOKEN_AXIS_METRICS:  # explicit names take precedence over WandbCallback's "*" -> global_step
                wandb.define_metric(key, step_metric="train/total_tokens_seen")


class StepTimerCallback(TrainerCallback):
    """Times each optimizer step (on_step_begin -> on_step_end excludes eval) and prints the median at the end."""

    def __init__(self, plan, flops_per_token, start_time):
        self.plan, self.flops_per_token, self.start_time = plan, flops_per_token, start_time
        self.times, self.steps, self.t0 = [], [], None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.times.append(time.time() - self.t0)
        self.steps.append(state.global_step)
        if state.global_step == 1 and state.is_world_process_zero:  # includes torch.compile time if enabled
            print(f"=== First step done at {time.time() - self.start_time:.0f}s ({self.times[0]:.1f}s for step 1)")

    def last_step_throughput(self):
        """Training tokens/s (all GPUs) and MFU of the latest optimizer step (the lab logs it "on that step")."""
        tokens_per_sec = self.plan.block_size * self.plan.batch_at(self.steps[-1]) / self.times[-1]
        return tokens_per_sec, tokens_per_sec * self.flops_per_token / (PEAK_FLOPS_PER_GPU * self.plan.world_size)

    def on_train_end(self, args, state, control, **kwargs):
        gib = 1024 ** 3
        total = torch.cuda.get_device_properties(args.device).total_memory
        print(f"=== Peak GPU memory (rank {args.process_index}): allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB of {total / gib:.0f} GiB")
        if state.is_world_process_zero and len(self.times) > 10:
            t = sorted(self.times[5:])  # skip warm-up steps
            med = t[len(t) // 2]
            mean = sum(t) / len(t)  # differs from the median when some steps do extra work (Sophia's Hessian pass)
            tokens = self.plan.block_size * self.plan.batch_at(state.global_step)
            print(f"=== Step time: median {med:.3f} s, mean {mean:.3f} s over {len(t)} steps (excl. eval), "
                  f"{tokens / med:,.0f} tokens/s at the median, "
                  f"{tokens / mean:,.0f} at the mean")


class SophiaHessianCallback(TrainerCallback):
    """Sophia's Hessian estimate, as in the official train_sophiag.py: every k steps, after the optimizer step, forward +
    backward the step's batch with labels sampled from the model, clip, and EMA the squared gradient into the Hessian.
    Reuses the step's own micro-batches (official: the next batches) so no extra data tokens are consumed. Gradients
    are averaged over all GPUs through DDP, so every rank keeps the same Hessian. Also logs the "win rate" (fraction of
    coordinates whose update is not clipped, official train/win_rate; should stay ~0.1-0.5, else tune rho) into `stats`.
    Set .trainer after the Trainer is built."""

    def __init__(self, cli_args, plan, model, optimizer, sophia_opt, stats):
        self.cli_args, self.plan, self.model, self.optimizer, self.sophia_opt, self.stats = cli_args, plan, model, optimizer, sophia_opt, stats
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        cli, model, sophia_opt, trainer = self.cli_args, self.model, self.sophia_opt, self.trainer
        if state.global_step % cli.sophia_hess_interval:
            return
        ddp_model, batches = trainer.model_wrapped, trainer.step_batches
        for i, inputs in enumerate(batches):
            sync = contextlib.nullcontext() if i == len(batches) - 1 or ddp_model is model else ddp_model.no_sync()
            with sync, trainer.compute_loss_context_manager():
                loss = ddp_model(**{k: v for k, v in inputs.items() if k != "labels"}, sample_labels=True).loss / len(batches)
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sophia_opt.update_hessian()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            unclipped = sum((st["exp_avg"].abs() < cli.sophia_rho * sophia_bs(self.plan, state.global_step) * st["hessian"]).sum()
                            for st in sophia_opt.state.values())
            self.stats["sophia_win_rate"] = (unclipped / sum(st["exp_avg"].numel() for st in sophia_opt.state.values())).item()


class ProfilerCallback(TrainerCallback):
    """torch.profiler over 5 steady-state steps: Chrome trace per rank, plus (rank 0) top ops by GPU and CPU time and the
    GPU busy fraction (sum of GPU-side events: kernels, memcpy, memset / wall time; NCCL overlap can push it slightly
    above 100%). Same sum as the table footer: CPU ops (aten::mm) and GPU-side user annotations (ProfilerStep*,
    DistributedDataParallel.forward) are excluded, since their self device time repeats the time of the kernels they
    launch or enclose. The GPU is synchronized only at the two window edges (to time the window), so the CPU can queue
    work ahead as in an unprofiled run; a sync after every step exposed the batch-loading gap as ~9% fake idle."""
    WARMUP, ACTIVE = 3, 5

    def __init__(self, out_dir, start, cpu):
        self.out_dir, self.t_active, self.t_end, self.cpu = out_dir, None, None, cpu
        self.first, self.last = start, start + self.ACTIVE  # timed window: steps first+1 .. last
        os.makedirs(out_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CUDA] + ([torch.profiler.ProfilerActivity.CPU] if cpu else [])
        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=start - self.WARMUP, warmup=self.WARMUP, active=self.ACTIVE, repeat=1),
            on_trace_ready=self.report,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (self.first, self.last):
            torch.cuda.synchronize()
        if state.global_step == self.first:
            self.t_active = time.time()  # active window starts with the next step
        if state.global_step == self.last:
            self.t_end = time.time()  # before prof.step(), which collects the trace (~0.25 s) and calls report
        self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()

    def report(self, prof):
        wall = self.t_end - self.t_active
        rank = int(os.environ.get("RANK", 0))
        prof.export_chrome_trace(os.path.join(self.out_dir, f"trace_rank{rank}.json"))
        # Peak since process start (covers optimizer state init and the logits/loss spike), vs. GPU capacity
        gib, total = 2**30, torch.cuda.get_device_properties(0).total_memory
        print(f"=== Memory rank {rank}: peak allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
              f"peak reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB, of {total / gib:.1f} GiB "
              f"({torch.cuda.max_memory_reserved() / total:.0%})")
        if rank != 0:
            return
        events = prof.key_averages()
        gpu_us = sum(e.self_device_time_total for e in events if e.device_type == DeviceType.CUDA and not e.is_user_annotation)
        print(f"=== Profile: steps {self.first + 1}-{self.last}, wall {wall:.3f} s ({wall / self.ACTIVE:.3f} s/step), "
              f"GPU kernel time {gpu_us / 1e6:.3f} s -> GPU busy {gpu_us / 1e6 / wall:.0%}, CPUs available {len(os.sched_getaffinity(0))}")
        for key in ("self_device_time_total",) + (("self_cpu_time_total",) if self.cpu else ()):
            print(f"=== Top ops by {key}")
            print(events.table(sort_by=key, row_limit=25, max_name_column_width=60))
