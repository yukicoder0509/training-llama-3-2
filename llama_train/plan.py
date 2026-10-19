"""The run's shape from the flags: steps, batch-size ramp, tokens per step, LR schedule, checkpoint steps.

No GPU or model needed (tests/plan_dump.py runs it on the CPU).
"""

import os

import torch


def stage_tokens(t, token_budget):  # "1e9", "1B", "300M" -> tokens; <= 1 -> fraction of the budget
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(t[-1].upper(), 1)
    x = float(t[:-1] if mult != 1 else t) * mult
    return x * token_budget if x <= 1 else x


class Plan:
    """Steps and batch sizes of a run. ramp: list of (global batch, optimizer steps), or None for a fixed batch."""

    def __init__(self, block_size, global_batch_size, world_size, per_device_batch, grad_accum, max_steps, ramp, eval_steps,
                 warmup_steps, decay_steps):
        self.block_size, self.global_batch_size, self.world_size = block_size, global_batch_size, world_size
        self.per_device_batch, self.grad_accum, self.max_steps, self.ramp = per_device_batch, grad_accum, max_steps, ramp
        self.eval_steps, self.warmup_steps, self.decay_steps = eval_steps, warmup_steps, decay_steps

    def batch_at(self, step):
        """Global batch size of optimizer step `step` (1-based)."""
        if self.ramp is None:
            return self.global_batch_size
        for b, n in self.ramp:
            if step <= n:
                return b
            step -= n
        return self.ramp[-1][0]

    def tokens_seen(self, step):
        """Training tokens processed up to and including optimizer step `step`."""
        if self.ramp is None:
            return step * self.global_batch_size * self.block_size
        total = 0
        for b, n in self.ramp:
            total += min(step, n) * b * self.block_size
            step -= min(step, n)
        return total + step * self.ramp[-1][0] * self.block_size


def make_plan(args, world_size):
    block_size = args.block_size
    per_device_batch = min(args.per_device_batch, args.global_batch_size // world_size)
    assert args.global_batch_size % (per_device_batch * world_size) == 0, "global batch must be a multiple of per-GPU batch x GPUs"
    grad_accum = args.global_batch_size // (per_device_batch * world_size)
    # Token budget -> steps. Every position of every sequence is a real token (packed documents, no padding).
    max_steps = args.max_steps or int(args.token_budget // (args.global_batch_size * block_size))
    # Batch-size ramp: list of (global batch, optimizer steps). Each stage gets its tokens (a count, or a fraction of the
    # budget), rounded down to whole steps, so the batch changes at the first step boundary at or before the stage's token
    # mark. The LR schedule (warmup / WSD decay) runs over the total optimizer steps, unchanged by the ramp.
    ramp = None
    if args.batch_ramp:
        stages = [part.split(":") for part in args.batch_ramp.split(",")]
        sizes = [int(st[0]) for st in stages]
        stage_toks = [stage_tokens(st[1], args.token_budget) for st in stages[:-1]]
        stage_toks.append(args.token_budget - sum(stage_toks))
        assert stage_toks[-1] > 0, f"--batch_ramp stages use {sum(stage_toks[:-1]):.3g} tokens, budget {args.token_budget:.3g}"
        # The Trainer's grad accum is fixed: enough micro-batches for the largest batch at --per_device_batch; smaller
        # batches use smaller micro-batches (one static compile per size)
        grad_accum = -(-max(sizes) // (world_size * args.per_device_batch))
        assert all(b % (world_size * grad_accum) == 0 for b in sizes), args.batch_ramp
        ramp = [(b, int(t // (b * block_size))) for b, t in zip(sizes, stage_toks)]
        if args.max_steps:  # short tests: keep the stage proportions in steps
            total = sum(n for _, n in ramp)
            ramp = [(b, max(1, round(n * args.max_steps / total))) for b, n in ramp]
        max_steps = sum(n for _, n in ramp)
        per_device_batch = sizes[0] // (world_size * grad_accum)
        print("Batch ramp:", ", ".join(f"{b} x {n} steps = {b * n * block_size / 1e9:.3g}B tokens ({b // (world_size * grad_accum)}/GPU)" for b, n in ramp),
              f"x grad accum {grad_accum}")

    eval_steps = max(1, max_steps // args.num_evals)
    plan = Plan(block_size, args.global_batch_size, world_size, per_device_batch, grad_accum, max_steps, ramp, eval_steps,
                int(args.warmup_frac * max_steps), int(args.decay_frac * max_steps))
    if args.eval and eval_steps > max(1, max_steps // 10):
        print(f"WARNING: eval every {eval_steps} steps is less often than the lab's 10% of {max_steps} steps")
    print(f"Token budget {args.token_budget:.3g} -> {max_steps} steps, {plan.tokens_seen(max_steps):,} tokens, eval every {eval_steps}")
    if ramp is None:
        print(f"Global batch {args.global_batch_size} x {block_size} tokens ({per_device_batch}/GPU x grad accum {grad_accum} x {world_size} GPUs)")
    assert plan.tokens_seen(max_steps) <= 6e9 or args.token_budget > 6e9, "over the 6B-token lab budget"
    return plan


def lr_lambda(args, plan):
    """train.py's own LR factor per step for --lr_schedule wsd2 and --schedule_unit tokens (a LambdaLR), else None: the
    Trainer builds its WSD / cosine scheduler from hf_schedule()."""
    if args.schedule_unit == "tokens":
        assert args.decay_type == "linear", "--decay_type is implemented for --schedule_unit steps"
        assert args.lr_schedule == "wsd", "--schedule_unit tokens is implemented for --lr_schedule wsd"
        # Warmup over warmup_frac x budget tokens, peak until (1 - decay_frac) x budget, then linear to 0 at the last step.
        # Before the decay the LR depends only on tokens seen and the nominal budget, so the batch ramp can't change it.
        warmup_tokens = args.warmup_frac * args.token_budget
        decay_start = (1 - args.decay_frac) * args.token_budget
        end_tokens = plan.tokens_seen(plan.max_steps)
        print(f"WSD in tokens: warmup {warmup_tokens:,.0f}, decay from {decay_start:,.0f} to 0 at {end_tokens:,}")

        def wsd_tokens_lambda(step):  # LR of optimizer step step + 1, from the tokens seen before it (as HF: step 0 -> 0)
            t = plan.tokens_seen(step)
            if t < warmup_tokens:
                return t / warmup_tokens
            if t < decay_start:
                return 1.0
            return max(0.0, (end_tokens - t) / (end_tokens - decay_start))
        return wsd_tokens_lambda
    if args.lr_schedule == "wsd2":  # warmup | stable1 | decay1 to r | stable2 at r | decay to 0 (decay_steps)
        max_steps, warmup_steps, decay_steps = plan.max_steps, plan.warmup_steps, plan.decay_steps
        decay1_steps, stable2_steps = int(args.decay1_frac * max_steps), int(args.stable2_frac * max_steps)
        stable1_end = max_steps - decay_steps - stable2_steps - decay1_steps
        assert stable1_end >= warmup_steps, "wsd2: warmup + decay1 + stable2 + decay exceed the run"
        print(f"wsd2: warmup {warmup_steps}, stable1 to step {stable1_end}, decay1 {decay1_steps} to {args.stable2_lr_ratio} x peak, "
              f"stable2 {stable2_steps}, decay {decay_steps} to 0")

        def wsd2_lambda(step):
            r = args.stable2_lr_ratio
            if step < warmup_steps:
                return step / max(1, warmup_steps)  # as HF's get_wsd_schedule
            if step < stable1_end:
                return 1.0
            if step < stable1_end + decay1_steps:
                return 1.0 - (1.0 - r) * (step - stable1_end) / decay1_steps
            if step < max_steps - decay_steps:
                return r
            return r * max(0.0, (max_steps - step) / decay_steps)
        return wsd2_lambda
    return None


def hf_schedule(args, plan):
    """TrainingArguments lr_scheduler_type, lr_scheduler_kwargs, warmup_steps.
    Linear warmup then constant at peak (get_constant_schedule_with_warmup), or with --decay_frac, warmup-stable-decay
    (get_wsd_schedule) with a final linear decay to 0, or with --lr_schedule cosine, cosine from the peak to
    min_lr_ratio x peak (Llama 2 / 3)."""
    decay_steps = plan.decay_steps
    sched_type = "cosine_with_min_lr" if args.lr_schedule == "cosine" else "warmup_stable_decay" if decay_steps else "constant_with_warmup"
    kwargs = ({"min_lr_rate": args.min_lr_ratio} if args.lr_schedule == "cosine"
              else {"num_decay_steps": decay_steps, "decay_type": args.decay_type} if decay_steps else {})
    return sched_type, kwargs, plan.warmup_steps


def ckpt_dir(args):
    return os.path.expanduser(args.ckpt_dir or os.path.join(os.path.expandvars("/work/$USER/llama_ckpt"), args.run_name or "latest"))


def save_steps(args, plan, ckpt_dir):
    """--save_at_tokens -> {step: mark}: step k with tokens_seen(k) <= mark < tokens_seen(k + 1), i.e. where a ramp stage
    of `mark` tokens ends."""
    steps = {}
    if args.save_at_tokens:
        for t in args.save_at_tokens.split(","):
            mark = stage_tokens(t.strip(), args.token_budget)
            k = max(s for s in range(plan.max_steps + 1) if plan.tokens_seen(s) <= mark)
            assert 0 < k < plan.max_steps, f"--save_at_tokens {t}: step {k} is outside the run"
            steps[k] = t.strip()
        print("Checkpoints at steps:", ", ".join(f"{k} ({plan.tokens_seen(k):,} tokens, mark {t})" for k, t in sorted(steps.items())), "->", ckpt_dir)
    return steps


class RampBatchSampler:
    """Index batches for --batch_ramp: per optimizer step, `accum` micro-batches of B/(GPUs x accum) indices per rank,
    in rank order (accelerate's BatchSamplerShard hands batch k to rank k % GPUs; no fixed batch_size attribute, so it accepts
    varying sizes). Random order without replacement, seeded; after the schedule it keeps the last size until the
    data runs out (the Trainer stops at max_steps)."""

    def __init__(self, num_samples, ramp, world_size, seed, accum=1):
        self.n, self.ramp, self.world, self.seed, self.accum = num_samples, ramp, world_size, seed, accum
        used = sum(b * k for b, k in ramp)
        assert used <= num_samples, f"ramp needs {used:,} sequences, dataset has {num_samples:,}"
        self.tail_steps = (num_samples - used) // ramp[-1][0]

    def __len__(self):
        return self.world * self.accum * (sum(k for _, k in self.ramp) + self.tail_steps)

    def __iter__(self):
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed)).tolist()
        sizes = [b for b, k in self.ramp for _ in range(k)] + [self.ramp[-1][0]] * self.tail_steps
        pos = 0
        for b in sizes:
            m = b // (self.world * self.accum)
            for _ in range(self.accum):
                for r in range(self.world):
                    yield perm[pos : pos + m]
                    pos += m
