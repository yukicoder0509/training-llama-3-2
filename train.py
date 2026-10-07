from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import time
from torch.autograd import DeviceType
from adam_mini import Adam_mini
from muon_adamw import build_muon_adamw

START_TIME = time.time()

# Check GPU availability
print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)

# Constants
BLOCK_SIZE = 1024

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Hub repo the saved model is meant for (pushed separately with push_model.py)")
parser.add_argument("--learning_rate", type=float, default=1.25e-3, help="Peak LR, held constant after warmup")
parser.add_argument("--warmup_frac", type=float, default=0.01, help="Linear warmup over this fraction of the steps (1%%: ppl 30.77 vs 32.34 at 10%%, 0.5%%: 31.92; jobs 504660, 504157, 504661)")
parser.add_argument("--decay_frac", type=float, default=0.2, help="0: constant after warmup; >0: warmup-stable-decay, linear decay to 0 over this fraction of the final steps")
parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (GPT-1: 0.01, GPT-3/nanoGPT: 0.1)")
parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2 for both optimizers (0.95: GPT-3/nanoGPT; 0.999: PyTorch default)")
parser.add_argument("--global_batch_size", type=int, default=128, help="Sequences per optimizer step (per-GPU batch x grad accum x GPUs)")
parser.add_argument("--batch_ramp", type=str, default=None, help='Global batch-size ramp "B1:f1,B2:f2,...,Bn": batch Bi for fraction fi of the training time, the last for the rest (e.g. "64:0.1,128"). One micro-batch of Bi/GPUs per GPU per step (no grad accum); overrides --global_batch_size / --per_device_batch')
parser.add_argument("--per_device_batch", type=int, default=128, help="Max sequences per GPU per micro-batch; grad accum covers the rest")
parser.add_argument("--dropout", type=float, default=0.0, help="Sets GPT-2's resid_pdrop, attn_pdrop and embd_pdrop (GPT-2 used 0.1; 0 won at < 0.2 epoch, job 500119)")
parser.add_argument("--pad_vocab", action=argparse.BooleanOptionalAction, default=False, help="Pad the vocab 50257 -> 50304 (multiple of 128) for fast LM-head GEMMs; trimmed back before saving. On in run.sbatch/profile.sbatch; --no-pad_vocab disables")
parser.add_argument("--fused_ce", action=argparse.BooleanOptionalAction, default=False, help="Liger fused LM head + cross-entropy: never materializes the full fp32 logits. On in run.sbatch/profile.sbatch; --no-fused_ce disables")
parser.add_argument("--attn_implementation", type=str, default="sdpa", help="HF attention backend: sdpa (cuDNN flash), flash_attention_3 (Hub kernel kernels-community/vllm-flash-attn3 via `kernels`), flash_attention_4 (pip flash-attn-4, CuTe DSL)")
parser.add_argument("--activation", type=str, default="gelu_pytorch_tanh", help="MLP activation: gelu_pytorch_tanh (fused, same formula) or gelu_new (GPT-2 original, ~8 elementwise kernels)")
parser.add_argument("--sec_per_step", type=float, default=None, help="Override the measured train-step time (s, excl. eval) used to size the run")
parser.add_argument("--max_steps", type=int, default=None, help="Override the time-budgeted step count (e.g. short speed tests)")
parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False, help="torch.compile each transformer block (not the mask setup or the Liger loss): fuses LayerNorm/cast/GELU/elementwise kernels, 1.12x faster steps. On in run.sbatch/profile.sbatch; --no-torch_compile disables")
parser.add_argument("--profile_dir", type=str, default=None, help="Profile 5 steps with torch.profiler into this dir; works inside a full run (see --profile_start) or in profile.sbatch")
parser.add_argument("--profile_start", type=int, default=8, help="Profile steps profile_start+1 .. profile_start+5 (pick a window without an eval step in a full run, e.g. 300)")
parser.add_argument("--profile_cpu", action=argparse.BooleanOptionalAction, default=True, help="Also record CPU ops (shows what causes GPU gaps, slightly slows launches); --no-profile_cpu = GPU kernels only")
parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True, help="Periodic eval (--no-eval for short speed/profile jobs, see profile.sbatch)")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam_mini", "muon"], help="adamw (torch AdamW), adam_mini (Adam-mini, pip adam-mini) or muon (Muon for block matrices + AdamW for the rest, muon_adamw.py)")
parser.add_argument("--muon_lr", type=float, default=1.25e-3, help="--optimizer muon: peak Muon LR (match_rms_adamw scaling; best of 1.25e-3..2e-2, >= 1e-2 diverges); --learning_rate is the AdamW part's")
parser.add_argument("--muon_momentum", type=float, default=0.95, help="--optimizer muon: Muon Nesterov momentum")
parser.add_argument("--data_dir", type=str, default=os.path.expandvars("/work/$USER/c4_gpt2"), help="Dir with train.bin and val.bin")
parser.add_argument("--run_name", type=str, default=None, help="W&B run name; also isolates checkpoints in results/<run_name>")
parser.add_argument("--save_dir", type=str, default=None, help="Where to save the final model (default: ~/gpt2_models/<run_name or latest>)")
args = parser.parse_args()
MODEL_NAME = args.model_name
DATA_DIR = os.path.expanduser(args.data_dir)
SAVE_DIR = os.path.expanduser(args.save_dir or os.path.join("~/gpt2_models", args.run_name or "latest"))

print("Model name:", MODEL_NAME, "LR:", args.learning_rate, "Optimizer:", args.optimizer, *(["Muon LR:", args.muon_lr] if args.optimizer == "muon" else []), "WD:", args.weight_decay, "beta2:", args.beta2, "Run name:", args.run_name, "Data dir:", DATA_DIR)

# The Hub upload runs after the job (push_model.py), so the 30-min limit only covers startup,
# training and a local save: ~40 s startup, local save + W&B finish < 30 s.
JOB_TIME_LIMIT = 1800
# Measured in job 500119 (1449 s total): 9 s from script start to first step, 34 s outside the script
# (venv/accelerate start + W&B finish after the save).
TIME_BUDGET = JOB_TIME_LIMIT - 30 - 45 - 120  # startup, save + W&B finish + launcher, safety margin
# Fixed number of evals (not fixed interval), so eval time doesn't grow when small batches run more steps.
NUM_EVALS = 37
EVAL_SEC = 2.6  # 5k val docs; measured 2.49 s/eval (max 2.54) with gelu_pytorch_tanh (job 500119); was 6.34 with gelu_new
# Train-step time (s, excl. eval) on 2 GPUs by global batch size, with 24 CPUs, gelu_pytorch_tanh, up to
# 64 seqs/GPU, dropout 0, --pad_vocab, --fused_ce and --torch_compile (the run.sbatch defaults).
# 128: full-run mean 0.145 s (job 504863: (1557 s train_runtime - 66 s eval) / 10263) + ~1% margin.
# 64 (32/GPU) and 256 (128/GPU, one micro-batch): speed-job medians 0.081 / 0.264 s (505613 / 505614) x 1.03 (full-run
# mean / median, 505032). 128/GPU is 6% faster per token than 64/GPU x accum 2 (0.280 s) with compile.
# 512: NOT measured with compile (no-compile value x 0.146 / 0.159). Re-measure with profile.sbatch if the setup changes.
# --pad_vocab --fused_ce without compile (speed jobs 504142-504145 x 1.017): 64: 0.093, 128: 0.159, 256: 0.313, 512: 0.627.
# --pad_vocab only (503436-503439): 64: 0.106, 128: 0.198, 256: 0.393, 512: 0.770.
# Neither: 64: 0.161, 128: 0.298, 256: 0.606, 512: 1.208. Pass --sec_per_step to size such runs.
# Before the throughput fixes (1 CPU, gelu_new, 32/GPU): 64: 0.249, 128: 0.469, 256: 0.912, 512: 1.824.
SEC_PER_STEP = {64: 0.083, 128: 0.146, 256: 0.272, 512: 0.576}
# Fewer, larger micro-batches: fewer kernel launches and fewer autocast weight casts per step (profile 499746).
PER_DEVICE_BATCH = min(args.per_device_batch, args.global_batch_size // int(os.environ.get("WORLD_SIZE", 1)))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
assert args.global_batch_size % (PER_DEVICE_BATCH * WORLD_SIZE) == 0, "global batch must be a multiple of per-GPU batch x GPUs"
GRAD_ACCUM = args.global_batch_size // (PER_DEVICE_BATCH * WORLD_SIZE)
sec_per_step = args.sec_per_step or SEC_PER_STEP.get(args.global_batch_size)
if args.max_steps:
    MAX_STEPS = args.max_steps
else:
    assert sec_per_step, f"no measured step time for batch {args.global_batch_size}; run a speed test or pass --sec_per_step"
    MAX_STEPS = int((TIME_BUDGET - NUM_EVALS * EVAL_SEC) / sec_per_step)
# Batch-size ramp: list of (global batch, optimizer steps). Each stage gets its fraction of the training time at its
# own measured step time, so a ramp run spends the same wall time as a fixed-batch run. The LR schedule (warmup /
# WSD decay) runs over the total optimizer steps, unchanged by the ramp.
RAMP = None
if args.batch_ramp:
    stages = [part.split(":") for part in args.batch_ramp.split(",")]
    sizes = [int(st[0]) for st in stages]
    fracs = [float(st[1]) for st in stages[:-1]]
    fracs.append(1 - sum(fracs))
    assert all(b % WORLD_SIZE == 0 for b in sizes) and fracs[-1] > 0, args.batch_ramp
    assert all(b in SEC_PER_STEP for b in sizes), f"no measured step time for some of {sizes}; run speed tests first"
    train_time = TIME_BUDGET - NUM_EVALS * EVAL_SEC
    RAMP = [(b, int(f * train_time / SEC_PER_STEP[b])) for b, f in zip(sizes, fracs)]
    if args.max_steps:  # short tests: keep the stage proportions in steps
        total = sum(n for _, n in RAMP)
        RAMP = [(b, max(1, round(n * args.max_steps / total))) for b, n in RAMP]
    MAX_STEPS = sum(n for _, n in RAMP)
    PER_DEVICE_BATCH, GRAD_ACCUM = sizes[0] // WORLD_SIZE, 1
    print("Batch ramp:", ", ".join(f"{b} x {n} steps" for b, n in RAMP))


def batch_at(step):
    """Global batch size of optimizer step `step` (1-based)."""
    if RAMP is None:
        return args.global_batch_size
    for b, n in RAMP:
        if step <= n:
            return b
        step -= n
    return RAMP[-1][0]


class RampBatchSampler:
    """Index batches for --batch_ramp: per optimizer step, one batch of B/GPUs indices per rank, in rank order
    (accelerate's BatchSamplerShard hands batch k to rank k % GPUs; no fixed batch_size attribute, so it accepts
    varying sizes). Random order without replacement, seeded; after the schedule it keeps the last size until the
    data runs out (the Trainer stops at max_steps)."""

    def __init__(self, num_samples, ramp, world_size, seed):
        self.n, self.ramp, self.world, self.seed = num_samples, ramp, world_size, seed
        used = sum(b * k for b, k in ramp)
        assert used <= num_samples, f"ramp needs {used:,} sequences, dataset has {num_samples:,}"
        self.tail_steps = (num_samples - used) // ramp[-1][0]

    def __len__(self):
        return self.world * (sum(k for _, k in self.ramp) + self.tail_steps)

    def __iter__(self):
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed)).tolist()
        sizes = [b for b, k in self.ramp for _ in range(k)] + [self.ramp[-1][0]] * self.tail_steps
        pos = 0
        for b in sizes:
            m = b // self.world
            for r in range(self.world):
                yield perm[pos + r * m : pos + (r + 1) * m]
            pos += b


EVAL_STEPS = max(1, MAX_STEPS // NUM_EVALS)
if RAMP is None:
    print(f"Global batch {args.global_batch_size} ({PER_DEVICE_BATCH}/GPU x grad accum {GRAD_ACCUM}), {MAX_STEPS} steps, eval every {EVAL_STEPS}")
else:
    print(f"Batch ramp {args.batch_ramp}: {MAX_STEPS} steps, eval every {EVAL_STEPS}")
WARMUP_STEPS = int(args.warmup_frac * MAX_STEPS)
DECAY_STEPS = int(args.decay_frac * MAX_STEPS)

# Prepare tokenizer and model
print("=== Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Weight init as in GPT-1/GPT-2: N(0, 0.02) for Linear/Conv1D/embedding weights, zero biases; GPT2's
# _init_weights also scales each block's residual output proj (c_proj) by 1/sqrt(2 * n_layer).
# 0.02 is already gpt2's config value; set explicitly so the choice is visible.
# gelu_pytorch_tanh is the same tanh approximation as GPT-2's gelu_new, but one fused kernel instead of ~8
# elementwise ones (generic elementwise kernels were ~29% of GPU time in profile 499746).
config = AutoConfig.from_pretrained(
    "gpt2", initializer_range=0.02, activation_function=args.activation,
    resid_pdrop=args.dropout, attn_pdrop=args.dropout, embd_pdrop=args.dropout,
    # The odd 50257 vocab sends the LM-head GEMMs to a slow sm75 cutlass kernel (~40% of GPU time, profile 503405).
    # Padded rows never appear as targets; training just pushes their logits down.
    **({"vocab_size": 50304} if args.pad_vocab else {}),
)
model = AutoModelForCausalLM.from_config(config, attn_implementation=args.attn_implementation)
print("Attention implementation:", model.config._attn_implementation)

# Fused LM head + cross-entropy (Liger): computes the logits, loss and their gradients chunk by chunk inside
# the forward, instead of HF's full (tokens x vocab) logits upcast to fp32 for log_softmax (~13% of GPU time and
# most of the memory, profile 503423). Same shift and normalization as HF's ForCausalLMLoss: sum / num_items_in_batch
# when the Trainer passes it (grad accum), else mean. Patched on the instance so the saved config stays GPT2LMHeadModel.
if args.fused_ce:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    fused_ce = {r: LigerFusedLinearCrossEntropyLoss(reduction=r) for r in ("mean", "sum")}

    def fused_ce_forward(input_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        hidden = model.transformer(input_ids=input_ids, **kwargs).last_hidden_state
        if labels is None:
            return CausalLMOutputWithCrossAttentions(logits=model.lm_head(hidden))
        hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))  # tokens < n predict n
        targets = labels[:, 1:].reshape(-1)
        if num_items_in_batch is None:
            loss = fused_ce["mean"](model.lm_head.weight, hidden, targets)
        else:
            loss = fused_ce["sum"](model.lm_head.weight, hidden, targets) / num_items_in_batch
        return CausalLMOutputWithCrossAttentions(loss=loss)

    model.forward = fused_ce_forward

# Compile only the transformer body: the memory-bound LayerNorm / cast / GELU / elementwise kernels are ~37% of GPU
# time there (profile 504143), while the LM head + loss stays in Liger's kernels outside the compiled graph.
if args.torch_compile:
    if args.batch_ramp:  # a static recompile per batch size, not slower dynamic-shape kernels after the first change
        torch._dynamo.config.automatic_dynamic_shapes = False
    for block in model.transformer.h:
        block.compile()

# Load English C4 dataset
print("=== Loading English C4 dataset...")
class TokenBlocks(Dataset):
    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")  # lazy, no RAM blowup
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i * self.block_size : (i + 1) * self.block_size].astype(np.int64))
        return {"input_ids": x, "labels": x}  # model shifts labels internally)

train_ds = TokenBlocks(os.path.join(DATA_DIR, "train.bin"), BLOCK_SIZE)
val_ds = TokenBlocks(os.path.join(DATA_DIR, "val.bin"), BLOCK_SIZE)

# Training arg
training_args = TrainingArguments(
    output_dir=os.path.join("./results", args.run_name or ""),
    run_name=args.run_name,
    # Global batch = per_device * grad_accum * num_gpus (default 32 * 2 * 2 = 128 sequences; GPT-1 used 512)
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    bf16=True,
    num_train_epochs=1,
    max_steps=MAX_STEPS,

    # Logging, eval and reporting
    logging_steps=10,
    report_to="wandb",  # Log to W&B
    eval_strategy="steps" if args.eval else "no",
    eval_steps=EVAL_STEPS,
    save_strategy="no",

    # Optimizer
    optim="adamw_torch",
    learning_rate=args.learning_rate,
    adam_beta1=0.9,
    adam_beta2=args.beta2,
    adam_epsilon=1e-8,
    weight_decay=args.weight_decay,  # [choice] GPT-1 used 0.01; 0.1 is the GPT-3/nanoGPT value

    # Scheduler: linear warmup then constant at peak (get_constant_schedule_with_warmup),
    # or with --decay_frac, warmup-stable-decay (get_wsd_schedule) with a final linear decay to 0.
    # Was: cosine to 10% of peak ("cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1})
    lr_scheduler_type="warmup_stable_decay" if DECAY_STEPS else "constant_with_warmup",
    lr_scheduler_kwargs={"num_decay_steps": DECAY_STEPS, "decay_type": "linear"} if DECAY_STEPS else {},
    warmup_steps=WARMUP_STEPS,  # --warmup_frac of the run (default 1%)
)
print("=== Training arguments: ", training_args)

# Safety net: if training runs slower than estimated (e.g. a slow node), stop early so the model
# is still saved before Slurm kills the job. The LR decay is cut short in that case.
class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        stop = torch.tensor(float(time.time() > self.deadline), device=args.device)
        if torch.distributed.is_initialized():  # all ranks must stop at the same step or the next collective hangs
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            print(f"=== Deadline reached at step {state.global_step}/{state.max_steps}, stopping to save the model")
            control.should_training_stop = True
        return control

# Times each optimizer step (on_step_begin -> on_step_end excludes eval) and prints the median at the end,
# to fill SEC_PER_STEP for new batch sizes.
# Model FLOPs per token (fwd + bwd): 6 * params + attention 12 * n_layer * n_embd * seq_len (PaLM/nanoGPT
# MFU formula). Peak: H200 dense bf16 ~989 TFLOPS per GPU.
FLOPS_PER_TOKEN = 6 * sum(p.numel() for p in model.parameters()) + 12 * config.n_layer * config.n_embd * BLOCK_SIZE
PEAK_FLOPS_PER_GPU = 989e12

class StepTimerCallback(TrainerCallback):
    def __init__(self):
        self.times, self.steps, self.t0 = [], [], None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.times.append(time.time() - self.t0)
        self.steps.append(state.global_step)
        if state.global_step == 1 and state.is_world_process_zero:  # includes torch.compile time if enabled
            print(f"=== First step done at {time.time() - START_TIME:.0f}s ({self.times[0]:.1f}s for step 1)")

    def recent_throughput(self, num_steps):
        """Training tokens/s (all GPUs) and MFU over the last num_steps optimizer steps, eval excluded."""
        recent = self.times[-num_steps:]
        tokens_per_sec = BLOCK_SIZE * sum(batch_at(s) for s in self.steps[-num_steps:]) / sum(recent)
        return tokens_per_sec, tokens_per_sec * FLOPS_PER_TOKEN / (PEAK_FLOPS_PER_GPU * WORLD_SIZE)

    def on_train_end(self, args, state, control, **kwargs):
        gib = 1024 ** 3
        total = torch.cuda.get_device_properties(args.device).total_memory
        print(f"=== Peak GPU memory (rank {args.process_index}): allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB of {total / gib:.0f} GiB")
        if state.is_world_process_zero and len(self.times) > 10:
            t = sorted(self.times[5:])  # skip warm-up steps
            print(f"=== Step time: median {t[len(t) // 2]:.3f} s over {len(t)} steps (excl. eval)")

# Optimizer. Adam-mini groups params by name: one lr per row for embeddings/MLP, one per tensor for the
# rest. HF GPT-2 names: wte (matched), wpe (added below), attn.c_attn (fused QKV) and attn.c_proj are not
# matched, so they get one lr per tensor; mlp.c_fc/c_proj are matched but are Conv1D (in, out), so
# "per row" is per input feature rather than per output neuron.
optimizer = None
if args.optimizer == "adam_mini":
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=args.learning_rate,
        betas=(0.9, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,  # Adam-mini skips decay on norm and bias params itself
        dim=config.n_embd,
        n_heads=config.n_head,
        verbose=int(os.environ.get("LOCAL_RANK", 0)) == 0,
    )
    optimizer.embd_names.add("wpe")
elif args.optimizer == "muon":
    optimizer, muon_names, adamw_names = build_muon_adamw(
        model, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum,
        adamw_lr=args.learning_rate, beta2=args.beta2, weight_decay=args.weight_decay,
    )
    n = dict(model.named_parameters())
    print(f"Muon: {len(muon_names)} tensors, {sum(n[k].numel() for k in muon_names) / 1e6:.1f}M params, lr {args.muon_lr}; "
          f"AdamW: {len(adamw_names)} tensors, {sum(n[k].numel() for k in adamw_names) / 1e6:.1f}M params, lr {args.learning_rate}")

# Trainer
step_timer = StepTimerCallback()

# torch.profiler over 5 steady-state steps: Chrome trace per rank, plus (rank 0) top ops by GPU and CPU time and the
# GPU busy fraction (sum of GPU-side events: kernels, memcpy, memset / wall time; NCCL overlap can push it slightly
# above 100%). Same sum as the table footer: CPU ops (aten::mm) and GPU-side user annotations (ProfilerStep*,
# DistributedDataParallel.forward) are excluded, since their self device time repeats the time of the kernels they
# launch or enclose. The GPU is synchronized only at the two window edges (to time the window), so the CPU can queue
# work ahead as in an unprofiled run; a sync after every step exposed the batch-loading gap as ~9% fake idle (504143).
class ProfilerCallback(TrainerCallback):
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

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        if RAMP is None:
            return super().get_train_dataloader()
        sampler = RampBatchSampler(len(self.train_dataset), RAMP, WORLD_SIZE, self.args.seed)
        return self.accelerator.prepare(DataLoader(self.train_dataset, batch_sampler=sampler,
                                                   collate_fn=self.data_collator, pin_memory=True))

    # Log PPL
    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
            if step_timer.times:
                logs["tokens_per_sec"], logs["mfu"] = step_timer.recent_throughput(self.args.logging_steps)
        if "eval_loss" in logs:
            logs["eval_ppl"] = math.exp(logs["eval_loss"])
        super().log(logs, *args, **kwargs) # Override the log to include perplexity (ppl)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[DeadlineCallback(START_TIME + JOB_TIME_LIMIT - 90), step_timer]
    + ([ProfilerCallback(args.profile_dir, args.profile_start, args.profile_cpu)] if args.profile_dir else []),
    optimizers=(optimizer, None),  # None -> Trainer builds AdamW from args; scheduler always from args
)

# Train
print("=== Starting training...")
trainer.train()
if args.pad_vocab:
    model.resize_token_embeddings(len(tokenizer))  # drop padded rows (tied wte/lm_head) so the saved model is plain GPT-2
trainer.save_model(SAVE_DIR)  # main process only; config + generation_config + safetensors
print(f"=== Training finished. Saved to {SAVE_DIR} at {time.time() - START_TIME:.0f}s. Push with:")
print(f"    python push_model.py --model_dir {SAVE_DIR} --repo_id {MODEL_NAME}")

# OpenAI team hyperparameters
"""
Batch size: 512 sequences.
Adam with a max learning rate of 2.5e-4
Linear warmup over 2,000 updates, then cosine annealing
Dropout of 0.1
Modified L2 weight decay of 0.01

If you're trying to reproduce GPT-2 training,
 community reproductions like Karpathy's nanoGPT and llm.c 
 are the practical reference. 
 They fill in the missing values, 
 for example a peak learning rate around 6e-4 for 124M and AdamW with betas (0.9, 0.95). Those values are their choices, not OpenAI's.
"""