from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import subprocess
import time
from torch.autograd import DeviceType
from adam_mini import Adam_mini
from muon_adamw import build_muon_adamw

START_TIME = time.time()

# Check GPU availability
print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)

# Constants
BLOCK_SIZE = 8192  # context window (DESCRIPTION.md)
TOKENIZER = "NousResearch/Llama-3.2-1B"  # tokenizer.json byte-identical to meta-llama/Llama-3.2-1B (gated for us)

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Hub repo the saved model is meant for (pushed separately with push_model.py)")
parser.add_argument("--token_budget", type=float, default=6e9, help="Training tokens (lab cap 6B; every token of a step counts, there is no padding). Sets the step count")
parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak LR (untuned starting point for 1B at ~0.5M-token batches)")
parser.add_argument("--warmup_frac", type=float, default=0.02, help="Linear warmup over this fraction of the steps")
parser.add_argument("--decay_frac", type=float, default=0.2, help="--lr_schedule wsd: 0 = constant after warmup; >0 = linear decay to 0 over this fraction of the final steps")
parser.add_argument("--lr_schedule", type=str, default="wsd", choices=["wsd", "cosine"], help="wsd (warmup-stable-decay, see --decay_frac) or cosine (Llama 2/3: warmup, then cosine to --min_lr_ratio x peak)")
parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="--lr_schedule cosine: final LR / peak (Llama 2: 0.1)")
parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon (Llama 2: 1e-5)")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (Llama 3 / GPT-3: 0.1)")
parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2 for both optimizers (0.95: Llama/GPT-3; 0.999: PyTorch default)")
parser.add_argument("--global_batch_size", type=int, default=64, help="Sequences of 8192 tokens per optimizer step (per-GPU batch x grad accum x GPUs); 64 = 0.5M tokens")
parser.add_argument("--batch_ramp", type=str, default=None, help='Global batch-size ramp "B1:f1,B2:f2,...,Bn": batch Bi for fraction fi of the token budget, the last for the rest (e.g. "32:0.1,64"). One micro-batch of Bi/GPUs per GPU per step (no grad accum); overrides --global_batch_size / --per_device_batch')
parser.add_argument("--per_device_batch", type=int, default=4, help="Max sequences per GPU per micro-batch; grad accum covers the rest")
parser.add_argument("--fused_ce", action=argparse.BooleanOptionalAction, default=False, help="Liger fused LM head + cross-entropy: never materializes the full fp32 logits (128k vocab x 8192 tokens = 4 GiB per sequence). On in run.sbatch/profile.sbatch; --no-fused_ce disables")
parser.add_argument("--attn_implementation", type=str, default="sdpa", help="HF attention backend: sdpa (cuDNN flash), flash_attention_3 (Hub kernel kernels-community/vllm-flash-attn3 via `kernels`), flash_attention_4 (pip flash-attn-4, CuTe DSL)")
parser.add_argument("--max_steps", type=int, default=None, help="Override the token-budget step count (e.g. short speed tests)")
parser.add_argument("--num_evals", type=int, default=20, help="Evals over the run (lab: at least every 10%% of the steps)")
parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False, help="torch.compile each decoder layer (not the rotary/mask setup or the Liger loss): fuses RMSNorm/cast/SwiGLU/elementwise kernels. On in run.sbatch/profile.sbatch; --no-torch_compile disables")
parser.add_argument("--profile_dir", type=str, default=None, help="Profile 5 steps with torch.profiler into this dir; works inside a full run (see --profile_start) or in profile.sbatch")
parser.add_argument("--profile_start", type=int, default=8, help="Profile steps profile_start+1 .. profile_start+5 (pick a window without an eval step in a full run, e.g. 300)")
parser.add_argument("--profile_cpu", action=argparse.BooleanOptionalAction, default=True, help="Also record CPU ops (shows what causes GPU gaps, slightly slows launches); --no-profile_cpu = GPU kernels only")
parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True, help="Periodic eval (--no-eval for short speed/profile jobs, see profile.sbatch)")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam_mini", "muon"], help="adamw (torch AdamW), adam_mini (Adam-mini, pip adam-mini) or muon (Muon for decoder-layer matrices + AdamW for the rest, muon_adamw.py)")
parser.add_argument("--muon_lr", type=float, default=1.25e-3, help="--optimizer muon: peak Muon LR (match_rms_adamw scaling; tuned on GPT-2, re-tune for Llama); --learning_rate is the AdamW part's")
parser.add_argument("--muon_momentum", type=float, default=0.95, help="--optimizer muon: Muon Nesterov momentum")
parser.add_argument("--data_dir", type=str, default=os.path.expandvars("/work/$USER/dolma3_llama"), help="Dir with train.bin and val.bin (uint32, prepare_data.py)")
parser.add_argument("--run_name", type=str, default=None, help="W&B run name; also isolates checkpoints in results/<run_name>")
parser.add_argument("--save_dir", type=str, default=None, help="Where to save the final model + tokenizer (default: /work/$USER/llama_models/<run_name or latest>)")
args = parser.parse_args()
MODEL_NAME = args.model_name
DATA_DIR = os.path.expanduser(args.data_dir)
SAVE_DIR = os.path.expanduser(args.save_dir or os.path.join(os.path.expandvars("/work/$USER/llama_models"), args.run_name or "latest"))

print("Model name:", MODEL_NAME, "LR:", args.learning_rate, "Optimizer:", args.optimizer, *(["Muon LR:", args.muon_lr] if args.optimizer == "muon" else []), "WD:", args.weight_decay, "beta2:", args.beta2, "Run name:", args.run_name, "Data dir:", DATA_DIR)


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


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
PER_DEVICE_BATCH = min(args.per_device_batch, args.global_batch_size // WORLD_SIZE)
assert args.global_batch_size % (PER_DEVICE_BATCH * WORLD_SIZE) == 0, "global batch must be a multiple of per-GPU batch x GPUs"
GRAD_ACCUM = args.global_batch_size // (PER_DEVICE_BATCH * WORLD_SIZE)
# Token budget -> steps. Every position of every sequence is a real token (packed documents, no padding).
MAX_STEPS = args.max_steps or int(args.token_budget // (args.global_batch_size * BLOCK_SIZE))
# Batch-size ramp: list of (global batch, optimizer steps). Each stage gets its fraction of the token budget. The LR
# schedule (warmup / WSD decay) runs over the total optimizer steps, unchanged by the ramp.
RAMP = None
if args.batch_ramp:
    stages = [part.split(":") for part in args.batch_ramp.split(",")]
    sizes = [int(st[0]) for st in stages]
    fracs = [float(st[1]) for st in stages[:-1]]
    fracs.append(1 - sum(fracs))
    assert all(b % WORLD_SIZE == 0 for b in sizes) and fracs[-1] > 0, args.batch_ramp
    RAMP = [(b, int(f * args.token_budget // (b * BLOCK_SIZE))) for b, f in zip(sizes, fracs)]
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


def tokens_seen(step):
    """Training tokens processed up to and including optimizer step `step`."""
    if RAMP is None:
        return step * args.global_batch_size * BLOCK_SIZE
    total = 0
    for b, n in RAMP:
        total += min(step, n) * b * BLOCK_SIZE
        step -= min(step, n)
    return total + step * RAMP[-1][0] * BLOCK_SIZE


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


EVAL_STEPS = max(1, MAX_STEPS // args.num_evals)
if args.eval and EVAL_STEPS > max(1, MAX_STEPS // 10):
    print(f"WARNING: eval every {EVAL_STEPS} steps is less often than the lab's 10% of {MAX_STEPS} steps")
print(f"Token budget {args.token_budget:.3g} -> {MAX_STEPS} steps, {tokens_seen(MAX_STEPS):,} tokens, eval every {EVAL_STEPS}")
if RAMP is None:
    print(f"Global batch {args.global_batch_size} x {BLOCK_SIZE} tokens ({PER_DEVICE_BATCH}/GPU x grad accum {GRAD_ACCUM} x {WORLD_SIZE} GPUs)")
assert tokens_seen(MAX_STEPS) <= 6e9 or args.token_budget > 6e9, "over the 6B-token lab budget"
WARMUP_STEPS = int(args.warmup_frac * MAX_STEPS)
DECAY_STEPS = int(args.decay_frac * MAX_STEPS)

# Prepare tokenizer and model
print("=== Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

# Llama 3.2 1B architecture (meta-llama/Llama-3.2-1B config.json), randomly initialized: N(0, 0.02) Linear/embedding
# weights (LlamaPreTrainedModel._init_weights), tied embeddings, GQA 32 query / 8 KV heads, rope_theta 5e5 with the
# llama3 RoPE scaling. Written out instead of loaded since the meta-llama repo is gated for us.
# The 128,256 vocab is already a multiple of 128 (fast LM-head GEMMs), so no vocab padding is needed.
config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
    num_attention_heads=32, num_key_value_heads=8, head_dim=64, hidden_act="silu",
    max_position_embeddings=131072, rms_norm_eps=1e-5, initializer_range=0.02, tie_word_embeddings=True,
    rope_theta=500000.0,
    rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192},
    attention_bias=False, attention_dropout=0.0, mlp_bias=False,
    bos_token_id=128000, eos_token_id=128001, pad_token_id=128004, use_cache=False,
)
model = AutoModelForCausalLM.from_config(config, attn_implementation=args.attn_implementation)
print("Attention implementation:", model.config._attn_implementation)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B")

# Fused LM head + cross-entropy (Liger): computes the logits, loss and their gradients chunk by chunk inside
# the forward, instead of HF's full (tokens x vocab) logits upcast to fp32 for log_softmax. Same shift and
# normalization as HF's ForCausalLMLoss: sum / num_items_in_batch when the Trainer passes it (grad accum), else mean.
# Patched on the instance so the saved config stays LlamaForCausalLM.
if args.fused_ce:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast
    fused_ce = {r: LigerFusedLinearCrossEntropyLoss(reduction=r) for r in ("mean", "sum")}

    def fused_ce_forward(input_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        hidden = model.model(input_ids=input_ids, **kwargs).last_hidden_state
        if labels is None:
            return CausalLMOutputWithPast(logits=model.lm_head(hidden))
        hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))  # tokens < n predict n
        targets = labels[:, 1:].reshape(-1)
        if num_items_in_batch is None:
            loss = fused_ce["mean"](model.lm_head.weight, hidden, targets)
        else:
            loss = fused_ce["sum"](model.lm_head.weight, hidden, targets) / num_items_in_batch
        return CausalLMOutputWithPast(loss=loss)

    model.forward = fused_ce_forward

# Compile only the decoder layers: the memory-bound RMSNorm / cast / SwiGLU / elementwise kernels, while the LM head +
# loss stays in Liger's kernels outside the compiled graph (compiling the whole body made GPT-2's attention 2.7x slower).
if args.torch_compile:
    if args.batch_ramp:  # a static recompile per batch size, not slower dynamic-shape kernels after the first change
        torch._dynamo.config.automatic_dynamic_shapes = False
    for layer in model.model.layers:
        layer.compile()

# Load Dolma 3 (prepare_data.py): flat uint32 token files of <|begin_of_text|> doc <|end_of_text|>, packed into blocks
print("=== Loading Dolma 3 tokens...")
class TokenBlocks(Dataset):
    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")  # lazy, no RAM blowup
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i * self.block_size : (i + 1) * self.block_size].astype(np.int64))
        return {"input_ids": x, "labels": x}  # model shifts labels internally

train_ds = TokenBlocks(os.path.join(DATA_DIR, "train.bin"), BLOCK_SIZE)
val_ds = TokenBlocks(os.path.join(DATA_DIR, "val.bin"), BLOCK_SIZE)
print(f"Train: {len(train_ds):,} blocks ({len(train_ds) * BLOCK_SIZE / 1e9:.2f}B tokens), val: {len(val_ds):,} blocks")
# One pass at most: the budget must not repeat tokens unintentionally (repeats would count against it anyway)
assert tokens_seen(MAX_STEPS) <= len(train_ds) * BLOCK_SIZE, "train.bin has fewer tokens than the budget; raise --train_frac in prepare_data.py"

# Training arg
training_args = TrainingArguments(
    output_dir=os.path.join("./results", args.run_name or ""),
    run_name=args.run_name,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    bf16=True,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    max_grad_norm=1.0,

    # Logging, eval and reporting
    logging_steps=10,
    report_to="wandb",  # Log to W&B (project from WANDB_PROJECT, see run.sbatch)
    eval_strategy="steps" if args.eval else "no",
    eval_steps=EVAL_STEPS,
    save_strategy="no",

    # Optimizer
    optim="adamw_torch_fused",
    learning_rate=args.learning_rate,
    adam_beta1=0.9,
    adam_beta2=args.beta2,
    adam_epsilon=args.adam_eps,
    weight_decay=args.weight_decay,

    # Scheduler: linear warmup then constant at peak (get_constant_schedule_with_warmup),
    # or with --decay_frac, warmup-stable-decay (get_wsd_schedule) with a final linear decay to 0,
    # or with --lr_schedule cosine, cosine from the peak to min_lr_ratio x peak (Llama 2 / 3).
    lr_scheduler_type="cosine_with_min_lr" if args.lr_schedule == "cosine" else "warmup_stable_decay" if DECAY_STEPS else "constant_with_warmup",
    lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio} if args.lr_schedule == "cosine"
    else {"num_decay_steps": DECAY_STEPS, "decay_type": "linear"} if DECAY_STEPS else {},
    warmup_steps=WARMUP_STEPS,
)
print("=== Training arguments: ", training_args)

# Safety net: stop early so the model is still saved before Slurm kills the job (e.g. a slow node, or a budget that
# doesn't fit the job's time limit). The LR decay is cut short in that case.
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

# Times each optimizer step (on_step_begin -> on_step_end excludes eval) and prints the median at the end.
# Model FLOPs per token (fwd + bwd): 6 * params + attention 12 * n_layer * hidden * seq_len (PaLM/nanoGPT
# MFU formula). Peak: H200 dense bf16 ~989 TFLOPS per GPU.
FLOPS_PER_TOKEN = 6 * sum(p.numel() for p in model.parameters()) + 12 * config.num_hidden_layers * config.hidden_size * BLOCK_SIZE
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

    def last_step_throughput(self):
        """Training tokens/s (all GPUs) and MFU of the latest optimizer step (the lab logs it "on that step")."""
        tokens_per_sec = BLOCK_SIZE * batch_at(self.steps[-1]) / self.times[-1]
        return tokens_per_sec, tokens_per_sec * FLOPS_PER_TOKEN / (PEAK_FLOPS_PER_GPU * WORLD_SIZE)

    def on_train_end(self, args, state, control, **kwargs):
        gib = 1024 ** 3
        total = torch.cuda.get_device_properties(args.device).total_memory
        print(f"=== Peak GPU memory (rank {args.process_index}): allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB of {total / gib:.0f} GiB")
        if state.is_world_process_zero and len(self.times) > 10:
            t = sorted(self.times[5:])  # skip warm-up steps
            med = t[len(t) // 2]
            print(f"=== Step time: median {med:.3f} s over {len(t)} steps (excl. eval), "
                  f"{BLOCK_SIZE * batch_at(state.global_step) / med:,.0f} tokens/s")

# Optimizer. Adam-mini recognizes the HF Llama names itself: embed_tokens (tied lm_head) one lr per row,
# q_proj/k_proj one lr per head, the rest one per output neuron / tensor.
optimizer = None
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

# Trainer
step_timer = StepTimerCallback()

# torch.profiler over 5 steady-state steps: Chrome trace per rank, plus (rank 0) top ops by GPU and CPU time and the
# GPU busy fraction (sum of GPU-side events: kernels, memcpy, memset / wall time; NCCL overlap can push it slightly
# above 100%). Same sum as the table footer: CPU ops (aten::mm) and GPU-side user annotations (ProfilerStep*,
# DistributedDataParallel.forward) are excluded, since their self device time repeats the time of the kernels they
# launch or enclose. The GPU is synchronized only at the two window edges (to time the window), so the CPU can queue
# work ahead as in an unprofiled run; a sync after every step exposed the batch-loading gap as ~9% fake idle.
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

    # Lab metrics. WandbCallback prefixes train logs with "train/" and "eval_*" with "eval/": loss (mean over the
    # logging interval), grad_norm and learning_rate come from the Trainer; tokens_per_second (of the logging step
    # itself), total_tokens_seen and eval_perplexity are added here.
    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)
            if step_timer.times:
                logs["tokens_per_second"], logs["mfu"] = step_timer.last_step_throughput()
        if "eval_loss" in logs:
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)  # x-axis for eval/perplexity vs tokens
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        super().log(logs, *args, **kwargs)


time_left = slurm_time_left()
deadline = START_TIME + time_left - 300 if time_left else float("inf")  # leave 5 min for the final eval + save
print("Deadline:", f"{(deadline - START_TIME) / 3600:.2f} h from start" if time_left else "none")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[DeadlineCallback(deadline), step_timer]
    + ([ProfilerCallback(args.profile_dir, args.profile_start, args.profile_cpu)] if args.profile_dir else []),
    optimizers=(optimizer, None),  # None -> Trainer builds AdamW from args; scheduler always from args
)

# Train
print("=== Starting training...")
trainer.train()
if args.eval and trainer.state.global_step % EVAL_STEPS:  # final eval unless the last step just ran one
    trainer.evaluate()
print(f"=== Trained on {tokens_seen(trainer.state.global_step):,} tokens in {trainer.state.global_step} steps")
trainer.save_model(SAVE_DIR)  # main process only; config + generation_config + safetensors
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(SAVE_DIR)  # the OJ pulls the tokenizer from the same repo
    # transformers 5 writes RoPE as "rope_parameters"; 4.x would ignore it and fall back to rope_theta 10000.
    # Add the 4.x keys too so the model evaluates the same with either version.
    import json
    cfg_path = os.path.join(SAVE_DIR, "config.json")
    cfg = json.load(open(cfg_path))
    rope = dict(cfg.get("rope_parameters") or {})
    cfg.setdefault("rope_theta", rope.pop("rope_theta", 500000.0))
    cfg.setdefault("rope_scaling", rope or None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"=== Training finished. Saved to {SAVE_DIR} at {time.time() - START_TIME:.0f}s. Push with:")
print(f"    python push_model.py --model_dir {SAVE_DIR} --repo_id {MODEL_NAME}")
