from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, set_seed
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import contextlib
import subprocess
import time
from torch.autograd import DeviceType
from adam_mini import Adam_mini
from muon_adamw import build_muon_adamw
from sophia import SophiaG
from soap_dist import DistributedSOAP

START_TIME = time.time()

# Check GPU availability
print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)

# Constants
TOKENIZER = "NousResearch/Llama-3.2-1B"  # tokenizer.json byte-identical to meta-llama/Llama-3.2-1B (gated for us)

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Hub repo the saved model is meant for (pushed separately with push_model.py)")
parser.add_argument("--token_budget", type=float, default=6e9, help="Training tokens (lab cap 6B; every token of a step counts, there is no padding). Sets the step count")
parser.add_argument("--learning_rate", type=float, default=None, help="Peak LR; with muon / muonium the AdamW (embedding, norms) LR. Default: 3e-3 for muonium (best run), 6e-4 otherwise (untuned starting point for 1B at ~0.5M-token batches)")
parser.add_argument("--warmup_frac", type=float, default=0.02, help="Linear warmup over this fraction of the steps")
parser.add_argument("--decay_frac", type=float, default=0.1, help="--lr_schedule wsd: 0 = constant after warmup; >0 = linear decay to 0 over this fraction of the final steps (default 0.1, MiniCPM WSD; ties 0.2 for Sophia at 1.1B, experiments.md 2026-10-14; runs before that used 0.2)")
parser.add_argument("--lr_schedule", type=str, default="wsd", choices=["wsd", "wsd2", "cosine"], help="wsd (warmup-stable-decay, see --decay_frac), wsd2 (warmup, stable, linear decay to --stable2_lr_ratio over --decay1_frac, stable2 at that LR for --stable2_frac, then linear decay to 0 over --decay_frac) or cosine (Llama 2/3: warmup, then cosine to --min_lr_ratio x peak)")
parser.add_argument("--decay1_frac", type=float, default=0.05, help="--lr_schedule wsd2: first decay (peak -> --stable2_lr_ratio x peak) over this fraction of the steps")
parser.add_argument("--stable2_frac", type=float, default=0.1, help="--lr_schedule wsd2: second stable stage at --stable2_lr_ratio x peak, fraction of the steps")
parser.add_argument("--stable2_lr_ratio", type=float, default=0.1, help="--lr_schedule wsd2: LR of the second stable stage / peak")
parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="--lr_schedule cosine: final LR / peak (Llama 2: 0.1)")
parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon (Llama 2: 1e-5)")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (Llama 3 / GPT-3: 0.1)")
parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2 for both optimizers (0.95: Llama/GPT-3; 0.999: PyTorch default)")
parser.add_argument("--block_size", type=int, default=8192, help="Context window / training block (8192 = DESCRIPTION.md; train_scorer.py: 2048)")
parser.add_argument("--config_overrides", type=str, default=None, help='JSON dict applied to the Llama 3.2 1B config with config.update, e.g. a smaller model (train_scorer.py)')
parser.add_argument("--global_batch_size", type=int, default=None, help="Sequences of 8192 tokens per optimizer step (per-GPU batch x grad accum x GPUs); 64 = 0.5M tokens. Default 64; giving it turns off the default --batch_ramp")
parser.add_argument("--batch_ramp", type=str, default=None, help='Default "32:0.3333,64" (best run) unless --global_batch_size is given; "none" for a fixed batch. Global batch-size ramp "B1:t1,B2:t2,...,Bn": batch Bi for the next ti tokens, the last for the rest. ti > 1 is a token count ("32:1e9,64", or with a K/M/B suffix: "32:1B,64"), ti <= 1 a fraction of the token budget ("32:0.1,64"). Grad accum is fixed (enough for the largest Bi at --per_device_batch), so the micro-batch is Bi/(GPUs x accum); overrides --global_batch_size')
parser.add_argument("--per_device_batch", type=int, default=4, help="Max sequences per GPU per micro-batch; grad accum covers the rest")
parser.add_argument("--fused_ce", action=argparse.BooleanOptionalAction, default=False, help="Liger fused LM head + cross-entropy: never materializes the full fp32 logits (128k vocab x 8192 tokens = 4 GiB per sequence). On in run.sbatch/profile.sbatch; --no-fused_ce disables")
parser.add_argument("--attn_implementation", type=str, default="flash_attention_3", help="HF attention backend (default flash_attention_3: needed by --packing doc): sdpa (cuDNN flash), flash_attention_3 (Hub kernel kernels-community/vllm-flash-attn3 via `kernels`), flash_attention_4 (pip flash-attn-4, CuTe DSL)")
parser.add_argument("--packing", type=str, default="doc", choices=["concat", "doc"], help="Default doc (~10%% faster, experiments.md 2026-10-14). concat: documents concatenated into 8192 blocks, attention runs across document boundaries. doc: same blocks, but each document attends only to itself and RoPE positions restart at each <|begin_of_text|> (FlashAttention varlen; needs --attn_implementation flash_attention_3). Applies to the evals too")
parser.add_argument("--max_steps", type=int, default=None, help="Override the token-budget step count (e.g. short speed tests)")
parser.add_argument("--num_evals", type=int, default=20, help="Evals over the run (lab: at least every 10%% of the steps)")
parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False, help="torch.compile each decoder layer (not the rotary/mask setup or the Liger loss): fuses RMSNorm/cast/SwiGLU/elementwise kernels. On in run.sbatch/profile.sbatch; --no-torch_compile disables")
parser.add_argument("--profile_dir", type=str, default=None, help="Profile 5 steps with torch.profiler into this dir; works inside a full run (see --profile_start) or in profile.sbatch")
parser.add_argument("--profile_start", type=int, default=8, help="Profile steps profile_start+1 .. profile_start+5 (pick a window without an eval step in a full run, e.g. 300)")
parser.add_argument("--profile_cpu", action=argparse.BooleanOptionalAction, default=True, help="Also record CPU ops (shows what causes GPU gaps, slightly slows launches); --no-profile_cpu = GPU kernels only")
parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True, help="Periodic eval (--no-eval for short speed/profile jobs, see profile.sbatch)")
parser.add_argument("--ddp_find_unused", action=argparse.BooleanOptionalAction, default=False, help="DDP find_unused_parameters. Trainer would set True (no gradient checkpointing), but Llama has no unused params; off is ~1%% faster (experiments.md)")
parser.add_argument("--ddp_bf16_grads", action=argparse.BooleanOptionalAction, default=False, help="All-reduce gradients in bf16 (DDP bf16_compress_hook): half the NCCL traffic, grads stay fp32 locally. No gain with 200 MB buckets")
parser.add_argument("--ddp_bucket_cap_mb", type=int, default=200, help="DDP gradient bucket size; 200 MB vs torch's 25 MB: ~5%% faster on 8 GPUs and ~10 GiB less memory (experiments.md)")
parser.add_argument("--optimizer", type=str, default="muonium", choices=["adamw", "adam_mini", "muon", "muonium", "sophia", "soap"], help="adamw (torch AdamW), adam_mini (Adam-mini, pip adam-mini), muon (Muon for decoder-layer matrices + AdamW for the rest, muon_adamw.py), muonium (pip muonium: Muon for the decoder-layer matrices + its fused AdamW for embedding / norms, one LR = --learning_rate, wd = --weight_decay, package defaults for the rest), sophia (SophiaG, official sophia.py) or soap (official soap.py, work split across GPUs by soap_dist.py); lr = --learning_rate for sophia and soap")
parser.add_argument("--soap_betas", type=str, default="0.95,0.95", help="SOAP beta1,beta2 (official defaults; beta2 also drives the preconditioner EMA). Epsilon: --adam_eps")
parser.add_argument("--soap_precond_freq", type=int, default=10, help="SOAP: re-estimate the preconditioner eigenbases every k steps (official default 10)")
parser.add_argument("--soap_impl", type=str, default="official", choices=["official", "meta"], help="official: soap.py (paper authors) split across GPUs by soap_dist.py. meta: facebookresearch distributed_shampoo with DefaultSOAPConfig (ZeRO-1 over DDP ranks, one AllGather of the updates, blocked preconditioners)")
parser.add_argument("--soap_block", type=int, default=8192, help="--soap_impl meta: max_preconditioner_dim; larger dims are split into blocks (8192: no layer matrix is split, the embedding is; README suggests 1024-8192)")
parser.add_argument("--soap_comm_dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="--soap_impl meta: dtype of the AllGather of the updates")
parser.add_argument("--soap_max_precond_dim", type=int, default=10000, help="SOAP: dims above this are not preconditioned (official 10000: covers 8192, skips the 128k vocab)")
parser.add_argument("--sophia_rho", type=float, default=0.05, help="SophiaG rho (clip threshold; official GPT-2 125M/770M: 0.05). Tune so train/sophia_win_rate stays in 0.1-0.5")
parser.add_argument("--sophia_betas", type=str, default="0.965,0.99", help="SophiaG beta1,beta2 (official defaults)")
parser.add_argument("--sophia_hess_interval", type=int, default=10, help="Re-estimate the diagonal Hessian (Gauss-Newton-Bartlett) every k steps (official: 10); one extra forward + backward of the full batch, i.e. ~1/k extra compute (official: same)")
parser.add_argument("--muon_lr", type=float, default=None, help="--optimizer muon / muonium: peak Muon LR, default 0.01 for muonium (best run), 1.25e-3 for muon (muon: match_rms_adamw scaling, tuned on GPT-2; muonium: x sqrt(max(1, rows/cols)), Keller Jordan's scale ~0.02); --learning_rate is the AdamW / Sophia part's")
parser.add_argument("--muonium_algorithm", type=str, default="normuon", choices=["muon", "normuon"], help="--optimizer muonium: update rule for the matrices (muon, or normuon: per-neuron second-moment normalization, muon_beta2 0.95)")
parser.add_argument("--muonium_rest", type=str, default="adamw", choices=["adamw", "sophia"], help="--optimizer muonium: optimizer for the non-matrix params (tied embedding, norms) at --learning_rate / --weight_decay: muonium's AdamW, or SophiaG (--sophia_* flags, Hessian pass as in --optimizer sophia)")
parser.add_argument("--muon_wd", type=float, default=None, help="--optimizer muonium: weight decay of the Muon matrices (default: --weight_decay)")
parser.add_argument("--muon_momentum", type=float, default=0.95, help="--optimizer muon: Muon Nesterov momentum")
parser.add_argument("--data_dir", type=str, default=os.path.expandvars("/work/$USER/dolma3_llama"), help="Dir with train.bin and val.bin (uint32, prepare_data.py)")
parser.add_argument("--seed", type=int, default=None, help="Seed for the init, data order and Sophia's label sampling (default: torch's fixed default RNG for the init, Trainer seed 42 for the rest; set it for repeat-seed runs)")
parser.add_argument("--run_name", type=str, default=None, help="W&B run name; also isolates checkpoints in results/<run_name>")
parser.add_argument("--save_dir", type=str, default=None, help="Where to save the final model + tokenizer (default: /work/$USER/llama_models/<run_name or latest>)")
args = parser.parse_args()
# Defaults = the best run so far, muonium-normuon0.01-adamw3e-3-ramp-3b (experiments.md 2026-10-16): NorMuon 0.01 for the
# matrices, AdamW 3e-3 for embedding + norms, wd 0.1, WSD 2% / 10%, batch 32 for the first third of the tokens, then 64.
# LRs tuned for muonium don't carry over to other optimizers, which keep their own defaults.
if args.learning_rate is None:
    args.learning_rate = 3e-3 if args.optimizer == "muonium" and args.muonium_rest == "adamw" else 6e-4
if args.muon_lr is None:
    args.muon_lr = 0.01 if args.optimizer == "muonium" else 1.25e-3
if args.batch_ramp is None and args.global_batch_size is None:
    args.batch_ramp = "32:0.3333,64"
if args.batch_ramp and args.batch_ramp.lower() == "none":
    args.batch_ramp = None
if args.global_batch_size is None:
    args.global_batch_size = 64
ARGS = args  # callbacks get the TrainingArguments as `args`
BLOCK_SIZE = args.block_size  # context window
MODEL_NAME = args.model_name
DATA_DIR = os.path.expanduser(args.data_dir)
SAVE_DIR = os.path.expanduser(args.save_dir or os.path.join(os.path.expandvars("/work/$USER/llama_models"), args.run_name or "latest"))

print("Model name:", MODEL_NAME, "LR:", args.learning_rate, "Optimizer:", args.optimizer, *(["Muon LR:", args.muon_lr] if args.optimizer in ("muon", "muonium") else []), "WD:", args.weight_decay, "beta2:", args.beta2, "Run name:", args.run_name, "Packing:", args.packing, "Data dir:", DATA_DIR)


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
# Batch-size ramp: list of (global batch, optimizer steps). Each stage gets its tokens (a count, or a fraction of the
# budget), rounded down to whole steps, so the batch changes at the first step boundary at or before the stage's token
# mark. The LR schedule (warmup / WSD decay) runs over the total optimizer steps, unchanged by the ramp.
RAMP = None
if args.batch_ramp:
    stages = [part.split(":") for part in args.batch_ramp.split(",")]
    sizes = [int(st[0]) for st in stages]
    def stage_tokens(t):  # "1e9", "1B", "300M" -> tokens; <= 1 -> fraction of the budget
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(t[-1].upper(), 1)
        x = float(t[:-1] if mult != 1 else t) * mult
        return x * args.token_budget if x <= 1 else x
    stage_toks = [stage_tokens(st[1]) for st in stages[:-1]]
    stage_toks.append(args.token_budget - sum(stage_toks))
    assert stage_toks[-1] > 0, f"--batch_ramp stages use {sum(stage_toks[:-1]):.3g} tokens, budget {args.token_budget:.3g}"
    # The Trainer's grad accum is fixed: enough micro-batches for the largest batch at --per_device_batch; smaller
    # batches use smaller micro-batches (one static compile per size)
    GRAD_ACCUM = -(-max(sizes) // (WORLD_SIZE * args.per_device_batch))
    assert all(b % (WORLD_SIZE * GRAD_ACCUM) == 0 for b in sizes), args.batch_ramp
    RAMP = [(b, int(t // (b * BLOCK_SIZE))) for b, t in zip(sizes, stage_toks)]
    if args.max_steps:  # short tests: keep the stage proportions in steps
        total = sum(n for _, n in RAMP)
        RAMP = [(b, max(1, round(n * args.max_steps / total))) for b, n in RAMP]
    MAX_STEPS = sum(n for _, n in RAMP)
    PER_DEVICE_BATCH = sizes[0] // (WORLD_SIZE * GRAD_ACCUM)
    print("Batch ramp:", ", ".join(f"{b} x {n} steps = {b * n * BLOCK_SIZE / 1e9:.3g}B tokens ({b // (WORLD_SIZE * GRAD_ACCUM)}/GPU)" for b, n in RAMP),
          f"x grad accum {GRAD_ACCUM}")


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


EVAL_STEPS = max(1, MAX_STEPS // args.num_evals)
if args.eval and EVAL_STEPS > max(1, MAX_STEPS // 10):
    print(f"WARNING: eval every {EVAL_STEPS} steps is less often than the lab's 10% of {MAX_STEPS} steps")
print(f"Token budget {args.token_budget:.3g} -> {MAX_STEPS} steps, {tokens_seen(MAX_STEPS):,} tokens, eval every {EVAL_STEPS}")
if RAMP is None:
    print(f"Global batch {args.global_batch_size} x {BLOCK_SIZE} tokens ({PER_DEVICE_BATCH}/GPU x grad accum {GRAD_ACCUM} x {WORLD_SIZE} GPUs)")
assert tokens_seen(MAX_STEPS) <= 6e9 or args.token_budget > 6e9, "over the 6B-token lab budget"
WARMUP_STEPS = int(args.warmup_frac * MAX_STEPS)
DECAY_STEPS = int(args.decay_frac * MAX_STEPS)
if args.lr_schedule == "wsd2":  # warmup | stable1 | decay1 to r | stable2 at r | decay to 0 (DECAY_STEPS)
    DECAY1_STEPS, STABLE2_STEPS = int(args.decay1_frac * MAX_STEPS), int(args.stable2_frac * MAX_STEPS)
    STABLE1_END = MAX_STEPS - DECAY_STEPS - STABLE2_STEPS - DECAY1_STEPS
    assert STABLE1_END >= WARMUP_STEPS, "wsd2: warmup + decay1 + stable2 + decay exceed the run"
    print(f"wsd2: warmup {WARMUP_STEPS}, stable1 to step {STABLE1_END}, decay1 {DECAY1_STEPS} to {args.stable2_lr_ratio} x peak, "
          f"stable2 {STABLE2_STEPS}, decay {DECAY_STEPS} to 0")

    def wsd2_lambda(step):
        r = args.stable2_lr_ratio
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)  # as HF's get_wsd_schedule
        if step < STABLE1_END:
            return 1.0
        if step < STABLE1_END + DECAY1_STEPS:
            return 1.0 - (1.0 - r) * (step - STABLE1_END) / DECAY1_STEPS
        if step < MAX_STEPS - DECAY_STEPS:
            return r
        return r * max(0.0, (MAX_STEPS - step) / DECAY_STEPS)

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
if args.config_overrides:
    import json
    overrides = json.loads(args.config_overrides)
    if "rope_scaling" in overrides and overrides["rope_scaling"] is None:
        # transformers 5 keeps RoPE in rope_parameters (rope_theta included); None there breaks the model, so
        # "no scaling" = plain RoPE with the same theta
        overrides.pop("rope_scaling")
        overrides["rope_parameters"] = {"rope_type": "default", "rope_theta": config.rope_parameters["rope_theta"]}
    config.update(overrides)
    print("Config overrides:", args.config_overrides)
if args.seed is not None:
    set_seed(args.seed)  # the init draws from the global RNG; without --seed it is torch's fixed default seed
assert args.packing == "concat" or args.attn_implementation.startswith("flash_attention"), "--packing doc needs FlashAttention varlen (--attn_implementation flash_attention_3)"
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

    def fused_ce_forward(input_ids=None, labels=None, num_items_in_batch=None, sample_labels=False, **kwargs):
        if "cu_seq_lens_q" in kwargs:  # --packing doc: the document count varies per batch; one dynamic-shape compile, not one per count
            torch._dynamo.maybe_mark_dynamic(kwargs["cu_seq_lens_q"], 0)
        hidden = model.model(input_ids=input_ids, **kwargs).last_hidden_state
        if sample_labels:  # Sophia's Hessian estimate: mean CE against labels drawn from the model's own predictions
            return CausalLMOutputWithPast(loss=sampled_label_loss(hidden))
        if labels is None:
            return CausalLMOutputWithPast(logits=model.lm_head(hidden))
        hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))  # tokens < n predict n
        targets = labels[:, 1:].reshape(-1)
        if num_items_in_batch is None:
            loss = fused_ce["mean"](model.lm_head.weight, hidden, targets)
        else:
            loss = fused_ce["sum"](model.lm_head.weight, hidden, targets) / num_items_in_batch
        return CausalLMOutputWithPast(loss=loss)

    def sampled_label_loss(hidden, chunk=2048):
        """CE of every position against y ~ softmax(logits) (official train_sophiag.py), without the full logits:
        exact categorical samples by Gumbel-max (argmax of logits - log Exp(1)) chunk by chunk."""
        h = hidden.reshape(-1, hidden.size(-1))
        w = model.lm_head.weight
        with torch.no_grad():
            samples = []
            for c in h.split(chunk):
                logits = (c @ w.t()).float()
                samples.append(logits.sub_(torch.empty_like(logits).exponential_().log_()).argmax(-1))
        return fused_ce["mean"](w, h, torch.cat(samples))

    model.forward = fused_ce_forward

# Compile only the decoder layers: the memory-bound RMSNorm / cast / SwiGLU / elementwise kernels, while the LM head +
# loss stays in Liger's kernels outside the compiled graph (compiling the whole body made GPT-2's attention 2.7x slower).
if args.torch_compile:
    if args.batch_ramp:  # a static recompile per batch size, not slower dynamic-shape kernels after the first change
        torch._dynamo.config.automatic_dynamic_shapes = False
    if args.attn_implementation.startswith("flash_attention"):
        # The FlashAttention Hub kernel is not traceable (graph break at the call), and the frame resumed after it would
        # recompile per layer_idx (static int attribute) and fall back to eager past the recompile limit (8): make
        # it dynamic. Also skip HF's warning_once that it casts autocast's fp32 q/k/v to bf16 (another graph break).
        import logging
        torch._dynamo.config.ignore_logging_functions.add(logging.Logger.warning_once)
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.recompile_limit = 32
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

# --packing doc: each block is split at its <|begin_of_text|> tokens (the block's first segment may be the tail of a
# document cut by the previous block). position_ids restart at 0 per segment (RoPE), and cu_seq_lens (segment starts in
# the micro-batch flattened to one sequence, plus the end) make HF's FlashAttention call the varlen kernel, so attention
# stays inside each document. max_length is the 8192 upper bound (a Python int: no per-batch recompile / host sync).
# Labels are unchanged (an <|end_of_text|> still predicts the next <|begin_of_text|>), so the loss counts the same tokens.
BOS_ID = config.bos_token_id

def doc_packing_collate(features):
    ids = torch.stack([f["input_ids"] for f in features])
    starts = ids == BOS_ID
    starts[:, 0] = True
    first = torch.where(starts, torch.arange(ids.size(1)), 0).cummax(1).values  # index of the segment's first token
    flat_starts = starts.reshape(-1).nonzero().view(-1)
    cu = torch.cat([flat_starts, torch.tensor([ids.numel()])]).to(torch.int32)
    return {"input_ids": ids, "labels": ids.clone(), "position_ids": torch.arange(ids.size(1)) - first,
            "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": BLOCK_SIZE, "max_length_k": BLOCK_SIZE}

train_ds = TokenBlocks(os.path.join(DATA_DIR, "train.bin"), BLOCK_SIZE)
val_ds = TokenBlocks(os.path.join(DATA_DIR, "val.bin"), BLOCK_SIZE)
# Our own dev set (select_rows.py dirs): evaluated next to val.bin, logged as eval/dev_loss, eval/dev_perplexity
dev_ds = TokenBlocks(os.path.join(DATA_DIR, "dev.bin"), BLOCK_SIZE) if os.path.exists(os.path.join(DATA_DIR, "dev.bin")) else None
print(f"Train: {len(train_ds):,} blocks ({len(train_ds) * BLOCK_SIZE / 1e9:.2f}B tokens), val: {len(val_ds):,} blocks"
      + (f", dev: {len(dev_ds):,} blocks" if dev_ds is not None else ""))
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

    # DDP (see --ddp_* flags; the bf16 comm hook is set in CustomTrainer)
    ddp_find_unused_parameters=args.ddp_find_unused,
    ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,

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
    seed=42 if args.seed is None else args.seed,  # data order (and RNG after Trainer init)
)
print("=== Training arguments: ", training_args)

# Runs are bounded by the token budget: MAX_STEPS = budget / tokens per step, and training ends after MAX_STEPS. The job
# time limit is only a safety net: stop early so the model is still saved before Slurm kills the job (slow node, or
# a --time too short for the budget). Such a run saw fewer tokens and its LR decay was cut short, so it is not
# comparable to others. It warns as early as step 50 if the projected finish is past the deadline, and records
# budget_completed in the W&B summary (see the end of the script).
class DeadlineCallback(TrainerCallback):
    PROJECT_AT = 50  # compile and warm-up are done by then

    def __init__(self, deadline):
        self.deadline = deadline
        self.t_project = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.PROJECT_AT // 2:
            self.t_project = time.time()
        elif state.global_step == self.PROJECT_AT and self.t_project and state.is_world_process_zero:
            per_step = (time.time() - self.t_project) / (self.PROJECT_AT - self.PROJECT_AT // 2)  # includes evals in that window
            finish = time.time() + per_step * (state.max_steps - state.global_step)
            if finish > self.deadline:
                print(f"=== WARNING: projected finish {(finish - START_TIME) / 60:.0f} min after start is past the deadline "
                      f"({(self.deadline - START_TIME) / 60:.0f} min): the run will stop before the token budget. Raise --time.")
            else:
                print(f"=== Projected finish {(finish - START_TIME) / 60:.0f} min after start (deadline {(self.deadline - START_TIME) / 60:.0f} min), plus evals")
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
    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        if state.is_world_process_zero and wandb.run is not None:
            wandb.define_metric("train/total_tokens_seen")
            # WandbCallback logs TrainingArguments and the model config; add every train.py flag (W&B config "args.*")
            # and the derived run shape ("derived.*"), so runs can be filtered / compared by them
            wandb.config.update({"args": vars(ARGS), "derived": {
                "max_steps": MAX_STEPS, "warmup_steps": WARMUP_STEPS, "decay_steps": DECAY_STEPS, "eval_steps": EVAL_STEPS,
                "tokens": tokens_seen(MAX_STEPS), "tokens_per_step": None if RAMP else batch_at(1) * BLOCK_SIZE,
                "batch_ramp_steps": RAMP, "world_size": WORLD_SIZE, "per_device_batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM,
                "block_size": BLOCK_SIZE, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")}},
                allow_val_change=True)
            for key in TOKEN_AXIS_METRICS:  # explicit names take precedence over WandbCallback's "*" -> global_step
                wandb.define_metric(key, step_metric="train/total_tokens_seen")

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
            mean = sum(t) / len(t)  # differs from the median when some steps do extra work (Sophia's Hessian pass)
            print(f"=== Step time: median {med:.3f} s, mean {mean:.3f} s over {len(t)} steps (excl. eval), "
                  f"{BLOCK_SIZE * batch_at(state.global_step) / med:,.0f} tokens/s at the median, "
                  f"{BLOCK_SIZE * batch_at(state.global_step) / mean:,.0f} at the mean")

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

USES_SOPHIA = args.optimizer == "sophia" or (args.optimizer == "muonium" and args.muonium_rest == "sophia")
if USES_SOPHIA:
    assert args.fused_ce, "SophiaG samples labels inside the fused CE forward; use --fused_ce"

    def sophia_bs(step):  # official: step(bs=total_bs * block_size), tokens per step; follows --batch_ramp
        return batch_at(step) * BLOCK_SIZE

    class SophiaGTrainer(SophiaG):
        """The HF Trainer calls optimizer.step() without SophiaG's bs argument."""
        t = 0  # optimizer steps taken (1-based step of this call)

        def step(self, closure=None, bs=None):
            self.t += 1
            return super().step(closure, bs=sophia_bs(self.t))

    def build_sophia(params):
        # Same decay split as the HF Trainer's AdamW (matrices + embedding; not the RMSNorm weights)
        opt = SophiaGTrainer(
            [{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
            lr=args.learning_rate, betas=tuple(float(b) for b in args.sophia_betas.split(",")), rho=args.sophia_rho,
            weight_decay=args.weight_decay,
        )
        print(f"SophiaG: lr {args.learning_rate}, rho {args.sophia_rho}, betas {args.sophia_betas}, wd {args.weight_decay}, "
              f"bs {sophia_bs(1):,}{f' -> {sophia_bs(MAX_STEPS):,}' if RAMP else ''} tokens, Hessian every {args.sophia_hess_interval} steps")
        return opt


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


sophia_opt = None  # the SophiaG instance the Hessian callback updates

if args.optimizer == "muonium":
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
    muon_group = {"params": muon_p, "algorithm": args.muonium_algorithm}
    if args.muonium_rest == "adamw":
        optimizer = Muon([muon_group, {"params": rest_p, "algorithm": "adamw", "lr": args.learning_rate, "wd": args.weight_decay}],
                         lr=args.muon_lr, wd=muon_wd)
    else:
        sophia_opt = build_sophia(rest_p)
        optimizer = ChainedOptimizer(Muon([muon_group], lr=args.muon_lr, wd=muon_wd), sophia_opt)
    print(f"muonium {args.muonium_algorithm}: {len(muon_p)} tensors, {sum(p.numel() for p in muon_p) / 1e6:.1f}M params, "
          f"lr {args.muon_lr}, wd {muon_wd}; {args.muonium_rest}: {len(rest_p)} tensors, "
          f"{sum(p.numel() for p in rest_p) / 1e6:.1f}M params, lr {args.learning_rate}, wd {args.weight_decay}")

elif args.optimizer == "sophia":
    sophia_opt = optimizer = build_sophia([p for p in model.parameters() if p.requires_grad])

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
         # RMSNorm weights: plain Adam, as official SOAP (precondition_1d=False), no weight decay
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0, "start_preconditioning_step": math.inf,
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
    # Same decay split as the HF Trainer's AdamW (matrices + embedding; not the RMSNorm weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = DistributedSOAP(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=args.learning_rate, betas=tuple(float(b) for b in args.soap_betas.split(",")), eps=args.adam_eps,
        weight_decay=args.weight_decay, precondition_frequency=args.soap_precond_freq,
        max_precond_dim=args.soap_max_precond_dim,
    )
    print(f"SOAP: lr {args.learning_rate}, betas {args.soap_betas}, eps {args.adam_eps}, wd {args.weight_decay}, "
          f"precondition every {args.soap_precond_freq} steps, max dim {args.soap_max_precond_dim}")

# Sophia's Hessian estimate, as in the official train_sophiag.py: every k steps, after the optimizer step, forward +
# backward the step's batch with labels sampled from the model, clip, and EMA the squared gradient into the Hessian.
# Reuses the step's own micro-batches (official: the next batches) so no extra data tokens are consumed. Gradients
# are averaged over all GPUs through DDP, so every rank keeps the same Hessian. Also logs the "win rate" (fraction of
# coordinates whose update is not clipped, official train/win_rate; should stay ~0.1-0.5, else tune rho).
sophia_stats = {}

class SophiaHessianCallback(TrainerCallback):
    def on_step_end(self, args_, state, control, **kwargs):
        if state.global_step % args.sophia_hess_interval:
            return
        ddp_model, batches = trainer.model_wrapped, trainer.step_batches
        for i, inputs in enumerate(batches):
            sync = contextlib.nullcontext() if i == len(batches) - 1 or ddp_model is model else ddp_model.no_sync()
            with sync, trainer.compute_loss_context_manager():
                loss = ddp_model(**{k: v for k, v in inputs.items() if k != "labels"}, sample_labels=True).loss / len(batches)
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args_.max_grad_norm)
        sophia_opt.update_hessian()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            unclipped = sum((st["exp_avg"].abs() < args.sophia_rho * sophia_bs(state.global_step) * st["hessian"]).sum()
                            for st in sophia_opt.state.values())
            sophia_stats["sophia_win_rate"] = (unclipped / sum(st["exp_avg"].numel() for st in sophia_opt.state.values())).item()

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
    step_batches, _batches_step = [], -1

    def training_step(self, model, inputs, num_items_in_batch=None):
        if USES_SOPHIA:  # keep this optimizer step's micro-batches for the Hessian estimate
            if self._batches_step != self.state.global_step:
                self.step_batches, self._batches_step = [], self.state.global_step
            self.step_batches.append(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if args.ddp_bf16_grads and self.accelerator.ddp_handler is not None:
            from accelerate.utils import DDPCommunicationHookType
            self.accelerator.ddp_handler.comm_hook = DDPCommunicationHookType.BF16  # registered when DDP wraps the model

    def create_scheduler(self, num_training_steps, optimizer=None):
        if args.lr_schedule == "wsd2" and self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, wsd2_lambda)
            self._created_lr_scheduler = True
        return super().create_scheduler(num_training_steps, optimizer)

    def get_train_dataloader(self):
        if RAMP is None:
            return super().get_train_dataloader()
        sampler = RampBatchSampler(len(self.train_dataset), RAMP, WORLD_SIZE, self.args.seed, GRAD_ACCUM)
        return self.accelerator.prepare(DataLoader(self.train_dataset, batch_sampler=sampler,
                                                   collate_fn=self.data_collator, pin_memory=True))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if dev_ds is not None and eval_dataset is None and metric_key_prefix == "eval":
            metrics.update(super().evaluate(dev_ds, ignore_keys, "eval_dev"))
        return metrics

    # Lab metrics. WandbCallback prefixes train logs with "train/" and "eval_*" with "eval/": loss (mean over the
    # logging interval), grad_norm and learning_rate come from the Trainer; tokens_per_second (of the logging step
    # itself), total_tokens_seen and eval_perplexity are added here.
    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)
            if step_timer.times:
                logs["tokens_per_second"], logs["mfu"] = step_timer.last_step_throughput()
            logs.update(sophia_stats)
        if "eval_loss" in logs:
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)  # x-axis for eval/perplexity vs tokens
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        if "eval_dev_loss" in logs:
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)
            logs["eval_dev_perplexity"] = math.exp(logs["eval_dev_loss"])
        super().log(logs, *args, **kwargs)


time_left = slurm_time_left()
deadline = START_TIME + time_left - 300 if time_left else float("inf")  # leave 5 min for the final eval + save
print("Deadline:", f"{(deadline - START_TIME) / 3600:.2f} h from start" if time_left else "none")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=doc_packing_collate if args.packing == "doc" else None,
    # Sophia's Hessian pass before step_timer, so its time counts in tokens_per_second / MFU
    callbacks=([SophiaHessianCallback()] if USES_SOPHIA else [])
    + [DeadlineCallback(deadline), step_timer, WandbTokenAxisCallback()]
    + ([ProfilerCallback(args.profile_dir, args.profile_start, args.profile_cpu)] if args.profile_dir else []),
    optimizers=(optimizer, None),  # None -> Trainer builds AdamW from args; scheduler always from args
)

# Train
print("=== Starting training...")
trainer.train()
if args.eval and trainer.state.global_step % EVAL_STEPS:  # final eval unless the last step just ran one
    trainer.evaluate()
BUDGET_COMPLETED = trainer.state.global_step >= MAX_STEPS
print(f"=== Trained on {tokens_seen(trainer.state.global_step):,} tokens in {trainer.state.global_step}/{MAX_STEPS} steps"
      + ("" if BUDGET_COMPLETED else " -- WARNING: token budget NOT completed (deadline)"))
if trainer.is_world_process_zero():
    import wandb
    if wandb.run is not None:  # filter / check runs by this in W&B before comparing them
        wandb.run.summary.update({"budget_completed": BUDGET_COMPLETED, "final_tokens_seen": tokens_seen(trainer.state.global_step)})
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
