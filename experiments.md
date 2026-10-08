# Experiments

W&B project: `cerulean-labs/lab5-training-llama` on `https://app.forge.coreweave.com`.

## 2026-10-08 — Throughput tuning with the existing flags (Llama 3.2 1B, 8192 context)

**Question:** How much speed do the existing flags still give for Llama 3.2 1B at 8192 tokens? The GPT-2 log suggests
looking at micro-batch size, attention backend (attention is ~30% of FLOPs here vs ~10% for GPT-2), compile, and CPUs/NUMA.

Setup: `profile.sbatch` (30 steps, median over 25, profiler on steps 9–13, eval off), AdamW, `--fused_ce --torch_compile`,
toy data (one Dolma 3 shard tiled; content doesn't change speed). 2-GPU jobs use global batch 16, which matches the
8-GPU default's 4/GPU × accum 2.

### Profile of the default (job 511900, 2 GPUs)

GPU busy ~100%, MFU 56%. GPU time per step: matmuls ~54%, **cuDNN flash attention 27%** (backward 1.25 s vs forward
0.46 s per 5 steps), NCCL all-reduce 3%, compiled elementwise / casts a few %. Little memory-bound work is left
to fuse.

### Variants (2 GPUs)

| Variant | Job | Median step | Tokens/s/GPU | vs default | Peak reserved | Step 1 |
|---|---|---|---|---|---|---|
| **Default** (SDPA, 4/GPU × accum 2, compile) | 511987 | 1.249 s | **52.5k** | — | 80 GiB | 12.6 s |
| `--attn_implementation=flash_attention_3` | 511988 | 1.236 s | 53.0k | +1.0% | 82 GiB | 15.7 s |
| `--attn_implementation=flash_attention_4` | 511989 | 1.499 s | 43.7k | −17% | 82 GiB | 175 s |
| `--per_device_batch=2` (accum 4) | 511990 | 1.332 s | 49.2k | −6% | 54 GiB | 11.9 s |
| `--per_device_batch=8` (no accum) | 511991 | 1.229 s | 53.3k | +1.6% | 129 GiB | 11.7 s |
| `--no-torch_compile` | 511992 | 1.529 s | 42.9k | −18% | 94 GiB | 3.3 s |

### 8 GPUs (global batch 64 = 0.5M tokens)

| Variant | Job | Median step | Tokens/s (total / per GPU) | All-reduce per 5 steps | GPU busy | Peak reserved |
|---|---|---|---|---|---|---|
| **Default** (4/GPU × accum 2) | 512020 | 1.277 s | **410k / 51.3k** | 1.49 s | 121% (overlapped) | 80 GiB |
| `--per_device_batch=8` | 512021 | 1.261 s | 416k / 52.0k (+1.3%) | 0.69 s | 109% | 129 GiB |

### Findings

- **Little left to gain from the existing flags:** every variant that keeps the setup is within ±2% of the default.
  Compile is worth +22% (keep). FA3 is level with cuDNN (+1%, noise). FA4 is 17% slower on the H200, with a
  3-minute compile.
- 8 GPUs scale well: 51.3k vs 52.5k tokens/s/GPU (−2%). With accum 2 the all-reduce traffic doubles (1.49 vs 0.69 s
  per 5 steps; HF Trainer seems to sync every micro-batch), but it overlaps the backward pass.
- 8/GPU is +1.3% but uses 129 of 140 GiB, so any extra memory would OOM. Kept 4/GPU.
- Throughput for planning: **~410k tokens/s on 8 GPUs → 6B tokens in ~4.1 h (~33 H200-hours)**, plus evals.
- Larger gains need code changes, not flags: skip the duplicate all-reduce under grad accum, bf16 gradients or
  all-reduce, Liger's RMSNorm/SwiGLU/RoPE kernels instead of compile, document masking (changes the math), FP8.

## 2026-10-08 — First real run: Llama 2/3 default hyperparameters, 1 h on 8 H200

**Question:** Where does a plain Llama-recipe AdamW run get in ~1 h of 8 GPUs, as a reference point for tuning?

Data: `/work/$USER/dolma3_llama` (prep job 511899, 53 min): 104,020,922 docs, OJ split reproduced (last 50k of the
seed-42 shuffle held out), train = first 5% of shuffled train docs = 5,198,546 docs / **7.86B tokens**, val.bin = the first
10k held-out docs = 15.2M tokens (1,850 blocks of 8192).

Hyperparameters (Llama 2 / 3 recipe where it applies): AdamW β=(0.9, 0.95), ε=1e-5, weight decay 0.1, grad clip 1.0,
peak LR 3e-4, cosine to 10% of the peak (`--lr_schedule=cosine`). Two values had to be scaled down for a ~2k-step run:
global batch 0.5M tokens (64 × 8192; Llama used 4M) and warmup 2% = 41 steps (Llama: 2000 steps). Speed setup = defaults
(fused CE, compile, SDPA, 4/GPU × accum 2).

```
sbatch --time=1:00:00 --job-name=llama-default-1h run.sbatch --run_name=llama-default-1h-8gpu --token_budget=1.1e9 \
    --lr_schedule=cosine --min_lr_ratio=0.1 --learning_rate=3e-4 --adam_eps=1e-5 --weight_decay=0.1 --beta2=0.95 --warmup_frac=0.02
```

| Job | W&B run | Steps | Tokens | Final eval ppl | Final eval loss | Final train loss | Tokens/s | MFU | Job time |
|---|---|---|---|---|---|---|---|---|---|
| 512100 | `dvqj42sv` | 2098 | 1.10B | **50.68** | 3.926 | 3.731 (mean of last 10 steps) | 400k | 54% | 51.6 min (6.9 GPU-h) |

Eval perplexity by tokens seen: 55M: 970 · 164M: 310 · 273M: 171 · 382M: 105 · 491M: 79.6 · 600M: 66.6 · 709M: 59.3 ·
818M: 55.2 · 927M: 53.3 · 1.04B: 51.3 · 1.10B: **50.7**.

### Findings

- Stable: no loss spikes. Max grad norm 8.7 at step 10 (warmup), 0.31 at the end.
- Steady speed: median step 1.310 s (400k tokens/s), 2.5% slower than the 30-step profile (1.277 s). 21 evals and startup
  fit easily: the job finished at 51.6 min with no deadline stop.
- The cosine tail flattens the curve: from 0.82B to 1.10B tokens, ppl improves only 55.2 → 50.7 while the LR falls to 3e-5.
  The GPT-2 sweeps found WSD and a higher peak LR (5× the paper value) better; both are untested here.
- Train loss (3.73) is well below eval loss (3.93) after < 0.14 epoch, so this is not overfitting. The 10k held-out docs
  likely differ in mix from a 10-step window, and eval packs docs into 8192 blocks without masking across docs.

### Caveats

- The eval ppl is on packed 8192-token blocks of 10k held-out docs. The OJ probably scores each doc separately, so its
  number may differ; `eval_docs.jsonl` has all 50k held-out texts for checking that.
- One seed.

### Next

- Peak LR scan (e.g. 6e-4, 1.2e-3) and WSD vs cosine at this ~1 h scale.
- For the ppl-20 baseline: the full budget is ~5.5× the tokens of this run.
