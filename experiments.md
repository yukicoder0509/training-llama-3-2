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
- 8 GPUs scale well: 51.3k vs 52.5k tokens/s/GPU (−2%). With accum 2 the NCCL kernel time doubles (1.49 vs 0.69 s
  per 5 steps), but it overlaps the backward pass. (Not a duplicate sync, as first guessed: see the DDP entry below.)
- 8/GPU is +1.3% but uses 129 of 140 GiB, so any extra memory would OOM. Kept 4/GPU.
- Throughput for planning: **~410k tokens/s on 8 GPUs → 6B tokens in ~4.1 h (~33 H200-hours)**, plus evals.
- Larger gains need code changes, not flags: DDP settings (done, see the DDP entry below: +6%), bf16 gradients or
  all-reduce (no gain there), Liger's RMSNorm/SwiGLU/RoPE kernels instead of compile, document masking (changes the math), FP8.

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

## 2026-10-08 — DDP settings: find_unused_parameters, bucket size, bf16 all-reduce (8 H200)

**Question:** The flag sweep left communication as the only lead (8 GPUs were 2% slower per GPU than 2). Does
the suspected duplicate gradient sync under grad accum exist, and do bf16 gradients help?

**Duplicate sync: no.** The profiler traces of jobs 512020 (4/GPU × accum 2) and 512021 (8/GPU, no accum) have the same
all-reduces: 335 per 5 steps, ~1.24B elements per step (one copy of the gradients). Trainer already wraps the
first micro-batch in `no_sync`. With accum, the kernels take longer only because they run during the second micro-batch's
compute. Every large all-reduce overlaps compute, and compute streams are idle only 2–3% of the time on both 2 and 8 GPUs.
The real cost is contention: compute kernels take ~2% longer on 8 GPUs (6.23 vs 6.11 s per 5 steps).

**Found in the Trainer code:** without gradient checkpointing, Trainer sets DDP `find_unused_parameters=True`. That means
a graph walk every forward plus extra reducer work, though Llama has no unused parameters. New flags in `train.py`:
`--ddp_find_unused`, `--ddp_bucket_cap_mb` (torch default 25 MB), and `--ddp_bf16_grads` (DDP `bf16_compress_hook`, via the
Accelerate DDP handler).

Setup: `profile.sbatch` on 8 GPUs, 40 steps (median over 35), toy data, AdamW, 4/GPU × accum 2 (global batch 64), dev partition.

| Variant | Job | Median step | Tokens/s | vs baseline | Peak allocated |
|---|---|---|---|---|---|
| Baseline (find_unused on, 25 MB, fp32) | 514601 / 514624 | 1.280 / 1.295 s | 409.7k / 404.8k (mean 407k) | — | 79.2 GiB |
| `--no-ddp_find_unused` | 514602 | 1.269 s | 413.2k | +1.5% | 79.2 GiB |
| `--ddp_bf16_grads` | 514603 | 1.269 s | 413.0k | +1.4% | 80.0 GiB |
| `--ddp_bucket_cap_mb=100` | 514604 | 1.233 s | 425.1k | +4.4% | 69.2 GiB |
| no find_unused + 100 MB | 514622 | 1.218 s | 430.3k | +5.7% | 69.2 GiB |
| no find_unused + 100 MB + bf16 | 514606 | 1.221 s | 429.3k | +5.5% | 70.1 GiB |
| **no find_unused + 200 MB** | 514620 | 1.213 s | **432.4k** | **+6.2%** | 69.2 GiB |
| no find_unused + 200 MB + bf16 | 514623 | 1.211 s | 432.9k | +6.3% | 70.0 GiB |
| no find_unused + 500 MB | 514621 | 1.217 s | 430.8k | +5.8% | 69.2 GiB |
| New defaults (re-run) | 514647 | 1.214 s | 431.7k | +6.0% | 69.2 GiB |
| New defaults, `--per_device_batch=8` | 514648 | 1.211 s | 433.1k | +6.4% | 106.9 GiB |

### Findings

- **New defaults: `--no-ddp_find_unused --ddp_bucket_cap_mb=200`, +6% (407k → 432k tokens/s, 54.0k/GPU)**. That is now
  faster per GPU than the old 2-GPU default (52.5k). Most of the gain is bucket size: 335 → 175 NCCL kernels per
  5 steps (100 MB). The bucket size is flat from 100 to 500 MB.
- Bigger buckets also cut peak memory by 10 GiB (79 → 69 GiB). This was not investigated further.
- bf16 all-reduce: +1.4% with 25 MB buckets, no gain with large ones. It also reduces gradient precision across ranks.
  Kept off (flag available).
- Per-device batch 8 now fits easily (107 GiB) but adds only +0.3%. Kept 4.
- Planning: 6B tokens on 8 GPUs ≈ 3.9 h (~31 H200-hours), down from 4.1 h.
- The WSD 1 h run (job 514592) was started before this change, with the old DDP settings, so it is comparable to the cosine run.

## 2026-10-08 — WSD vs cosine at the 1 h scale (8 H200)

**Question:** The GPT-2 sweeps preferred WSD to cosine. Does that hold for the Llama 1 h run?

Same settings as the cosine run (job 512100) except the schedule: peak 3e-4, warmup 2%, constant, then linear decay
to 0 over the final 20% (`--lr_schedule=wsd --decay_frac=0.2`). Same data order (seed). Run with the old DDP settings,
so the speed is comparable too.

```
sbatch --time=1:00:00 --job-name=llama-wsd-1h run.sbatch --run_name=llama-wsd-1h-8gpu --token_budget=1.1e9 --lr_schedule=wsd \
    --decay_frac=0.2 --learning_rate=3e-4 --adam_eps=1e-5 --weight_decay=0.1 --beta2=0.95 --warmup_frac=0.02
```

| Schedule | Job | W&B run | Final eval ppl | Final eval loss | Train loss (last 100 steps) | Tokens/s | Job time |
|---|---|---|---|---|---|---|---|
| Cosine to 10% | 512100 | `dvqj42sv` | 50.68 | 3.926 | 3.73 | 400k | 51.6 min |
| **WSD, 20% linear decay** | 514592 | `dkyqsbe6` | **38.00** | **3.637** | 3.48 | 397k | 52.1 min |

Eval ppl by tokens seen (WSD / cosine): 273M: 172 / 171 · 491M: 73.9 / 79.6 · 709M: 50.8 / 59.3 · 818M: 46.0 / 55.2 ·
927M (decay under way): 42.4 / 53.3 · 1.10B: **38.0 / 50.7**.

### Findings

- **WSD is much better: −0.29 nats eval loss (ppl 50.7 → 38.0)** at the same tokens and cost.
- WSD is already ahead before its decay begins at 880M (46.0 vs 55.2 ppl at 818M). Its LR is still at the 3e-4 peak while
  cosine has dropped to ~7e-5. So keeping the LR high helps, which suggests 3e-4 is too low a peak for this model and
  batch. From 818M to the end, the decay plus the extra tokens give WSD a further −0.19 nats (ppl 46 → 38).
- Next: a peak LR scan with WSD (6e-4, 1.2e-3; GPT-2 preferred ~5× the paper LR). Also try a decay fraction of 0.1–0.3.

## 2026-10-08 — Sophia (SophiaG) at the 1 h scale, 0.5M-token batch (8 H200)

**Question:** Does the second-order Sophia beat AdamW at a fixed 1.1B-token budget?

Implementation: `sophia.py` is the official SophiaG (Liuhong99/Sophia @ a7e1572), vendored unchanged. Neither HF nor torch
has Sophia, and the PyPI `sophia-opt` is a third-party repackaging. `train.py --optimizer sophia` follows the official
`train_sophiag.py`:
- `step(bs = tokens per step)`.
- Every 10 steps, a Hessian pass after the optimizer step. It forwards + backwards the step's own micro-batches with labels
  sampled from the model, clips to 1.0, then calls `update_hessian()`. This reuses the step's batch, so no extra data
  tokens are used (official: the next batches). Labels are sampled inside the Liger fused CE by Gumbel-max, chunk by
  chunk (no full logits). DDP averages the gradients, so all ranks keep the same Hessian.
- The official GPT-2 values: ρ 0.05, betas (0.965, 0.99), weight decay 0.2 (2× AdamW).
- Same WSD schedule, warmup 2%, budget and data order as the AdamW-WSD run (514592).
- `train/sophia_win_rate` = fraction of coordinates not clipped (official `train/win_rate`, target 0.1–0.5).
- Cost: the Hessian pass is one extra full forward + backward every 10 steps, +10% compute. Mean step 1.404 s vs median
  1.263 s.

| Optimizer | LR | Job | W&B | Eval loss @ 164M / 273M / 491M / 709M / 818M | Final eval loss (ppl) | Train loss (last 100 steps) |
|---|---|---|---|---|---|---|
| AdamW (WSD) | 3e-4 | 514592 | `dkyqsbe6` | 5.746 / 5.148 / 4.303 / 3.929 / 3.829 | 3.637 (38.0) | 3.48 |
| Sophia | 6e-4 | 514770 | `xxsd6mx6` | 7.45 / 6.94 / 6.75 (430M) → 6.93 (654M), **cancelled** | — | — |
| Sophia | 3e-4 | 514769 | `5els0cbx` | 6.84 / 6.40 / 6.34 (430M) → 6.45 (654M), **cancelled** | — | — |
| **Sophia** | **1e-4** | 514819 | `wq8wp81b` | 6.156 / 5.448 / 4.275 / 3.782 / 3.661 | **3.394 (29.8)** | 3.25 |

### Findings

- **Sophia at LR 1e-4 beats AdamW by 0.24 nats (ppl 38.0 → 29.8)** at the same 1.1B tokens. Cost: 7.3 vs 6.9 GPU-h.
- It is slower at first (behind until ~450M tokens), then pulls ahead and the gap keeps growing through the decay.
- The LR rule from the official README ("about the AdamW LR") failed here. At 3e-4 and 6e-4 the loss stalled at ~6.3–6.9
  and then rose, so both runs were stopped at 654M tokens. Higher LR was worse.
  - Why: the Hessian EMA starts at 0 with no bias correction and only gets an update every 10 steps. So for the first few
    hundred steps nearly all coordinates are clipped (win rate 0.02 at 30M tokens) and get full-LR sign steps, Lion-like.
  - A 3e-4 sign step is far too large (Lion LRs are ~AdamW/3–10). The official GPT-2 runs hid this with 2,000 warmup
    steps, vs our 41.
- Win rate at 1e-4: 0.14 at ~150M tokens, rising slowly to 0.20. At 3e-4 / 6e-4 it was 0.36–0.54, but those runs had
  already stalled.
- Caveats: one seed. AdamW's LR (3e-4) is itself untuned, and Sophia's ρ is untuned.

## 2026-10-08 — SOAP implementation: official soap.py split across GPUs vs Meta's distributed_shampoo

Neither HF nor torch has SOAP. There are two trustworthy implementations, both now in the repo:

- `--soap_impl official` (default): `soap.py`, the paper authors' implementation (nikhilvyas/SOAP @ a1e5535),
  vendored unchanged. It is single-GPU only. Run as is under DDP, it would repeat ~90 TFLOP of fp32 preconditioner
  matmuls per step and ~30 GB of state (8192×8192 for each MLP matrix) on every GPU.
  - `soap_dist.py` splits the work. After DDP's all-reduce the gradients are identical on all ranks, so each tensor is
    updated only by one owner rank (assigned greedily by cost), which broadcasts the result.
  - Checked: identical to the replicated SOAP on a 2-process CPU test (max difference 0.0).
- `--soap_impl meta`: facebookresearch/optimizers `distributed_shampoo` @ d24a149 (BSD, authors of the Distributed
  Shampoo paper), with `DefaultSOAPConfig`.
  - ZeRO-1 over the DDP ranks, then one AllGather of the updates.
  - Blocked preconditioners (`--soap_block`); the RMSNorm weights use plain Adam, as in official SOAP.
  - Difference: with block ≤ 8192 the 128k×2048 embedding is preconditioned in blocks on both sides, while official SOAP
    skips its 128k dim.

Speed (8 H200, 40 steps, toy data, 4/GPU × accum 2 = 0.5M tokens). The QR eigenbasis update every 10 steps makes the mean
higher than the median:

| Implementation | Job | Median / mean step | Mean tokens/s | Peak alloc. |
|---|---|---|---|---|
| AdamW (reference) | 514647 | 1.214 s | 432k | 69 GiB |
| **official soap.py + soap_dist.py** | 514977 | 1.481 / **1.596 s** | 329k (−24%) | 65 GiB |
| meta, block 8192, fp32 AllGather | 515035 | 1.478 / 1.679 s | 312k | 70 GiB |
| meta, block 8192, bf16 AllGather | 515036 | 1.460 / 1.663 s | 315k | 67 GiB |
| meta, block 2048 | 515037 | 1.348 / 1.462 s | 359k | 68 GiB |

- At the same preconditioner (no blocking of the layer matrices), Meta's version is not faster than ours. Our per-tensor
  split already balances the work (relative load 0.97–1.0 per rank). The bf16 AllGather saves only 1%.
- Block 2048 is 9% faster, but it is a block-diagonal approximation (a different algorithm). Untested for quality.
- The overhead is per optimizer step, so it shrinks with batch size: ~0.38 s on top of ~1.2 s at 0.5M (+31%), but only
  ~8% at 2M.
- The batch-size sweep uses the official algorithm (`--soap_impl official`).

## 2026-10-08 — Batch size sweep: AdamW vs SOAP vs Sophia at 0.5M / 1M / 2M tokens (8 H200, 1.1B tokens)

**Question:** Second-order optimizers are expected to gain more at larger batches (SOAP README). How do AdamW, SOAP
and Sophia compare at a fixed 1.1B-token budget as the batch grows?

Setup:
- Every run sees exactly 1.1B tokens (`budget_completed` true for all; ended by the token budget, `--time=1:30`).
- WSD, warmup 2% + decay 20% of the steps (the same token span at every batch size). Same data order.
- Batch 64 / 128 / 256 sequences of 8192 = 0.5M / 1M / 2M tokens → 2098 / 1049 / 524 steps.
  Per GPU always 4 × accum 2 / 4 / 8.
- LR scaled with sqrt(batch): ×1 / ×1.41 / ×2. Bases:
  - AdamW 3e-4 (β 0.9/0.95, ε 1e-5, wd 0.1).
  - SOAP 3e-4 (official soap.py + soap_dist.py, β 0.95/0.95, ε 1e-5, wd 0.1, precondition every 10 steps).
  - Sophia 1e-4 (ρ 0.05, β 0.965/0.99, wd 0.2, Hessian every 10 steps).
- The 0.5M AdamW and Sophia runs are the earlier ones (514592, 514819). 514592 used the old DDP settings, which change
  the speed but not the math.

| Optimizer | 0.5M (LR) | 1M (LR) | 2M (LR) |
|---|---|---|---|
| AdamW | 3.637 (3e-4) `dkyqsbe6` | 3.871 (4.24e-4) `s9efz8hg` | 4.433 (6e-4) `4zajrdxb` |
| SOAP | 3.604 (3e-4) `k6ow93ln` | 3.808 (4.24e-4) `2vdor46s` | 4.202 (6e-4) `4kpkd7o8` |
| Sophia | **3.394** (1e-4) `wq8wp81b` | 4.610 (1.41e-4) `xh20k60c` | 5.936 (2e-4) `r6edh4pt` |

Final eval loss (ppl): AdamW 38.0 / 48.0 / 84.2, SOAP 36.7 / 45.0 / 66.8, Sophia 29.8 / 100.5 / 378.

Speed (mean step incl. the periodic extra work) and cost:

| Optimizer | 0.5M | 1M | 2M |
|---|---|---|---|
| AdamW | 1.21 s* (432k tok/s), 6.9 GPU-h | 2.44 s (430k), 6.4 GPU-h | 4.84 s (433k), 6.4 GPU-h |
| SOAP | 1.63 s (322k), 8.4 GPU-h | 2.83 s (370k), 7.4 GPU-h | 5.19 s (404k), 6.8 GPU-h |
| Sophia | 1.40 s (373k), 7.3 GPU-h | 2.71 s (387k), 7.1 GPU-h | 5.35 s (392k), 7.0 GPU-h |

\* With the new DDP settings (514647). The 0.5M AdamW run itself (old settings) ran at 1.32 s.

Plots: `plots/bs_sweep_final.png` (final loss vs batch), `plots/bs_sweep_curves.png` (train/eval loss vs tokens).

### Findings

- **At a fixed token budget, every optimizer gets worse as the batch grows.** 1.1B tokens is far from the regime where
  big batches pay off: 4× the batch means 4× fewer updates (524 at 2M). Best overall: Sophia at 0.5M.
- **SOAP's advantage over AdamW grows with the batch, as the paper predicts:** −0.03 / −0.06 / −0.23 nats at
  0.5M / 1M / 2M. Its per-step overhead also amortizes: +31% time at 0.5M, +7% at 2M. At 2M, SOAP costs 6% more
  GPU-hours for 0.23 nats lower loss.
- **Sophia falls apart at larger batches:** +0.74 nats worse than AdamW at 1M and +1.50 at 2M, after being −0.24 better at
  0.5M. Likely causes, not separated yet:
  - The Hessian EMA (β2 0.99, no bias correction, every 10 steps) gets only 105 / 52 updates, so it reaches only
    65% / 41% of its scale. Most coordinates stay clipped to sign steps: final win rate 0.19 at 1M, 0.10 at 2M,
    vs 0.20 at 0.5M.
  - sqrt-scaling raised the LR (1.41e-4 / 2e-4). Sophia was already very LR-sensitive at 0.5M (3e-4 stalled), so the scaled
    LR is probably too high. The interval and β2 are defined per step, so they should scale with the step count too.
- Caveats:
  - One seed.
  - Only one LR per point, set by a rule (no per-batch tuning). AdamW's base LR is itself untuned.
  - So "SOAP > AdamW at 2M" partly measures robustness to the LR rule.
  - Sophia's 1M/2M results reflect its fixed-per-step hyperparameters as much as the batch size.

### Next

- Sophia at large batch: Hessian every 2–5 steps and/or a lower β2 (0.95), and LR not scaled up. Or bias-correct the
  Hessian EMA (a change to the official algorithm).
- A small LR scan per optimizer at 2M before drawing firm conclusions about large batches.
- At the full 6B budget, larger batches lose fewer updates (11k steps at 0.5M, 2.9k at 2M), so the comparison may change.

## 2026-10-08 — LR scan at a 2M-token batch, 500M tokens (8 H200)

**Question:** Was the 2M point of the batch sweep decided by the sqrt LR rule? Scan the LR per optimizer at 2M.

Setup: batch 256 × 8192 = 2M tokens, budget 500M tokens = **238 steps**. WSD, warmup 2% (4 steps), decay 20% (48 steps).
10 evals. Everything else as in the batch sweep. Grid ×2 apart. Two extra Sophia runs with the Hessian every 2 steps
(`--sophia_hess_interval=2`, +50% compute) at the middle of its grid (picked before the results). All 14 completed
238/238 steps with no divergence (max grad norm ≤ 8.1).

| Optimizer | LR → final eval loss | Best |
|---|---|---|
| AdamW | 3e-4: **6.331** · 6e-4: 6.380 · 1.2e-3: 6.587 · 2.4e-3: 6.515 | 3e-4 (grid edge) |
| SOAP | 3e-4: 6.259 · 6e-4: **5.686** · 1.2e-3: 6.177 · 2.4e-3: 6.200 | 6e-4 |
| Sophia, Hessian every 10 | 2.5e-5: **6.215** · 5e-5: 6.226 · 1e-4: 6.450 · 2e-4: 7.193 | 2.5e-5 ≈ 5e-5 (grid edge) |
| Sophia, Hessian every 2 | 5e-5: **5.965** · 1e-4: 6.465 | 5e-5 |

Jobs 515359–515372, W&B runs `lr2m-*`. Cost: AdamW 3.1, SOAP 3.3, Sophia 3.3 (k=2: 4.5) GPU-h per run, 47 GPU-h in total.
Plot: `plots/lr_scan_2m.png`.

### Findings

- **This budget is too short for a clean LR scan.** Every run sits on the early plateau (train loss ~8.0–8.2,
  unigram-level) for roughly the first 40–60 steps. The step where it escapes varies from run to run and dominates the
  final loss at 238 steps. Hence the non-monotonic grids: AdamW 2.4e-3 beats 1.2e-3, and SOAP 6e-4 is 0.5 nats ahead of
  both neighbours because it escaped first (train loss 7.51 at step 60 vs 7.8–7.9 for the others). Differences under
  ~0.2–0.3 nats here are probably within seed noise. That has not been measured: there is no repeat seed yet.
- What still holds across the grids:
  - SOAP's whole grid (5.69–6.26) is at or below AdamW's best (6.33), so SOAP ≥ AdamW at 2M is not an artifact of the LR rule.
  - Sophia with the Hessian every 10 steps is no better than AdamW at any LR (best 6.22). The win rate stays at 0.06–0.07:
    with only 23 Hessian updates, almost everything is clipped.
  - Updating the Hessian every 2 steps helps Sophia at 5e-5 (6.226 → 5.965, win rate 0.19) but not at 1e-4. Its cost is
    +36% time vs k=10.
  - High LRs hurt Sophia most (2e-4: 7.19). At 2M, the sqrt rule's 2e-4 (used in the batch sweep) was the worst choice,
    which explains part of Sophia's collapse there.
- AdamW and Sophia (k=10) peak at the low edge of their grids, but their lowest two points are within 0.01–0.05 of each other,
  inside the noise above. So no grid extension was launched.

### Next (not launched)

- Measure the noise: rerun SOAP 6e-4 and AdamW 3e-4 with another seed (2 runs). This needs a small `--seed` flag in
  train.py first. (Correction: the init comes from torch's fixed default RNG seed, and TrainingArguments' seed 42
  fixes only the data order. `--seed` now sets both; see the next entries.)
- Rather than a longer scan at this batch: the plateau dominates any short run at 2M. A more reliable scan would use
  ~1B+ tokens or a longer warmup.

## 2026-10-08 — Scan winners at a 2M batch, 1B tokens

The two lowest points of the 2M LR scan, rerun at 1B tokens (476 steps; otherwise the scan's setup, 20 evals).

| Run | Final eval loss (ppl) | Same optimizer at 0.5M, 1.1B tokens |
|---|---|---|
| SOAP 6e-4 (`64lwbqmx`, job 516348) | 4.521 (92) | 3.604 (`k6ow93ln`) |
| Sophia 5e-5, Hessian every 2 (`6bkypk4d`, job 516356, win rate 0.18) | 4.898 (134) | 3.394 (`wq8wp81b`, k=10, LR 1e-4) |

Cost: 6.4 and 8.8 GPU-h. (A first Sophia job, 516349, died at DDP init on `25a-hgpn062`: NCCL "Multiple Ranks are using
the same GPU". The rerun excluded that node.)

- **The 0.5M Sophia run (`llama-sophia-lr1e-4-1h-8gpu`) is still the best model** (3.394, on the Hub). Both 2M runs are
  >0.9 nats behind their optimizer's 0.5M result at about the same token count.
- SOAP 6e-4 at 1B (4.521) is no better than the batch sweep's SOAP 2M run, which used the same LR (4.226 at 1.04B, 4.202 at
  1.10B). So SOAP 6e-4's big lead in the scan (5.69 vs ~6.2) was plateau-escape noise, as suspected.
- At ≤1.1B tokens the problem is the step count, not the optimizer: 476–524 steps at 2M vs 2,098 at 0.5M. Second-order
  methods narrow the gap at 2M (SOAP 4.20 vs AdamW 4.43) but don't close it.
- For the 6B run, 0.5M gives ~11.4k steps and 2M ~2.9k. The 1B results say to keep 0.5M unless a 6B-scale test shows
  otherwise.

## 2026-10-08 — Sophia tuning at a 0.5M batch, 1.1B tokens

Each run changes one setting of the best run so far (`llama-sophia-lr1e-4-1h-8gpu`, `wq8wp81b`: LR 1e-4, ρ 0.05,
Hessian every 10 steps, wd 0.2, WSD, 2,098 steps). New `--seed` flag: sets the init (`set_seed` before the model is built),
the data order and Sophia's sampled labels; without it, runs reproduce the old behaviour. All five completed 2,098/2,098 steps.

| Run (W&B) | Change | Final eval loss (ppl) | Final win rate |
|---|---|---|---|
| baseline (`wq8wp81b`) | — | 3.394 (29.8) | 0.15 |
| `s05-seed1` (`aks0lvt7`) | `--seed=1` | 3.324 (27.8) | 0.17 |
| `s05-lr5e-5` (`omad8mgf`) | LR 5e-5 | **3.320 (27.7)** | 0.21 |
| `s05-lr2e-4` (`g8f5wal3`) | LR 2e-4 | 4.484 (88.6) | 0.54 |
| `s05-rho0.1` (`e2susbrf`) | ρ 0.1 | 3.321 (27.7) | 0.29 |
| `s05-k5` (`cmb4skie`) | Hessian every 5 steps | 3.374 (29.2) | 0.18 |

Jobs 516656–516660, ~7.3 GPU-h each (k=5: 8.0; lr2e-4 ran on a slower node: 8.9).

- **Seed noise is ~0.07 nats** (3.394 vs 3.324 for the same settings, from one pair). LR 5e-5, ρ 0.1 and k=5 all land
  within that of the two baseline seeds, so none is a clear win. LR 5e-5 and ρ 0.1 both shrink the step (ρ divides the
  Hessian-scaled step) and both match the better seed, which hints that slightly smaller steps help, but it is not shown.
- **LR 2e-4 is clearly too high** (+1.1 nats): the 1e-4–2e-4 range is a cliff, consistent with 3e-4 / 6e-4 stalling earlier.
  So the usable LR is ≤1e-4 at this batch.
- A Hessian every 5 steps does not help at 0.5M (+9% time), unlike every 2 steps at 2M where only ~240 steps were run.
- Best single run: `s05-lr5e-5` (3.320), but within noise of `s05-seed1` and `s05-rho0.1`.

## 2026-10-09 — Sophia LR 5e-5: second seed, and 2B tokens

Same settings as `s05-lr5e-5` (LR 5e-5, ρ 0.05, Hessian every 10, wd 0.2, WSD 2% / 20%, 0.5M batch).

| Run (W&B, job) | Budget | Final eval loss (ppl) |
|---|---|---|
| `s05-lr5e-5-seed1` (`odfrtif3`, 517026) | 1.1B, 2,098 steps, `--seed=1` | 3.323 (27.7) |
| `s05-lr5e-5-2b` (`a9yvdpi3`, 517027) | 2B, 3,814 steps | **3.047 (21.1)** |

Cost: 7.3 and 12.6 GPU-h.

- Two seeds per LR at 1.1B: LR 5e-5 gives 3.320 / 3.323, LR 1e-4 gives 3.394 / 3.324. LR 5e-5 is at least as good and much
  more consistent across seeds; keep it.
- 2B tokens: 3.047, 0.27 nats below the 1.1B runs, the best model so far.
