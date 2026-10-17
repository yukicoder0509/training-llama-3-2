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

## 2026-10-09 — Sophia LR 5e-5 at 3B tokens: below the ppl-20 baseline

Same settings as `s05-lr5e-5-2b`, budget 3B tokens (5,722 steps). Job 519268, W&B `s05-lr5e-5-3b` (`2yvqk5x7`),
2 h 19 min, 18.5 GPU-h, `budget_completed` true.

| Tokens | Final eval loss (ppl) |
|---|---|
| 1.1B | 3.320 (27.7) |
| 2B | 3.047 (21.1) |
| 3B | **2.897 (18.1)** |

- Beats the ppl-20 baseline (loss ≤ 2.996) by 0.10 nats, more than the ~0.07 seed noise.
- Gain per added token is shrinking (1.1B→2B: −0.27; 2B→3B: −0.15), as expected. The 6B budget is still unused.

## 2026-10-13 — dolma2_qc distribution and a "1" < 0.001 filter

`dolma2_qc` = fastText {"0", "1"} probabilities, on common_crawl docs only (~121B of ~157B tokens; wiki, arXiv,
stack_edu, finemath and olmocr PDFs have no score). Scan: `qc_scan.sbatch` (every 10th shard per directory, 11.6M CC
docs), plot: `plot_qc.py` → `plots/dolma2_qc.png`. Weights are per directory: each topic is split into `-00NN` bucket
directories ~1000x apart in size, and the first scan (weights per topic) over-counted the small buckets (it predicted
29% of CC docs below 0.001; the full count is 17.6%; the per-directory scan gives 17.7%).

- Very skewed: half the CC docs have "1" < 0.012, 75% < 0.085, 5% ≥ 0.99.
- **The `-00NN` buckets are bins of this score**: buckets ≤0015 are 100% below 0.001, 0018 has median 0.0025,
  0019 median 0.052 and 0% below 0.001. 0019 is the largest (~half the sampled CC docs), i.e. the mix draws most from
  the high-"1" end, which suggests AI2 treat high "1" as high quality. Against that: the ≥0.99 docs are 25% short
  (<500 chars) and 12% adult content, vs 2% / 0.2% below 0.001, and spot checks look like ads / SEO. Unresolved.
- Filter `prepare_data.py --qc_max=0.001 --train_frac=0.11` (job 536445, 19 min) → `/work/$USER/dolma3_llama_qc1e-3`:
  over the whole dataset it keeps 34.4M / 104.0M docs, 34.2% of bytes (CC: 17.6% of docs; unscored sources: all).
  `train.bin` = 6.25B tokens (just above the 6B budget). The mix shifts from CC 79% of bytes to 40%: olmocr PDFs 33%,
  stack_edu 19%, finemath 6%. `val.bin` and `eval_docs.jsonl` are byte-identical to the unfiltered ones.

## 2026-10-14 — Exact dedup by text_hash, 1B-token comparison

`prepare_data.py --dedup --min_train_tokens=7e9` (jobs 537088 keys + 537294, ~50 min, 12 CPUs) →
`/work/$USER/dolma3_llama_dedup`. Key per doc: `metadata.text_hash` (128-bit, CC only, truncated to 64 bits), else a
64-bit blake2b of the text. Same-hash texts checked identical on one shard; copies share `id` and URL.

- Whole dataset: 104.0M docs → **35.8M distinct texts (34%), 40% of the characters** (~63B unique train tokens).
  One text has 189,538 copies. In a top-bucket shard (`common_crawl-health-0019`) 56,908 docs are 8,186 texts:
  Dolma 3 upsamples by repetition.
- 365k train docs share their text with an eval doc; all their copies are dropped (eval files untouched; `val.bin`
  byte-identical).
- First version took a prefix of the seed-42 shuffle after keeping each text's first copy. That sample was 92% of the
  original `train.bin`: a text with k copies lands in a 5% prefix with probability 1 − 0.95^k, so the upsampling
  survives. Fixed: train docs are drawn uniformly over the unique texts (new permutation, seed 43).
  `train.bin` = 4.07M docs, 7.18B tokens.

1B tokens (1,907 steps), Sophia LR 5e-5, ρ 0.05, k 10, wd 0.2, WSD 2% / 20%, 0.5M batch:

| Data | Run (W&B, job) | Final eval loss (ppl) |
|---|---|---|
| original | `orig-sophia-lr5e-5-1b` (`3py1n6n3`, 537138) | 3.387 (29.6) |
| dedup | `dedup-sophia-lr5e-5-1b` (`e9g8u50v`, 537295) | **3.371 (29.1)** |

- −0.016 nats, well inside the ~0.07 seed noise: no measurable effect at 1B. Expected to be small here: at ≤6B tokens
  from a 7B sample neither run repeats much; the difference is mainly the mix (dedup drops AI2's quality upsampling
  and decontaminates vs eval). Dedup is at least not worse even though the upsampled top-bucket texts lose weight.

## 2026-10-14 — Data scaling (MiniCPM "Data Scaling with WSD"): Sophia vs AdamW, 0.5–3B tokens

One full WSD run per budget (not decays branched from one run), original data, 0.5M batch, 8 H200. Sophia LR 5e-5,
ρ 0.05, k 10, wd 0.2; AdamW LR 3e-4, ε 1e-5, β2 0.95, wd 0.1 (= `llama-wsd-1h-8gpu`). WSD 2% / 20%. Fit:
`fit_scaling.py` → `plots/data_scaling.png` (grid over α / τ, least squares for L∞, A).

| Tokens | Sophia (W&B) | AdamW (W&B) | Gap |
|---|---|---|---|
| 0.5B | 4.021 (`x5f6h6uw`) | 4.256 (`mm5s28r5`) | 0.235 |
| 1.0B | 3.387 (`3py1n6n3`) | 3.730 (`tz5xr7v5`) | 0.343 |
| 1.1B | 3.320 / 3.323 (`omad8mgf`, `odfrtif3`) | 3.637 (`dkyqsbe6`) | 0.316 |
| 1.5B | 3.176 (`z3jp88ke`) | 3.497 (`3o529zb2`) | 0.321 |
| 2.0B | 3.047 (`a9yvdpi3`) | 3.354 (`vm0ylqxu`) | 0.307 |
| 3.0B | 2.897 (`2yvqk5x7`) | – | |

| Fit | L∞ | Exponent | RMSE | 3B | 6B (ppl) |
|---|---|---|---|---|---|
| Sophia, power L∞ + A·D^−α | 2.58 | α 0.83 | 0.009 | 2.907 | **2.766 (15.9)** |
| Sophia, exp L∞ + A·e^(−D/τ) | 2.91 | τ 0.63B | 0.027 | 2.932 | 2.911 (18.4) |
| Sophia, power, fit on ≤2B only | | | | 2.937 (actual 2.897) | 2.817 (16.7) |
| Sophia, exp, fit on ≤2B only | | | | 3.011 (actual 2.897) | 3.002 (20.1) |
| AdamW, power | 2.65 | α 0.59 | 0.012 | 3.205 | 3.017 (20.4) |
| AdamW, exp | 3.27 | τ 0.65B | 0.014 | 3.286 | 3.266 (26.2) |

- The power law fits; the exponential does not: it flattens by ~2.5B, and the hold-out shows it (predicts 3.011 for 3B,
  actual 2.897). The ≤2B power law is 0.04 high at 3B, so read the power-law 6B numbers as conservative-ish.
- The Sophia lead over AdamW is ~0.31–0.34 nats from 1B to 2B and shrinks only slowly; the power-law fits still have
  a 0.25 gap at 6B (2.77 vs 3.02). Caveat: AdamW has no 3B point, and Sophia's LR / ρ were tuned at 1.1B while AdamW's
  3e-4 has never been tuned at a 0.5M batch (only 3e-4 was run there), so part of the gap may be AdamW tuning. Both L∞ fits (~2.6) are poorly constrained by 0.5–3B data.

## 2026-10-14 — Batch ramp: half batch for the first half of the steps (Sophia, 1.1B tokens)

`--batch_ramp=32:0.3333,64`: 1,398 steps at 32 × 8192 (0.26M tokens), then 1,398 at 64 (1/3 / 2/3 of the tokens),
2,796 steps vs 2,098 at a fixed 64. Same Sophia settings as above (LR 5e-5, ρ 0.05, k 10, wd 0.2, WSD 2% / 20% of
all steps). Two `train.py` fixes for this: Sophia's `bs` (tokens per step, in its update clip) now follows the ramp,
and the ramp keeps grad accumulation fixed (2) and varies the micro-batch (2 → 4 per GPU); before, it had no
accumulation and 8 seqs/GPU went OOM (139.8 GiB; now 69 GiB).

| Run (W&B, job) | Steps | Final eval loss (ppl) | Time |
|---|---|---|---|
| `s05-lr5e-5` / `-seed1` (`omad8mgf` / `odfrtif3`) | 2,098 | 3.320 / 3.323 (27.7) | |
| `ramp-s05-lr5e-5-1.1b` (`1v6ed1yd`, 540186) | 2,796 | **3.270 (26.3)** | 58.5 min (job) |

- −0.05 nats, against 0.003 between the two fixed-batch seeds. Ahead before the decay too: 3.592 at 0.58B tokens vs
  3.634 at 0.71B for the fixed batch. One seed; can't separate "smaller early batch" from "33% more steps / LR updates".
- Cost: the half-batch stage runs below full throughput (smaller micro-batch), the job took 58.5 min vs ~50.

At 3B tokens (`--token_budget=3e9`, same ramp: 3,814 + 3,814 = 7,628 steps vs 5,722 fixed):

| Run (W&B, job) | Steps | Final eval loss (ppl) | Job time |
|---|---|---|---|
| `s05-lr5e-5-3b` (`2yvqk5x7`, 519268) | 5,722 | 2.897 (18.1) | 2:18:54 |
| `ramp-s05-lr5e-5-3b` (`um7ivu4h`, 540798) | 7,628 | **2.870 (17.6)** | 2:25:40 |

- −0.027 nats, about half the 1.1B gain (−0.05): the ramp still helps at 3B but by less, consistent with the gain
  coming mostly from extra early steps, which matter less as the run gets longer. +5% wall time.

## 2026-10-14 — Document-masked packing (`--packing doc`), 1B tokens

Same 8192 blocks and tokens as before, but each document attends only to itself and RoPE positions restart at each
`<|begin_of_text|>` (FlashAttention 3 varlen via HF `cu_seq_lens`; `--attn_implementation flash_attention_3`). Labels
unchanged. Checks: `test_packing.py` (packed forward = per-document forwards, max token-loss diff 0.02 in bf16; concat
differs by up to 4.2). Compile fix for FA3: HF's `warning_once` and the untraceable Hub kernel break each layer's graph,
and the resumed frame recompiled per `layer_idx` until the limit (8), leaving layers 8–15 eager; now
`allow_unspec_int_on_nn_module` + `ignore_logging_functions` (also affects the old FA3 speed numbers).

Speed, 2 GPUs, Sophia (Hessian every 5), median step: sdpa concat 4.80 s, FA3 concat 4.70 s, **FA3 doc 4.26 s (−11%)**;
peak 74 GiB. Full 1B jobs: 46.7 vs 50.9 min.

1B tokens, Sophia LR 5e-5, ρ 0.05, wd 0.2, WSD 2% / 20%, original data. `eval_packing.py` scores both saved models
on the same `val.bin` under both protocols:

| Model (W&B) | Eval concat | Eval doc |
|---|---|---|
| `orig-sophia-lr5e-5-1b` (`3py1n6n3`), trained concat | **3.387** | 3.408 |
| `docpack-sophia-lr5e-5-1b` (`t93qlrhd`), trained doc | 3.417 | **3.368** |

- Each model is best under its own protocol. Like for like: doc-trained is −0.040 under doc eval, +0.030 under concat
  eval. Own-protocol (what each run's W&B eval shows): 3.368 vs 3.387.
- Which number counts depends on how the OJ scores the 50k eval docs: one document per sequence (BOS-prefixed, as
  `prepare_data.py` assumes) is the doc protocol, where packing wins; packed with cross-document context is concat.

## 2026-10-14 — Sophia weight decay and WSD decay fraction (1.1B tokens, 0.5M batch)

Sophia LR 5e-5, ρ 0.05, k 10, warmup 2%, fixed 0.5M batch, 2,098 steps, `--packing concat` (code before the doc
default). One change per run from the baseline (wd 0.2, decay 20%, seeds 3.320 / 3.323). Runs log all flags to W&B
(`args.*`, `derived.*`) from here on.

| Run (W&B) | Weight decay | Decay fraction | Final eval loss | vs baseline mean 3.3215 |
|---|---|---|---|---|
| `s05-lr5e-5-wd0.1` (`mu9kf2gs`) | 0.1 | 0.2 | 3.333 | +0.012 |
| `s05-lr5e-5` / `-seed1` | 0.2 | 0.2 | 3.320 / 3.323 | — |
| `s05-lr5e-5-wd0.4` (`tdmklec7`) | 0.4 | 0.2 | 3.326 | +0.005 |
| `s05-lr5e-5-decay0.1` (`i4uyuv7l`) | 0.2 | **0.1** | **3.318** | −0.003 |
| `s05-lr5e-5-decay0.3` (`gy1wds72`) | 0.2 | 0.3 | 3.344 | +0.023 |
| `s05-lr5e-5-decay0.4` (`gzim3p9b`) | 0.2 | 0.4 | 3.359 | +0.038 |

- Weight decay is flat over 0.1–0.4 (within 0.012; 0.2 best). The AdamW-experience prior (wd 0.1 better) does not
  carry over to Sophia here; Sophia's official 2x-AdamW value holds.
- Decay fraction: monotone, shorter is better at this length: 10% ties 20% (−0.003, inside the seed spread), 30% and
  40% are clearly worse. Before the decay the longer-decay runs are ahead (decay0.4 3.464 vs 3.494 at 0.87B), but
  they spend fewer steps at the peak LR and end higher. Keep 20% (or 10%); not worth a combined wd0.1 + decay0.1 run.
- `--decay_frac` default changed to 0.1 (MiniCPM WSD value; not worse than 0.2 here). Earlier runs passed or defaulted to 0.2.

## 2026-10-15 — Two-stage WSD decay (`--lr_schedule wsd2`): stable2 length scan (Sophia, 1B tokens)

Schedule: warmup 2% → stable1 at peak → linear decay to 0.1 × peak over 5% of steps → stable2 at 0.1 × peak for
`--stable2_frac` → linear decay to 0 over 5% (`--decay1_frac=0.05 --stable2_lr_ratio=0.1 --decay_frac=0.05`); stable1
gets the rest. Checked in W&B (s2 = 0.3): 5e-5 to step 1,144, 5e-6 from ~1,240. Sophia LR 5e-5, ρ 0.05, k 10, wd 0.2,
0.5M batch, doc packing, 1,907 steps.

| Run (W&B) | Schedule | Loss at 0.80B | Final eval loss (doc) |
|---|---|---|---|
| `sophia-lr5e-5-decay0.1-1b` (`hpruhuhp`) | plain WSD, 10% linear decay (baseline) | 3.567 | **3.375** |
| `docpack-sophia-lr5e-5-1b` (`t93qlrhd`) | plain WSD, 20% linear decay | 3.549 | 3.368 |
| `sophia-wsd2-s2-0-1b` (`w2tyale7`) | wsd2, stable2 0% | 3.524 | 3.394 |
| `sophia-wsd2-s2-0.1-1b` (`e4axicm4`) | wsd2, stable2 10% | 3.540 | 3.404 |
| `sophia-wsd2-s2-0.2-1b` (`6qixkxsc`) | wsd2, stable2 20% | 3.518 | 3.470 |
| `sophia-wsd2-s2-0.3-1b` (`axdkidep`) | wsd2, stable2 30% | 3.583 | 3.539 |

- Every stable2 length is worse than the plain linear decay, and longer is monotonically worse (+0.019, +0.029,
  +0.095, +0.164). The plateau at 0.1 × LR makes little progress (s2 = 0.2: 3.518 at 0.80B, 3.470 at the end), and
  the steps it takes come out of the peak-LR stage. Even stable2 = 0 (a kinked decay: 1 → 0.1 in 5%, 0.1 → 0 in 5%)
  loses 0.019 to the straight 10% line. Same picture as the decay-fraction scan: at 1B tokens the run is short of
  peak-LR steps, so no time should be spent at low LR beyond a short final decay.
- Decay 10% vs 20% under doc packing: 3.375 vs 3.368 (+0.007, about the seed spread); the 10% default stands.

## 2026-10-15 — Muon (pip `muonium` 0.2.0) at 1B tokens: package defaults and LR sweep

`--optimizer muonium`: Muon for the 112 decoder-layer matrices (973M params; momentum 0.95 Nesterov, Polar Express 5
steps, LR × sqrt(max(1, rows/cols)), cautious decoupled wd), muonium's fused AdamW (β 0.9 / 0.95, ε 1e-8) for the tied
embedding + norms (263M). 0.5M batch, doc packing, 1,907 steps, 8 H200. The first step compiles muonium's update for
~8–12 min (`TORCHINDUCTOR_CACHE_DIR=/work/$USER/inductor_cache` shares the cache across jobs); then ~ AdamW speed.

| Run (W&B) | Muon LR | AdamW LR | wd | Decay | Final eval loss (doc) |
|---|---|---|---|---|---|
| `muonium-default-1b` (`ctf5qm8f`) | 1e-3 (package default, one LR) | 1e-3 | 0.1 | 20% | 3.380 |
| `muonium-muon-lr0.005-1b` (`wxqu9zmt`) | 0.005 | 3e-4 | 0.1 | 10% | 3.091 |
| `muonium-muon-lr0.01-1b` (`lbuf7ebv`) | **0.01** | 3e-4 | 0.1 | 10% | **3.063** |
| `muonium-muon-lr0.02-1b` (`us4l6inz`) | 0.02 | 3e-4 | 0.1 | 10% | 3.156 |
| `muonium-muon-lr0.04-1b` (`pxqevoqw`) | 0.04 | 3e-4 | 0.1 | 10% | 3.526 |
| `muonium-normuon-lr0.01-1b` (`gzlluua7`), NorMuon | 0.01 | 3e-4 | 0.1 | 10% | **3.055** |
| `muonium-normuon-sophia-1b` (`upzn271q`), NorMuon + Sophia (lr 5e-5, ρ 0.05, wd 0.2) on embedding + norms | 0.01 | – | 0.1 / 0.2 | 10% | 3.159 |
| Sophia baseline `sophia-lr5e-5-decay0.1-1b` (`hpruhuhp`) | – | – | 0.2 | 10% | 3.375 |

- **Muon at 0.01 beats Sophia by 0.31 nats at 1B** (3.063 vs 3.375), about what Sophia reached at 2B (3.047).
  Clear optimum at 0.01–0.005 (half of Keller Jordan's 0.02); 0.04 is unstable early (4.0 at 0.35B).
- The package default (1e-3) is 10x too low here but already roughly ties Sophia (3.380 vs 3.368, both 20% decay).
- **NorMuon (same settings, LR 0.01) edges out Muon: 3.055 vs 3.063** (0.008, ~2.5x the 0.003 seed noise seen at 1.1B;
  one seed each). NorMuon is picked for the Sophia hybrid (Sophia replacing AdamW on embedding + norms).
- **Sophia in place of AdamW for embedding + norms hurts: 3.159 vs 3.055** (+0.10), and halves throughput
  (2.1e5 vs 4.2e5 tok/s: Hessian re-forwards plus the slower path). Curves track until ~0.75B, then the AdamW hybrid
  gains much more in the decay. Keep NorMuon + AdamW.

## 2026-10-15 — How much does the original train.bin repeat? (`repetition.py`, job 547494, 52 s)

Exact repetition of whole docs (identical token sequences) in the original sample vs the dedup one. The first 3B tokens
stand for a 3B run (docs are shuffled at prepare time; runs draw random blocks).

| Range | Docs | Distinct texts | Tokens in repeat copies | Tokens in texts with 2 / 3–4 / ≥5 copies | Train docs = a val doc |
|---|---|---|---|---|---|
| original, first 3B | 2.29M | 91.1% | **0.26B (8.6%)** | 14.8% / 1.7% / 0.0% | 2,597 docs (697 of 9,980 val texts) |
| original, whole 7.86B | 5.20M | 91.4% | 0.59B (7.5%) | 13.1% / 1.4% / 0.0% | 13,054 docs (1,590 val texts) |
| dedup, first 3B | 2.09M | 99.7% | 0.0% | 0.0% | 14 val texts, 0.01M tokens |
| dedup, whole 7.18B | 4.07M | 99.8% | 0.0% | 0.0% | 20 val texts, 0.01M tokens |

- Repetition in the original sample is mild: nearly all of it is texts seen 2–4 times. The heavily upsampled texts
  (thousands of copies, up to 9,329) are tiny docs that add up to ~0 tokens. The leftover copies in dedup are the same
  kind of near-empty docs.
- Upper bound for 3B: if all 8.6% repeat tokens were worthless, it equals ~9% fewer tokens. Sophia's local slope
  (2B 3.047 → 3B 2.897, dL/dlnD ≈ −0.37) gives ≤ ~0.03 nats. 2–4x repeats are nearly as good as fresh data
  (Muennighoff et al. 2023), so the real effect should be well below that.
- Contamination is the bigger issue: 7% of val texts (697) appear in the first 3B of original train, ~3.7 copies
  each; 16% (1,590) in the whole file. That favours the original data on `val.bin` eval.


## 2026-10-15 — Is train.bin representative of the mix? Token shares per (source × topic × quality vigintile × length) (`metadata_dist.py`, job 548010, 2 min)

Cells come from each doc's directory (`<source>-<topic>[-<CC quality vigintile>]`) × 6 token-length bins, 1,252
non-empty cells. Population: all 104.0M train-eligible docs (~157B tokens, from per-doc chars × the cell's
tokens/char). Train sets: exact token counts (docs line up with `train.bin` in file order). Plot: `plots/metadata_dist.png`.

| Train set | TV distance vs mix (all cells) | source | topic | CC vigintile | length |
|---|---|---|---|---|---|
| original (5% uniform doc sample) | **0.0087** (sampling noise alone: 0.0084 ± 0.0005) | 0.002 | 0.004 | 0.001 | 0.001 |
| dedup (uniform over unique texts) | **0.327** | 0.146 | 0.214 | 0.130 | 0.072 |

- **Original train.bin is as close to the mix as a random sample can be.** Doc counts per cell: chi²/dof 0.95, 1 cell
  of 686 (≥1000 docs) beyond |z| 3 (~1.9 expected). The only visible deviations are tiny olmOCR topics (<0.01%).
- **Dedup changes the mixture a lot.** Dolma 3 sets its mix weights by repeating docs, so one copy per text undoes them:
  CC vigintile 19 (top quality) 41.7% → 22.6%, olmOCR PDFs 12.6% → 27.1%, finemath 2.6% → 1.4%, stack_edu 7.0% →
  5.7%, docs ≥32k tokens 10.2% → 15.3%. Within a 5% sample the upsampling shows up as more docs from those cells, not
  as repeats (hence only 8.6% repeat tokens in the first 3B, entry above).
- So the dedup vs original runs compared two different mixtures more than repeated vs unique data, and were scored on
  `val.bin`, which follows the original mix (and overlaps the original train). Dedup still came out 0.016 ahead at 1B.
  A clean test of dedup alone would draw unique texts with per-cell weights matching the mix's token shares (enough
  unique tokens exist: ~63B in total, ~14B in CC vigintile 19 vs 2.5B needed for 6B).

## 2026-10-15 — Small scoring model for perplexity filtering: baseline (`train_scorer.py`)

`train_scorer.py` = train.py with the Llama 3.2 1B config shrunk to 768 hidden / 2048 FFN / 12 layers / 12 query + 4 KV
heads / head_dim 64, 2048 context, plain RoPE (θ 5e5, no llama3 scaling): 174M params (98.5M tied embedding + 75.5M
layers). 2048-token blocks × 256 = 0.5M tokens/step (32/GPU, no grad accum), doc packing, WSD 2% / 10%, wd 0.1.
Muon (muonium) on the layer matrices, AdamW on embedding + norms. W&B project
`bendoris0509-national-yang-ming-chiao-tung-university/llama3.2-arch-small-scoring-model`.

| Run (W&B, job) | Tokens | Muon LR | AdamW LR | Final eval loss (2048 blocks, ppl) |
|---|---|---|---|---|
| `scorer-muon-1b` (`o5uvb6uk`, 548614) | 1B (1,907 steps) | 0.01 | 1e-3 | 3.269 (26.3) |

- 2.13M tokens/s on 8 H200 (MFU 0.34); 19 min job, ~9 min of it the first-step compile. Saved to
  `/work/$USER/llama_models/scorer-muon-1b`.
- Eval loss uses 2048-token blocks: not comparable with the 8192-context 1B runs.
- The decay took it from 3.382 to 3.269; before the decay it was still dropping ~0.012 per eval (~50M tokens), so a
  longer budget should help.
- Fixes: `rope_scaling: null` in `--config_overrides` now means plain RoPE (transformers 5 otherwise sets
  `rope_parameters` to None and the model build fails, job 548573).

## 2026-10-15 — NorMuon + AdamW at 3B tokens vs the current Hub model

`muonium-normuon-lr0.01-3b` (`1c1uxoe1`, job 547240): NorMuon LR 0.01 (matrices) + AdamW 3e-4 (embedding, norms), wd 0.1,
doc packing, WSD 2% / 10%, fixed 0.5M batch (no ramp), 5,722 steps, 2:12 h, 427k tokens/s. In-training eval 2.774.
Both checkpoints through `eval_packing.py` (job 549540, same 1,850 val blocks):

| Model | concat eval | doc eval |
|---|---|---|
| `ramp-s05-lr5e-5-3b` (Sophia + batch ramp, concat-trained; on the Hub) | 2.870 (ppl 17.6) | 2.879 (17.8) |
| `muonium-normuon-lr0.01-3b` | **2.812 (16.7)** | **2.773 (16.0)** |

- NorMuon + AdamW wins under both protocols: −0.058 concat, −0.106 doc, with fewer steps (5,722 vs 7,628) and no ramp.

### Scoring model LR sweep (1B tokens, 1,907 steps; same setup as `scorer-muon-1b`)

| Run (W&B) | Muon LR | AdamW LR | Final eval loss (2048 blocks) |
|---|---|---|---|
| `scorer-muon0.005-adamw1e-3-1b` (`wpywcrtf`) | 0.005 | 1e-3 | 3.285 |
| `scorer-muon-1b` (`o5uvb6uk`) | 0.01 | 1e-3 | 3.269 |
| `scorer-muon0.02-adamw1e-3-1b` (`rx9j83fy`) | 0.02 | 1e-3 | 3.283 |
| `scorer-muon0.04-adamw1e-3-1b` (`b616ncdh`) | 0.04 | 1e-3 | 3.442 |
| `scorer-muon0.01-adamw3e-4-1b` (`cqkuj3eh`) | 0.01 | 3e-4 | 3.347 |
| `scorer-muon0.01-adamw3e-3-1b` (`lfsusnqc`) | 0.01 | **3e-3** | **3.252** |
| `scorer-muon0.01-adamw1e-2-1b` (`7t6g91jk`) | 0.01 | 1e-2 | 3.256 |

- Muon LR 0.01 is best here too (same as the 1B model). AdamW (embedding + norms) prefers a higher LR than the 1B
  model's 3e-4: 3e-3 and 1e-2 tie (within 0.004), 3e-4 is 0.095 worse.
- First 0.005 attempt (549195) hung at startup on 25a-hgpn083 (no output for 45 min); rerun as 550143.
- 6B run with Muon 0.01 / AdamW 3e-3: `scorer-muon0.01-adamw3e-3-6b` (job 550297), then `score_docs.py` on it (550298).

## 2026-10-15 — NorMuon + AdamW + batch ramp at 1B tokens

Same as `muonium-normuon-lr0.01-1b` (NorMuon 0.01, AdamW 3e-4, wd 0.1, WSD 2% / 10%, doc packing) plus
`--batch_ramp=32:0.3333,64` (batch 32 for the first 1/3 of the tokens, then 64; LRs unchanged).

| Run (W&B, job) | Steps | Final eval loss (doc) |
|---|---|---|
| `muonium-normuon-lr0.01-1b` (`gzlluua7`, 544764), fixed 64 | 1,907 | 3.055 |
| `muonium-normuon-lr0.01-ramp-1b` (`91r4lm1m`, 550180) | 2,542 | **3.003** |

- The ramp helps NorMuon too: −0.052 at 1B (Sophia gained −0.050 at 1.1B, −0.027 at 3B). One seed each, but far
  beyond the 0.003 seed noise seen at 1.1B. 1:03 h job (vs 0:57), the extra steps are small-batch steps.

## 2026-10-15 — 6B scoring model, and per-doc scores of 1B held-out tokens (`score_docs.py`)

`scorer-muon0.01-adamw3e-3-6b` (`ap3qkov5`, job 550297): 174M scorer, Muon 0.01 / AdamW 3e-3, 6B tokens (11,444 steps),
55 min. Final eval loss **2.986** (2048 blocks; 1B run: 3.252).

Scored (job 550298, 39 min, 1 GPU): the 661,539 docs right after train.bin in the seed-42 shuffle (0.996B tokens; not in
train.bin, not eval), each in 2048-token windows. Output in `/work/$USER/dolma3_llama/scores/scorer-muon0.01-adamw3e-3-6b/`:
`scores.parquet` (rank, text_hash, key, source, topic, vigintile, n_tokens, n_pred, mean_loss), `percentiles.csv`
(p1–p99 per group and over all, weighted by docs and by tokens).

| Percentile of mean_loss (all docs) | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| by docs | 1.00 | 1.51 | 1.84 | 2.55 | 3.10 | 3.49 | 3.86 | 4.14 | 5.11 |
| by tokens | 0.71 | 1.55 | 1.93 | 2.55 | 3.05 | 3.42 | 3.71 | 3.92 | 4.66 |

- Loss depends mostly on the source: median per doc is 1.85 for code (stack_edu), 2.55–2.66 for arXiv / finemath /
  PDFs, 3.2 for common_crawl and wiki. A single global cut would remove almost only common_crawl (high cut) or
  only code and math (low cut); per-group percentiles keep the mix.
- Both tails hold junk candidates: near-zero loss on huge docs (three finemath docs of ~520k tokens at loss 0.004–
  0.005, i.e. repeated text; a 3.0M-token PDF at 0.46); 0.6% of tokens have loss < 0.5, and 0.6% have loss > 5.
- 188 docs over 100k tokens hold 3.9% of the tokens. In small groups one doc sets the token-weighted percentiles
  (e.g. pdf:software_development p1–p20 = 0.465 is the single 3.0M-token doc, 22% of the group), so doc-weighted
  percentiles are the steadier basis for thresholds; the token-weighted ones show how much data a cut removes.

### LR retune with the batch ramp (NorMuon + AdamW, 1B tokens, `--batch_ramp=32:0.3333,64`, 2,542 steps)

| Run (W&B) | Muon LR | AdamW LR | Final eval loss (doc) |
|---|---|---|---|
| `muonium-normuon0.005-adamw3e-4-ramp-1b` (`ou44oo9j`) | 0.005 | 3e-4 | 3.023 |
| `muonium-normuon0.0075-adamw3e-4-ramp-1b` (`rva7kuvs`) | 0.0075 | 3e-4 | 3.004 |
| `muonium-normuon-lr0.01-ramp-1b` (`91r4lm1m`) | 0.01 | 3e-4 | 3.003 |
| `muonium-normuon0.015-adamw3e-4-ramp-1b` (`zna1ualx`) | 0.015 | 3e-4 | 3.029 |
| `muonium-normuon0.02-adamw3e-4-ramp-1b` (`flb6pxhf`) | 0.02 | 3e-4 | 3.101 |
| `muonium-normuon0.01-adamw1e-3-ramp-1b` (`nemgffef`) | 0.01 | 1e-3 | 2.981 |
| `muonium-normuon0.01-adamw3e-3-ramp-1b` (`nlo7gfgz`) | 0.01 | **3e-3** | **2.941** |

- Muon LR: flat optimum at 0.0075–0.01 (same as without the ramp).
- AdamW LR (embedding + norms) matters more than expected: 3e-4 → 3e-3 gains 0.062, and is still improving at the
  edge of the grid (as for the scorer). Extending to 6e-3 and 1e-2 (jobs 551467, 551468).

| `muonium-normuon0.01-adamw6e-3-ramp-1b` (551467) | 0.01 | 6e-3 | 2.943 |
| `muonium-normuon0.01-adamw1e-2-ramp-1b` (551468) | 0.01 | 1e-2 | 2.950 |

- AdamW LR peaks at 3e-3 (6e-3 within 0.002, 1e-2 +0.009).
- Scaled to 3B tokens: `muonium-normuon0.01-adamw3e-3-ramp-3b` (job 552056), then `eval_packing.py` against
  `muonium-normuon-lr0.01-3b` (job 552057, afterok).

## 2026-10-15 — Perplexity bands on the proxy (174M scorer setup, 1B tokens)

`score_bin.py` (job 551506) scored every train.bin doc with `scorer-muon0.01-adamw3e-3-6b`; `build_bands.py` drops each
group's lowest / highest mean-loss docs (per-group doc percentiles). Proxy: `train_scorer.py`, Muon 0.01 / AdamW 3e-3,
1B tokens, same eval (2048 blocks) as `scorer-muon0.01-adamw3e-3-1b`.

| Run (job) | Drop low / high | Tokens kept | Final eval loss |
|---|---|---|---|
| `scorer-muon0.01-adamw3e-3-1b` (549199), unfiltered | – | 100% | 3.252 |
| `proxy-band-hi5-1b` (551512) | 0 / 5% | 97.2% | 3.253 |
| `proxy-band-lo5-1b` (551514) | 5% / 0 | 95.4% | 3.250 |
| `proxy-band-lo5-hi5-1b` (551516) | 5% / 5% | 92.6% | 3.257 |
| `proxy-band-lo5-hi10-1b` (552130) | 5% / 10% | 89.5% | 3.262 |
| `proxy-band-lo5-hi15-1b` (552132) | 5% / 15% | 85.7% | 3.267 |

- 5% cuts: all within 0.007 of unfiltered (seed noise ~0.003), i.e. no measurable effect on eval perplexity.
- Eval loss rises steadily with the high cut (lo5: hi 5 / 10 / 15 → 3.257 / 3.262 / 3.267, +0.005 per 5%) while train loss
  falls (3.277 / 3.228 / 3.202); dropping low-loss docs does the reverse (lo5 alone: train 3.320, eval 3.250). Filtering
  makes the training data easier but less like the unfiltered eval set, so it does not help in-distribution perplexity.
- Checked: each run read its band's train.bin, val.bin is the same file, and score doc starts / ends land on BOS / EOS
  in train.bin (2,000 sampled docs, token counts sum to train.bin's length).

## 2026-10-15 — Loss-based soft reweighting of rows (setup)

Draw rows with probability ~ exp(beta * z), z = scorer loss standardized within topic; no row dropped or copied.
Setup in `config.yaml`; `build_pool.py` (steps 1-3), `select_rows.py` (step 4; not `select.py`, which would shadow the
stdlib `select` module), `train.py` evaluates `dev.bin` too when the data dir has one (`eval/dev_loss`).

- Pool: rows with splitmix64(seed, global row index) < 12.73% (target 19B tokens), among ranks
  [5,198,546, dev start): excludes the scorer's training rows (train.bin), our dev set and the official eval.
  Row index, not the `id` field, so duplicate rows are drawn independently. Rows are tokenized once
  (`/work/$USER/dolma3_reweight/tokens/`); selection copies tokens instead of re-reading text.
- Dev set: the 10k ranks right before the official eval (`dolma3_reweight/dev.bin`); pool rows with a dev text_hash
  are dropped. Official eval only for reference.
- Topic for z: `scoring.group_name` (cc:/pdf: WebOrganizer topic, code: language, else the source).
- Scorer: `scorer-muon0.01-adamw3e-3-6b`, BOS + first 511 tokens per unique text_hash, z clipped to ±3, n_scored < 32 → 0.
- Jobs: manifest 552579 (array 0-7, 8 min), score 552580 (20 min; 20-doc check max diff 2.2e-7), merge 552727
  (552588 failed: 2 files hold only repeats of texts first seen elsewhere, so no score file). Proxies (train_scorer.py, 1B tokens from a 3B
  subpool, Muon 0.01 / AdamW 3e-3): beta 0 seeds 0/1/2 (552729/552731/552733), beta −0.25 (552735), beta 0.25
  (552737), beta 0.5 (552739). Same seed 0 subpool for all beta runs.
- Pool: 12,574,477 rows, 10,271,749 unique texts (18.3% of rows repeat a text).
- Merge (552727): pool 12,547,298 rows / 18.97B tokens after dropping 27,179 rows whose text is a dev text; 1.10% of
  rows (0.01% of tokens) have n_scored < 32 → z = 0; 1.0% of rows clipped at |z| = 3.
- Topic row shares vs the full mix: max 0.012 pp before the dev drop. After it, pdf:science_math_and_technology is
  −0.121 pp: one 6-token text, "[REMOVED]" (a PDF placeholder), has 22,832 copies in the pool and one in dev, so all
  copies go (137k tokens, 0.0007% of the pool). Kept the rule; token shares are unaffected.
- Selections (1B from a 3B subpool, seed 0): selected fraction per z bin is flat at 0.33 for beta 0; for beta 0.5 it
  rises from 0.09 (z < −2.5) to 0.81 (z > 2.5), no bin saturated; mean z of the selection −0.19 / +0.18 / +0.35 for
  beta −0.25 / 0.25 / 0.5; topic token shares move ≤ 0.35 pp.

### Reweighting proxies (train_scorer.py, 1B tokens from a 3B subpool; dev = our 10k-doc dev set)

| Run (job) | beta | Seed | Mean z selected | Dev loss | Official eval loss (reference) |
|---|---|---|---|---|---|
| `proxy-rw-b0-s0-1b` (552729) | 0 | 0 | 0.00 | 3.222 | 3.248 |
| `proxy-rw-b0-s1-1b` (552731) | 0 | 1 | | 3.225 | 3.248 |
| `proxy-rw-b0-s2-1b` (552733) | 0 | 2 | | 3.221 | 3.246 |
| `proxy-rw-b-0.25-s0-1b` (552735) | −0.25 | 0 | −0.19 | 3.227 | 3.252 |
| `proxy-rw-b0.25-s0-1b` (552737) | +0.25 | 0 | +0.18 | 3.223 | 3.249 |
| `proxy-rw-b0.5-s0-1b` (552739) | +0.5 | 0 | +0.35 | 3.229 | 3.255 |

- Baseline: mean 3.2227, std 0.0021 over 3 seeds; a win needs dev < 3.2185 (mean − 2 std). None qualifies.
- Against the same-subpool baseline (seed 0, 3.222): −0.25 is +0.005, +0.25 +0.001, +0.5 +0.007. Loss is lowest at
  beta 0 and rises both ways; dev and official eval agree (same order).
- Decision: keep beta = 0 (uniform random rows). Like hard band filtering, tilting toward easier or harder docs moves
  the training distribution away from the eval's and does not pay for itself at this scale.

## 2026-10-16 — `--batch_ramp` stages by absolute tokens

`--batch_ramp=B1:t1,...,Bn`: ti > 1 is now a token count (`32:1e9,64`, or `32:1B,64` / `32:333M,64`), so the switch
lands at the same tokens whatever `--token_budget` is; ti ≤ 1 is still a fraction of the budget (old commands give the
same steps: `32:0.3333,64` → 1,271 + 1,271 steps at 1B, 3,814 + 3,814 at 3B). The switch is at the last whole step
within the stage's tokens (`32:1B,64` at 8,192 context: 3,814 steps = 0.9998B tokens). Note the fraction form scales
with the budget: the 3B ramp run (552056) used batch 32 for 1B tokens, the 1B runs for 333M.

## 2026-10-16 — NorMuon + AdamW 3e-3 + batch ramp at 3B tokens: new best, now the train.py defaults

`muonium-normuon0.01-adamw3e-3-ramp-3b` (job 552056, 2:38 h): NorMuon 0.01, AdamW 3e-3, wd 0.1, WSD 2% / 10%, doc
packing, `--batch_ramp=32:0.3333,64` (batch 32 for the first 1B tokens), 3B tokens. In-training eval 2.714.
`eval_packing.py` (job 552057, same 1,850 val blocks):

| Model | concat eval | doc eval |
|---|---|---|
| `muonium-normuon-lr0.01-3b` (previous best: AdamW 3e-4, no ramp) | 2.812 (ppl 16.7) | 2.773 (16.0) |
| `muonium-normuon0.01-adamw3e-3-ramp-3b` | **2.779 (16.1)** | **2.715 (15.1)** |

- −0.059 doc / −0.033 concat. At 1B the same two changes gave −0.114 (3.055 → 2.941), so about half the gain remains
  at 3B.
- These settings are now train.py's defaults (`--optimizer muonium --muonium_algorithm normuon`, AdamW LR 3e-3,
  Muon LR 0.01, `--batch_ramp 32:0.3333,64`). Other optimizers keep their old LR defaults (6e-4; muon 1.25e-3).
  `--global_batch_size` or `--batch_ramp=none` turns the ramp off; train_scorer.py passes `--global_batch_size=256`, so
  the scorer / proxy runs are unchanged. profile.sbatch pins batch 64.

## 2026-10-16 — When to double the batch: switch-point sweep by branching from checkpoints (1B tokens)

Defaults (NorMuon 0.01, AdamW 3e-3, wd 0.1, doc packing), 1B tokens, `--schedule_unit tokens` (WSD warmup 20M tokens,
decay from 900M: the LR depends only on tokens seen, so every ramp shares the batch-32 prefix). One batch-32 trunk saves
full checkpoints (model, optimizer, scheduler, RNG, step) at each switch point; each branch resumes there with
`--batch_ramp=32:<T>,64`. The trunk run to the end is the fixed-32 point. Tokens: trunk 1B + branches 3.2B (+ fixed 64
1B) vs 7B for independent runs.

Check first (174M scorer, 100M tokens, batch 128 → 256 at 30M): trunk A saved at step 114 (29.9M tokens), branch B
resumed from it, reference C ran `128:30M,256` straight through. B vs C: final eval 4.445 vs 4.447, train loss within
0.005 from 38M tokens on, identical LR. (C's first try died on hgpn173: /work not writable; node now excluded.)

| Run (job) | Batch 32 for | Final eval loss (doc) |
|---|---|---|
| `ramp-sweep-fixed64-1b` (554165; step-unit WSD, same as tokens at a fixed batch) | 0 (fixed 64) | 2.991 |
| `ramp-branch-at100M-1b` (554352) | 100M | 2.977 |
| `ramp-branch-at200M-1b` (554353) | 200M | 2.968 |
| `ramp-branch-at333M-1b` (554354) | 333M | 2.957 |
| `ramp-branch-at500M-1b` (554355) | 500M | **2.945** |
| `ramp-branch-at667M-1b` (554356) | 667M | **2.945** |
| `ramp-trunk32-1b` (554351) | 1B (fixed 32) | 2.983 |

Reference: `muonium-normuon0.01-adamw3e-3-ramp-1b` (333M, step-unit WSD) 2.941.

- Later is better up to 500-667M (2.945, flat between them), −0.012 vs the current default switch at 333M and −0.046 vs
  fixed 64. Batch 32 to the end (fixed 32) is 0.038 worse than 667M: the last third, which includes the decay, wants 64.
- The 333M branch (2.957) is 0.016 worse than the step-unit run with the same ramp (2.941). Only the schedule differs:
  step-unit WSD with the ramp has a 133M-token decay (10% of 2,542 steps, mostly at batch 64) and a 13M-token warmup;
  token-unit has 100M and 20M. So a longer decay likely helps here; a `--decay_frac` sweep is the next branch to try
  (it branches from one checkpoint at the decay start).
- 1B-budget only: whether the best switch scales as a fraction of the budget (50-67%) or stays at ~0.5B tokens is not
  tested. Cost: trunk 1:03 h + branches 0:56 / 0:52 / 0:46 / 0:37 / 0:30 (4:44 h of 8 GPUs) vs ~6.5 h for six
  independent 1B runs (~1:05 each).
- Checkpoints kept in /work/$USER/llama_ckpt/ramp-trunk32-1b (5 x 11 GB) for more branches.

## 2026-10-16 — Switch-point sweep at 3B tokens: the switch scales as a fraction of the budget

Same method as the 1B sweep, budget 3e9, token-unit WSD (warmup 60M, decay 2.7B → 3B). Trunk `ramp-trunk32-3b`
(555967, batch 32, `--per_device_batch=2`) saved at 0.5/1/1.5/2/2.5B and was cancelled after the last one; each branch
`ramp-branch-at<T>-3b` resumes with `--batch_ramp=32:<T>,64` and runs to 3B.

| Run (job) | Batch 32 for | Fraction | Final eval loss (doc) |
|---|---|---|---|
| `ramp-branch-at500M-3b` (556126) | 0.5B | 0.17 | 2.737 |
| `ramp-branch-at1B-3b` (556257) | 1B | 0.33 | 2.731 |
| `ramp-branch-at1.5B-3b` (556462) | 1.5B | 0.50 | **2.728** |
| `ramp-branch-at2B-3b` (556638) | 2B | 0.67 | 2.730 |
| `ramp-branch-at2.5B-3b` (556708) | 2.5B | 0.83 | 2.745 |

Reference: `muonium-normuon0.01-adamw3e-3-ramp-3b` (552056; switch at 1B, step-unit WSD) 2.714.

- Fraction, not a fixed count: the 1B sweep's optimum (0.5-0.67B) would put the 3B best at 0.5B if it were a fixed token
  count, but 0.5B is 0.009 worse than the best. The best is at 0.5 of the budget, as at 1B (0.5-0.67), and the curve is
  flat from 0.33 to 0.67 (within 0.003). Switching very late (0.83) costs 0.017. For 6B: switch at ~3B (`32:0.5,64`).
- The flat region widens with budget: 0.33 costs 0.012 at 1B but 0.003 at 3B.
- All branches lose to 552056 (2.714), and the 1B branch has the same ramp: the 0.017 gap is the schedule, as at 1B (0.016).
  Step-unit WSD with this ramp gives a 400M-token decay (10% of the steps, all at batch 64) and 40M warmup vs 300M and
  60M in token units. The decay length matters more than the switch point; `--decay_frac` is the next sweep.
- Cost: trunk 2:13 h + branches 1:59 / 1:38 / 1:16 / 0:54 / 0:32 (8:32 h of 8 GPUs) vs ~13 h for five independent runs.
- Checkpoints in /work/$USER/llama_ckpt/ramp-trunk32-3b (5 x 11 GB).

## 2026-10-17 — train.py split into llama_train/ (no behavior change); same-seed run-to-run noise

`train.py` (872 lines) → `main()` + `llama_train/{cli,plan,model,data,optim,callbacks,trainer}.py` (README: Code layout).
Checked against the pre-refactor commit 353b50b:
- `tests/plan_dump.py`: per-step batch, tokens seen and LR factor, plus the printed plan lines, for 9 flag sets (default
  6B / 3B, token-unit trunk + branch, wsd2, fixed batch + `--save_at_tokens`, cosine, `--max_steps` ramp, scorer):
  byte-identical (52,689 lines).
- W&B config of a 1B 100M-token `--seed 0` run (all flags, derived plan, TrainingArguments; 226 keys): identical.
- The same run, twice per version (doc eval loss at 100M):

| Code | Run 1 | Run 2 |
|---|---|---|
| old (353b50b) | 4.094 (561387) | 4.102 (561520) |
| new | 4.106 (561392) | 4.098 (561521) |

  Same code and seed differ by 0.008 at the end (first eval 8.49-8.77): training is not bitwise reproducible (FA3 /
  compiled kernels, NCCL reduction order), so old vs new (means 4.098 / 4.102) is within noise. Treat differences
  under ~0.01 between single 1B 100M-token runs as ties.
- Resume from the new code's 30M checkpoint (561393): 4.105 vs 4.106 uninterrupted. Scorer 50M: 5.795 old vs 5.800 new.
- `test_packing.py` OK; `eval_packing.py` doc loss = in-training eval (4.0942 / 4.1056), via `DocPackingCollate` import.
