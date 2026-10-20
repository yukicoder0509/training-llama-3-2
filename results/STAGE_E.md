# Stage E: batch schedule and gradient noise scale (Llama-3.2-1B, 1B tokens, 2026-10-19)

All runs: train.py defaults (NorMuon 0.01 + AdamW 3e-3, wd 0.1), `--decay_frac 0.3 --decay_type 1-sqrt` (step units),
`--dyn_every 50 --dyn_noise`, W&B `bendoris0509-national-yang-ming-chiao-tung-university/llama3.2-1b-optim-dynamics`.
E3 (weight decay) skipped by decision; E2a / E2b dropped (answered by the 2026-10-16 switch-point sweeps).

| Run | W&B | Job | Batch schedule | Steps | Final eval | Δ vs baseline mean 2.9090 |
|---|---|---|---|---|---|---|
| `normuon-decay0.3-1sqrt-1b-dyn` (baseline, Stage D) | `3ldi2uxu` | 570176 | 32:0.5,64 | 2,860 | 2.9080 | −0.0010 |
| E1 `normuon-decay0.3-1sqrt-1b-dyn-gns-s1` (`--seed 1`) | `9hg351ju` | 575032 | 32:0.5,64 | 2,860 | 2.9100 | +0.0010 |
| E2c `normuon-decay0.3-1sqrt-ramp128-1b-dyn-gns` | `s2wuxb2e` | 575033 | 32:0.5,64:0.1,128 | 2,478 | 2.9565 | **+0.0475** |
| E2c `…-ramp128-1b-dyn-gns-s1` (`--seed 1`) | `5xvihssq` | 575034 | 32:0.5,64:0.1,128 | 2,478 | 2.9525 | **+0.0435** |

Decision: **E2c rejected** (both seeds 0.04 worse; the seed pairs differ by 0.002 / 0.004). Confounded with the decay: with
the step schedule E2c's 30% decay starts at step 1,735 (455M tokens, batch 32) and has 743 optimizer steps, vs 858 steps
all at batch 64 from 550M tokens in the baseline. E2b not needed: B_noise crosses the batch-64 size at ~0.4-0.5 of the
tokens, inside the known flat optimum 0.33-0.67.

Gradient noise scale (B_simple, tokens; pooled over the 3 runs, identical training up to step 1,735; 90% bootstrap CI):

| Steps | global | embed | attn_v | attn_q | mlp_up | mlp_down |
|---|---|---|---|---|---|---|
| 101-300 | 123k [92, 177] | 101k | 79k | 470k | 187k | 129k |
| 301-700 | 292k [229, 381] | 345k | 146k | 499k | 531k | 320k |
| 701-1100 | 352k [293, 426] | 437k | 174k | 625k | 846k | 446k |
| 1501-1700 | 479k [398, 581] | 543k | 301k | 702k | 1,065k | 610k |
| decay, E1 2101-2350 / 2351-2860 | 1.7M / 3.4M | | | | | |

(1101-1500 omitted: one measurement with G2 ≈ 0 makes the window ratio unstable, CI 0.35-5.5M.) The plan's precision
check fails at 10 measurements per 500-step window in one run (halves disagree > 30% in most stable windows); pooling 3
runs gives usable CIs. Plots and tables: `plots/stageE_e1/`, `plots/stageE_e2c/` (`gns.md`, `phases.md`).

## Follow-up (2026-10-19): E2c' (decay-matched) and E4 (LR x √2 at the ramp)

| Run | W&B | Job | Final eval | Δ vs 2.9090 |
|---|---|---|---|---|
| E2c' `normuon-decay0.192-1sqrt-ramp128-1b-dyn-gns` | `dtvc9be6` | 575799 | 2.9295 | +0.021 |
| E2c' `…-s1` | `1kbw92mb` | 575800 | 2.9323 | +0.023 |
| E4 `normuon-decay0.3-1sqrt-ramplr1.41-1b-dyn-gns` | `9po7x95j` | 575829 | 2.9264 | +0.017 |
| E4 `…-s1` | `zvsqvsxf` | 575830 | 2.9211 | +0.012 |

Decisions: E2c' rejected (half of E2c's loss was the decay confound, the rest is the second ramp); E4 rejected. Keep
`32:0.5,64`, LR unchanged. B_noise jumps ~5x at the decay start (LR-driven), so it cannot time switches inside the decay.

## 3B run with noise measurement (2026-10-20)

`normuon-default-3b-dyn-gns-s1` (W&B `5h5zj9mo`, job 576372): 3B defaults (32:0.5,64, 10% linear decay from step 7,725 =
2.55B tokens), `--seed 1 --dyn_every 25 --dyn_noise`. Final eval **2.7053** (seed 0, job 566892: 2.7075).

B_simple per phase (tokens; ratio of window means, 90% bootstrap CI; batch 32 = 262k, 64 = 524k, 128 = 1.05M tokens):

| Steps | Tokens | B | 90% CI |
|---|---|---|---|
| 1-500 | 0-0.13B | 81k | [61, 125] |
| 501-1000 | 0.13-0.26B | 443k | [336, 600] |
| 1001-2000 | 0.26-0.52B | 565k | [454, 713] |
| 2001-3000 | 0.52-0.79B | 485k | [417, 566] |
| 4001-5000 | 1.05-1.31B | 585k | [385, 924] |
| 6701-7725 (batch 64) | 2.01-2.55B | 498k | [422, 614] |
| decay 7726-8583 | 2.55-3.0B | 1.35M | [1.07, 1.78] |

Windows 3001-4000, 5001-5722 and 5723-6700 are dominated by a few near-zero G2 measurements (CIs reach 3-10M); their
500-step medians of per-point B are 0.42-0.56M. B plateaus at ~0.45-0.55M tokens (≈ batch 64) from ~0.25B tokens to the
decay start and does not grow; it never reaches the batch-128 size (1.05M) in the stable phase.

## 3B run with batch 32 → 64 → 128 (2026-10-20)

`normuon-ramp128-b-3b-dyn-gns-s1` (W&B `wwav5ny2`, job 584171): `--batch_ramp 32:0.1,64:0.4,128 --decay_frac 0.08805`
(4,862 steps; switches at 0.3B / 1.5B tokens; 428-step linear decay from 2.55B tokens, same token as the default),
`--seed 1 --dyn_every 25 --dyn_noise`. Final eval **2.6864** vs 2.7053 (default, seed 1) / 2.7075 (seed 0): **−0.019**.

Eval loss at matched tokens (ramp128 vs default): 0.25B 3.544 vs ~3.58, 0.46B 3.250 vs 3.352, 1.48B 3.009 vs 3.107,
pre-decay 2.84 vs 2.92, final 2.686 vs 2.705. The lead is built before the batch-128 stage (−0.10 at 1.5B tokens) and
shrinks in the decay. Confounds: the switch to 64 at 0.3B (default 1.5B) and the warmup (2% of steps: 97 steps = 25M
tokens vs 171 = 45M).

B_simple: 472k (median) at 0.13-0.30B, 389k at 0.3-0.75B, 468k at 0.75-1.5B, 412k at 1.5-2.55B (batch 128 = 1.05M),
2.1M in the decay. B stays at the batch-64 size while batch 128 trains better: B_simple underestimates the usable batch.

## 3B batch-schedule ablations (2026-10-20, all seed 1, warmup 97 steps except the default, decay from 2.55B tokens)

| Run | W&B | Schedule (tokens) | Decay | Pre-decay eval | Final eval |
|---|---|---|---|---|---|
| default `normuon-default-3b-dyn-gns-s1` | `5h5zj9mo` | 32 → 64 @1.5B | 858 steps @64 | 2.917 (2.55B) | 2.7053 |
| `normuon-ramp64e-3b-dyn-gns-s1` (585920) | `41dergdv` | 32 → 64 @0.3B | 857 @64 | 2.931 (2.50B) | 2.7160 |
| `normuon-ramp128-decay64-3b-dyn-gns-s1` (586953, branch @1.5B of 585920) | `l1d17eee` | 32 → 64 @0.3B → 128 @1.5B → 64 @2.55B | 857 @64 | 2.838 (2.51B) | 2.7052 |
| `normuon-ramp128-b-3b-dyn-gns-s1` (584171) | `wwav5ny2` | 32 → 64 @0.3B → 128 @1.5B | 428 @128 | 2.839 (2.49B) | **2.6864** |
| `normuon-ramp128-decay256-3b-dyn-gns-s1` (587391, branch @2.55B of 586953) | `ip3bpegm` | … → 128 @1.5B → 256 @2.55B | 214 @256 | 2.838 | **2.6849** |
| `normuon-ramp128at1b-3b-dyn-gns-s1` (587392, branch @1.0B of 585920) | `vupvh2vw` | 32 → 64 @0.3B → 128 @1.0B | 429 @128 | 2.856 (2.35B) | 2.6917 |
| `normuon-ramp64e-decay32-3b-dyn-gns-s1` (585921) | | → 32 @2.55B | 1,715 @32 | | cancelled (decision) |

- The early switch to 64 alone is worse (+0.011 vs default): its lead at 1B tokens (3.073 vs 3.162) is gone by the decay.
- Batch 128 in the stable phase (1.5-2.55B): −0.011 at the end (2.7052 vs 2.7160), −0.09 before the decay.
- Batch 128 in the decay instead of 64: −0.019 (2.6864 vs 2.7052) with half the decay steps. Same pre-decay state
  (2.838 / 2.839). More decay steps at a smaller batch hurt; consistent with B_simple rising to 1-2M in the decay.
- Checkpoints: `/work/$USER/llama_ckpt/normuon-ramp64e-3b-dyn-gns-s1/` (0.3B, 1.0B, 1.5B, 2.0B, 2.55B) and
  `/work/$USER/llama_ckpt/normuon-ramp128-decay64-3b-dyn-gns-s1/` (2.0B, 2.55B; the 2.55B one is also the decay start
  of the 128-decay run).
- Decay batch at the same pre-decay state: 64 → 2.7052, 128 → 2.6864, 256 → 2.6849. 128 → 256 is within seed noise
  (~0.002): larger is better up to 128, flat after. 256 needs `--per_device_batch 4` x accum 8; a resumed branch may
  now have earlier stages that do not split evenly (`plan.py`: only before the resume step).
- 128 from 1.0B instead of 1.5B: +0.005 (2.6917 vs 2.6864). The switch at 1.5B (0.5 of the budget) is at least as good.
- Checkpoints of `normuon-ramp128at1b-3b-dyn-gns-s1`: 2.0B, 2.55B (decay start).
