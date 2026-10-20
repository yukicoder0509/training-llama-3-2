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
