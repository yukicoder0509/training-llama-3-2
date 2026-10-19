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
