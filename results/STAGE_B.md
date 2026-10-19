# Stage B — does switching NorMuon → AdamW help? (6b scale)

Setup: 174M proxy (`train_scorer.py`), 6b config: batch 256 x 2048 (524,288 tokens / step), 6B tokens, 11,444 steps,
WSD decay over the last 1,144 steps (from step ~10,300), norm wd on, `--dyn_every 50`. W&B: personal workspace
(`llama3.2-arch-small-scoring-model`). Eval losses at W&B precision. NorMuon replaces the plan's Muon (user decision).

## B1. Baselines with checkpoints (steps 3000 / 5500 / 8000)

| Run | Job | W&B | Flags | Final eval |
|---|---|---|---|---|
| `proxy-dyn-normuon-6b-ckpt` | 567242 | 89gb67q3 | NorMuon 0.01 + AdamW 3e-3, default seed | **2.98197** |
| `proxy-dyn-normuon-6b-s1` | 567244 | fdq92ve2 | same, `--seed=1`, no checkpoints | 2.98527 |
| `proxy-dyn-adamw-normwd-6b-ckpt` | 567243 | ope22mfm | AdamW 1e-3, `--norm_wd` | 3.01669 |

- **6b seed noise (NorMuon): 0.0033.** AdamW's same-seed repeat of 563002 (3.014): 0.003. For B4 the plan's threshold
  would therefore be ≥ 0.005 (the floor the plan sets when B1 gives less).
- NorMuon at 6b (2-seed mean 2.9836) ties Muon (2.983, 561805, one seed): no NorMuon gain with more tokens either
  (this is the Stage C question; Stage C's NorMuon run was replaced by this second seed).
- Checkpoint integrity: each branch's first eval matches the main run at the same step (e.g. step 8008: NorMuon 3.1770
  vs 3.1770, AdamW 3.1555 vs 3.1557).

## B2. Decay branches (1,144 linear decay steps from S, Stage 0 recipe)

| Base | S | Job | W&B | Stable eval at S | R_opt(S) = branch final | floor = stable − R |
|---|---|---|---|---|---|---|
| NorMuon | 3000 | 567625 | x11l8s5y | 3.2785 | 3.0726 | 0.206 |
| NorMuon | 5500 | 567626 | x15v2e7l | 3.2076 | 3.0216 | 0.186 |
| NorMuon | 8000 | 567627 | unc4uj37 | 3.1771 | 2.9959 | 0.181 |
| NorMuon | 10300 (full run) | 567242 | 89gb67q3 | 3.1647 | 2.9820 | 0.183 |
| AdamW | 3000 | 567628 | 7x6c7u93 | 3.3356 | 3.1660 | 0.170 |
| AdamW | 5500 | 567629 | i1ddei3n | 3.2125 | 3.0841 | 0.128 |
| AdamW | 8000 | 567630 | ugixlagy | 3.1559 | 3.0408 | 0.115 |
| AdamW | 10300 (full run) | 567243 | ope22mfm | 3.1263 | 3.0167 | 0.110 |

Stable eval at S is interpolated linearly between the evals every 572 steps.

## B3. Analysis

**The noise floor is not constant.** It shrinks with S for both optimizers, much more for AdamW (0.170 → 0.110, −35%)
than for NorMuon (0.206 → 0.183, −11%, flat after 5500). NorMuon's floor is 0.07 larger late in training, consistent
with its stable-phase oscillation (2026-10-17): more of its loss is "noise" that the decay removes. Comparing stable
evals at matched step therefore understates NorMuon: at step 8000 AdamW's stable eval is 0.021 *lower* (3.1559 vs
3.1771), but its decayed loss is 0.045 *higher*.

**Rates at matched R.** R(S) fitted per optimizer as R = a + b (S/1000)^−c (4 points, residuals ≤ 0.0007):
NorMuon a 2.869, b 0.345, c 0.481; AdamW a 2.770, b 0.605, c 0.385. In the R range both reach (3.0167 … 3.0726):

| R | NorMuon step | NorMuon −dR/dS | AdamW step | AdamW −dR/dS | AdamW / NorMuon |
|---|---|---|---|---|---|
| 3.0726 | 3,001 | 32.6e-6 | 6,034 | 19.3e-6 | 0.59 |
| 3.0615 | 3,377 | 27.3e-6 | 6,654 | 16.9e-6 | 0.62 |
| 3.0503 | 3,826 | 22.7e-6 | 7,366 | 14.7e-6 | 0.64 |
| 3.0391 | 4,370 | 18.7e-6 | 8,189 | 12.7e-6 | 0.68 |
| 3.0279 | 5,036 | 15.1e-6 | 9,144 | 10.9e-6 | 0.72 |
| 3.0167 | 5,865 | 12.1e-6 | 10,261 | 9.3e-6 | 0.77 |

Fit-free check (piecewise slopes between branch points, per step): NorMuon 20.4e-6 (3000-5500), 10.3e-6 (5500-8000),
6.0e-6 (8000-10300); AdamW 32.8e-6, 17.3e-6, 10.5e-6. AdamW's 5500-8000 segment (R 3.084 → 3.041) overlaps NorMuon's
3000-5500 segment (3.073 → 3.022): 17.3e-6 vs 20.4e-6, the same direction.

- **At every matched R, NorMuon is faster: AdamW's rate is 0.59-0.77x NorMuon's.** The ratio rises as R falls, but
  never reaches 1 inside the overlap, let alone 1.2.
- Extrapolating both fits beyond the data (not evidence, only a direction): equal rates at R ≈ 2.98, where NorMuon is
  at ~10,600 steps (its 6B end); AdamW 20% faster only below R ≈ 2.97, i.e. past the 6B budget.

## Decision

Rule: switch if AdamW's −dR/dS exceeds NorMuon's by ≥ 20% at some R* and stays higher below it. **Not met** (max
ratio 0.77). Conclusion: **no switch benefit at this budget → skip B4** (conditional; awaiting the user's confirmation).

Budget used: 2 full 6b runs + 6 branches (+ the NorMuon 6b second seed, counted under Stage C).
