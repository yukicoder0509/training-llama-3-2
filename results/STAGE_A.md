# Stage A — per-block candidates (1b scale, NorMuon baseline)

Setup: 174M proxy (`train_scorer.py`), NorMuon 0.01 + AdamW 3e-3 (embedding, norms), norm wd on, batch 128 x 2048,
1B tokens (3,814 steps, decay from 3,434), `--dyn_every 50`, no probe. W&B: personal workspace
(`llama3.2-arch-small-scoring-model`). Analysis: `analyze_dynamics.py --baseline NorMuon` → `plots/stageA/phases.md`
(gate window = steps 500-1500; overshoot = D1 / grad_norm).

Baseline (2 seeds): 566231 3.239, 566904 (`--seed=1`) 3.237 → **mean 3.238, better seed 3.237** (the plan's
3.2420 / 3.2403 were plain-Muon numbers). Eval losses are logged to 3 decimals.

## Runs

| Run | Job | Flags vs baseline | Final eval | Pre-decay (step 3420) | Decay gain | Δ vs mean 3.238 |
|---|---|---|---|---|---|---|
| A0 `proxy-dyn-normuon-1b-bs128` | 566231 | — | 3.239 | 3.403 | 0.164 | |
| A0 `proxy-dyn-normuon-1b-bs128-s1` | 566904 | `--seed=1` | 3.237 | 3.399 | 0.162 | |
| A1 `proxy-dyn-normuon-splitkv-1b-bs128` | 566905 | `--muon_split_heads=kv` | 3.239 | 3.401 | 0.162 | +0.001 |
| A1b `proxy-dyn-normuon-splitqkv-1b-bs128` | 566906 | `--muon_split_heads=qkv` | **3.234** | 3.395 | 0.161 | **−0.004** |
| A2 `proxy-dyn-normuon-kv0.5-1b-bs128` | 566907 | `--muon_lr_mult_kv=0.5` | 3.237 | 3.395 | 0.158 | −0.001 |
| A3 `proxy-dyn-normuon-normlr1e-3-1b-bs128` | 566908 | `--norm_lr=1e-3` | 3.245 | 3.411 | 0.166 | +0.007 |
| A4 `proxy-dyn-normuon-soapembed-1b-bs128-v2` | 567070 | `--embed_optimizer=soap --learning_rate=2.8e-3 --norm_lr=3e-3` | 3.240 | 3.396 | 0.156 | +0.002 |

A4's first run (566909, 3.380) is invalid: see "Bug" below.

## Gates (medians over steps 500-1500; baseline = mean of the two seeds)

| Run | Gate | Measured | Result |
|---|---|---|---|
| A1 | k / v overshoot ≥ 30% lower **and** k / v grad_cos_prev less negative | overshoot k 0.0556 vs 0.0487 (+14%), v 0.104 vs 0.0512 (+103%); cos k −0.28 vs −0.44, v −0.55 vs −0.71 | **fail** (overshoot up) |
| A1b | as A1, also q | overshoot k +19%, v +107%, q 0.0357 vs 0.0199 (+79%); cos k −0.32, v −0.52, q −0.30 vs −0.56 | **fail** (overshoot up) |
| A2 | as A1 | overshoot k 0.0452 (−7%), v 0.0527 (+3%); cos k −0.38, v −0.69 | **fail** |
| A3 | norm grad_cos_prev ≥ −0.4 | −0.77 vs −0.71 | **fail** |
| A4 | embed w_srank ≥ 8 and upd_erank_frac ≥ 0.8 | w_srank 12.4 vs 2.7, erank 0.85 vs 0.53 | **pass** |

### Correction (2026-10-18, from the user's review): the overshoot gate was mis-specified

D1 / grad_norm is cos(next gradient, update), so it also rises when the update follows −g more closely. Per-head
orthogonalization does exactly that: upd_grad_align 0.089 → 0.163 globally, 0.137 → 0.313 for v. Normalized by the
alignment (overshoot_norm = D1 / grad_norm / upd_grad_align, now in `phases.md`), splitqkv **lowers** the overshoot
(steps 500-1500; baseline seeds 566231 / 566904 → 566906):

| Block | overshoot_norm | grad_cos_prev |
|---|---|---|
| global | 0.133 / 0.137 → 0.109 | −0.64 / −0.64 → −0.41 |
| attn_q | 0.118 / 0.125 → 0.098 | −0.47 / −0.65 → −0.29 |
| attn_k | 0.195 / 0.201 → 0.167 | −0.39 / −0.50 → −0.32 |
| attn_v | 0.383 / 0.380 → 0.340 | −0.71 / −0.72 → −0.52 |
| mlp_up (not split) | 0.137 / 0.136 → 0.093 | −0.63 / −0.63 → −0.25 |

- A1b achieved its mechanism, more than any other run, and has the best loss: **3.2339** at W&B precision (−0.0044 vs
  the NorMuon mean 3.2383).
- Together the runs say **oscillation is a whole-network property**: lowering one block's LR (kv0.5, norm_lr) leaves
  that block's own oscillation unchanged, while splitting attention heads reduces it in MLP blocks it never touches.
  Per-block gates mislead; future gates use global metrics plus alignment-normalized overshoot for the targeted block.
- Revised decision: A1b second seed (567269, `--seed=1`). Accept if the 2-seed mean ≤ 3.2343 and both seeds < 3.2372;
  if accepted, add `--muon_split_heads qkv` to the 6B runs in Stage C. A1, A2, A3, A4 closed (neutral losses; norm_lr
  worse, +0.007).

## Decisions (plan procedure, before the correction)

- A1, A2, A3: **mechanism not achieved** → stop.
- A1b: **mechanism not achieved** by the plan's gate → stop per the plan. But its loss is the only candidate inside the
  step-3 window (−0.004 vs the mean, beats the better seed by 0.003). Second seed pending the user's call.
- A4: **mechanism achieved, no loss effect** (Δ +0.002; the step-3 threshold is ≤ −0.002) → no second seed. Lower
  pre-decay eval (3.396 vs 3.401) but a smaller decay gain (0.156 vs 0.163).

## Observations

- **Head splitting moves the two gate metrics in opposite directions.** Oscillation falls in every attention block and
  globally (grad_cos_prev −0.64 → −0.44 / −0.41), but the overshoot D1 / grad_norm of v doubles. upd_rel barely
  changes (k 0.0080 vs 0.0082, v 0.0068 vs 0.0071), so the change is in the update's direction, not its size. Why v
  in particular is not established.
- **A smaller k / v LR (A2) does not reduce the overshoot** (−7% / +3% at 0.5x LR, upd_rel −26% / −21%): the k / v
  overshoot is not a step-size effect.
- **A3 (norm LR 1e-3)**: the norm-gain step shrinks 3.2x (upd_rel 0.00022 vs 0.00071) but the oscillation does not
  change, consistent with finding 1 (it comes from the NorMuon-updated matrices). Loss +0.007.
- **A4**: SOAP gives the tied embedding a ~4.5x higher weight stable rank (12.4 vs 2.7) and higher update rank, with
  half the overshoot, but no loss change.

## Bug found and fixed during Stage A

`ChainedOptimizer` (used by `--embed_optimizer soap` and `--muonium_rest sophia`) trained at a **constant LR** — no
warmup, no decay — while logging the correct schedule. accelerate's `prepare()` round-trips the optimizer through
`load_state_dict` (device placement); torch then gives the inner optimizers new param-group dicts, while
`ChainedOptimizer.param_groups` kept the old ones, which the LR scheduler and the dynamics logger kept updating.
Evidence in 566909: mlp_up upd_rel 0.0071 at steps 3400 / 3550 / 3700 / 3800 vs the baseline's 0.0072 → 0.0003.

- Fix: `ChainedOptimizer.load_state_dict` relinks `param_groups` to the inner optimizers' dicts.
- Test added to `tests/test_param_groups.py`: wrap in accelerate's `AcceleratedOptimizer`, apply a LambdaLR factor 0.5,
  check that SOAP and muonium see it (job 567050, ALL OK). The rerun 567070 warms up like the baseline (mlp_up upd_rel
  0.0098 vs 0.0094 at step 50).
- Also invalid: the 1B-model `muonium-normuon-sophia-1b` (2026-10-15, 3.159), marked in experiments.md.
- Stage 0's T2 round-trip test passed because both sides of it had the same constant LR.

## Not run

Second seeds (none qualified under the plan; A1b pending decision). Stage A budget used: 7 of 10 runs (6 + A4 rerun).

## A1b second seed: rejected

| Run | W&B | Final eval (W&B precision) |
|---|---|---|
| NorMuon (566231) | 8uspqif8 | 3.23933 |
| NorMuon `--seed=1` (566904) | suvjvcjl | 3.23725 |
| splitqkv (566906) | b8hwuuy2 | 3.23392 |
| splitqkv `--seed=1` (567269) | 6plctb32 | **3.24032** |

- splitqkv 2-seed mean 3.23712 vs NorMuon 3.23829: **−0.0012**. Rule (mean ≤ 3.2343 and both seeds < 3.2372): **fails
  both parts** (seed 2 is the worst of the four). The first seed's −0.0044 was seed luck; the split seeds differ by
  0.0064, about 3x the baseline seeds' 0.0021.
- The mechanism reproduces in seed 2 (steps 500-1500): global overshoot_norm 0.116 (0.109 in seed 1, baseline
  0.133-0.137), grad_cos_prev −0.43 (−0.41; baseline −0.64); attn_v 0.349 / −0.54; mlp_up 0.095 / −0.31.
- Result: **mechanism achieved, no loss effect** (as NorMuon's row normalization). Less oscillation across the whole
  network does not translate into lower loss at this scale. Stage C does not get `--muon_split_heads qkv`.
