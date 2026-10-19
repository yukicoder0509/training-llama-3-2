# Stage 0 — implementation tasks (EXPERIMENT_PLAN.md)

Decisions (2026-10-18): no probe metrics for now (`--dyn_probe` dropped; gates use D1 and `grad_cos_prev`); NorMuon
wherever the plan says Muon; existing flags wherever they already do the job. All tests on the 174M proxy via Slurm.

## Changes

| Task | Implementation | Flags |
|---|---|---|
| T1 per-block muonium options | `llama_train/optim.py`: `muon_head_groups()` (q / k / v groups, muonium `split_sizes` = head_dim row blocks, k / v LR multiplier), `param_group_table()` printed at startup and stored as `args.param_groups` (in the W&B config with the other args) | new: `--muon_split_heads {none,kv,qkv}`, `--muon_lr_mult_kv`, `--norm_lr` |
| T2 SOAP on the tied embedding | existing `ChainedOptimizer` (concatenated groups, steps both, nested state_dict) + `DistributedSOAP` for the embedding; norms stay in muonium's AdamW (`--norm_lr`) | new: `--embed_optimizer {adamw,soap}`; SOAP LR = `--learning_rate`, other SOAP settings from `--soap_*` |
| T3 checkpoints and decay branches | **no new code**: `--save_at_tokens` (S x tokens/step = exactly step S at a fixed batch) + `--resume_from` with `--token_budget=(S+N) x tokens/step --decay_frac=(N+0.5)/(S+N)` (the +0.5 makes int() give exactly N decay steps) | existing |
| T4 switch at a step | deferred (conditional on Stage B). LR-step control: existing `--lr_schedule wsd2` steps down **all** groups; a NorMuon-only step-down would be new code | — |
| T5 analysis | `analyze_dynamics.py` extended (reads the JSONL, no W&B pulls): `phases.md` with medians over step windows (`--gate 500:1500`, stable = `--stable_from` … decay start − 50, decay), overshoot D1/grad_norm, fraction of D1 > 0, deltas vs `--baseline`, eval table from `--logs NAME=slurm.out` (final, pre-decay, decay gain, Δ flagged under `--noise`). No dir_sharpness / curvature_K | new analysis flags |

## Tests

**T1 / T2 param groups** (`tests/test_param_groups.py`, job 566778, 1 GPU, ALL OK):

| Flags | Groups (index: algorithm, lr, split, tensors) |
|---|---|
| defaults (= 566231) | 0 normuon 0.01 – 84 · 1 adamw 3e-3 embed · 2 adamw 3e-3 norms 25 (wd 0.1); matches 566231's lr/group0 = 0.01, group1 = group2 = 0.003 |
| `--muon_split_heads kv` | 0 normuon 60 · 1 normuon 0.01 (64,)x4 k_proj+v_proj 24 · 2 embed · 3 norms |
| `--muon_split_heads qkv` | 0 normuon 48 · 1 q_proj (64,)x12 · 2 k_proj+v_proj (64,)x4 · 3 embed · 4 norms |
| `--muon_lr_mult_kv 0.5` | 1 k_proj+v_proj at 0.005, no split |
| `--norm_lr 1e-3` | embed 3e-3, norms 1e-3 |
| `--embed_optimizer soap --learning_rate 2.8e-3 --norm_lr 3e-3` | 0 normuon · 1 adamw 3e-3 norms · 2 SOAP 2.8e-3 embed (wd 0.1); embedding moves after SOAP's init step; state_dict nests both |

Each flag set also took an optimizer step (muonium accepted the groups). Head split mechanism (plain Muon, wd 0, first
step, k_proj of layer 0, heads given correlated gradients = shared component + noise): cosine of each head's update
block with the exact polar factor of that head's gradient block = **0.996 with the split vs 0.786 without**. (A first
version with i.i.d. random gradients could not separate them, 0.995 vs 0.962: random rows are already near-orthogonal.)

**T3 decay branches** (`tests/stage0_ckpt_tests.sh`, batch 128, W&B off). Main `s0-t3-main` (566701): 400 steps, WSD
decay over steps 361-400, checkpoints at 200 and 360.

| Run | Final eval | Train loss 370 / 380 / 390 / 400 | LR at 370 / 380 / 390 / 400 |
|---|---|---|---|
| main (566701) | 4.025 | 4.088 / 4.094 / 4.037 / 3.997 | 7.75e-3 / 5.25e-3 / 2.75e-3 / 2.5e-4 |
| branch from 360, N = 40 (566775) | **4.025** | 4.088 / 4.094 / 4.037 / 3.997 | identical |

- Pass: the branch from the decay start reproduces the main run exactly (eval and logged losses to 3 decimals; plan
  tolerance 0.003). The resumed scheduler follows the branch's own flags.
- Branch from 200, N = 40 (566776): LR 0.01 at 200, then 7.75e-3 / 5.25e-3 / 2.75e-3 / 2.5e-4 at 210 / 220 / 230 / 240,
  the same linear shape as the main run's decay; final eval 4.602. Pass.

**T2 save → load → continue** (`--embed_optimizer soap`, 60 steps, checkpoint at 20):

| Run | Train loss 30 / 40 / 50 / 60 | Final eval |
|---|---|---|
| straight (566702) | 7.397 / 6.817 / 6.506 / 6.306 | 6.226 |
| resumed from 20 (566777) | 7.397 / 6.817 / 6.506 / 6.306 | 6.226 |

Pass: identical (ChainedOptimizer state with muonium + SOAP restores fully).

**T5 analysis** (job after a fix: `--logs` is now repeatable, one `--logs NAME=log` per run; with `nargs="*"` it swallowed
the positional runs). On NorMuon 566231 vs Muon 565026 / 565032 (`--max_steps 3814 --baseline NorMuon`):
- Eval table: final 3.239 / 3.240 / 3.244, pre-decay (step 3420, decay starts 3434) 3.403 / 3.404 / 3.403, decay gain
  0.164 / 0.164 / 0.159; Muon +0.001 flagged "< noise". Matches the logs.
- Overshoot (stable): k / v 0.045 / 0.053 (NorMuon), 0.044 / 0.058 (Muon), the values computed by hand from the JSONL;
  D1 > 0 at every gate / stable measurement in all three runs, 0.75 of them in the decay.

## Branch recipe for Stage B (6b, 524,288 tokens/step, N = 1144)

```
--resume_from /work/$USER/llama_ckpt/<base>/checkpoint-<S> --token_budget=$(( (S + 1144) * 524288 ))
--decay_frac=$(python3 -c "print(1144.5 / (S + 1144))") --run_name=<base>-branch<S>
```
Main runs: `--save_at_tokens=$((3000*524288)),$((5500*524288)),$((8000*524288))`.

## Not done

- Probe metrics (dropped for now). T4 (conditional).
