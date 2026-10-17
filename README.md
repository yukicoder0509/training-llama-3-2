# Llama 3.2 1B on Dolma 3 (Lab 5)

Train the Llama 3.2 1B architecture from scratch on `allenai/dolma3_mix-150B-1025` with a fixed token budget
(default 6B tokens, 8192-token context). Goal and rules: [`DESCRIPTION.md`](DESCRIPTION.md). Results: [`experiments.md`](experiments.md).

# Quick Start
1. `source .venv/bin/activate`
2. Log in to Hugging Face and W&B
3. Create `.env` with `MODEL_NAME=` (the Hub repo `push_model.py` pushes to)

Run heavy work (data prep, training) through Slurm, not on the login node.

## 1. Prepare data

```
sbatch prepare.sbatch
```

Downloads the dataset (~110 GB, pinned revision) and writes to `$OUT_DIR` (default `/work/$USER/dolma3_llama`):

- The OJ split, reproduced exactly: `load_dataset(...)["train"].shuffle(seed=42)`, last 50,000 docs held out. Computed from
  per-file doc counts (`doc_counts.json`) instead of building the ~400 GB arrow dataset.
- `train.bin`: the first 5% of the shuffled train docs (`--train_frac`), about 7.5B tokens. This is a uniform sample of
  the mix, and there is enough to cover 6B tokens without repeating any.
- `val.bin`: the first 10,000 held-out docs (20%, `--num_val_docs`), used for in-training eval.
- `eval_docs.jsonl`: all 50k held-out texts, for exact OJ-style evaluation later. Also `meta.json` with the counts.

Tokens are uint32 (Llama 3 vocab 128,256), one doc = `<|begin_of_text|> text <|end_of_text|>`, packed into 8192 blocks.
The tokenizer is `NousResearch/Llama-3.2-1B` (its `tokenizer.json` is byte-identical to the gated `meta-llama/Llama-3.2-1B`). By default the data is not cleaned.

Optional cleaning (train docs only; the eval split, `val.bin` and `eval_docs.jsonl` stay byte-identical). Write each
variant to its own `OUT_DIR` and train on it with `--data_dir`:

```
OUT_DIR=/work/$USER/dolma3_llama_dedup sbatch prepare.sbatch --dedup --min_train_tokens=7e9
OUT_DIR=/work/$USER/dolma3_llama_qc1e-3 sbatch prepare.sbatch --qc_max=0.001 --train_frac=0.11
```

- `--dedup`: exact dedup over the whole dataset. Dolma 3 upsamples by repetition (104M docs are only 35.8M distinct
  texts, 40% of the characters). Key per doc: `metadata.text_hash` (common_crawl), else a 64-bit blake2b of the text;
  keys are cached in `dedup_keys.npz`. Keeps one copy per text and drops every copy of a text that also appears in the
  eval split. The train sample is drawn uniformly over the unique texts (not a prefix of the seed-42 shuffle, which
  would keep the upsampling: a text with k copies would land in it with probability 1 − 0.95^k).
  - `--min_train_tokens`: with `--dedup`, take random unique texts until they hold about this many tokens (estimated
    from a tokenized sample), instead of `--train_frac` of them.
- `--qc_max=t`: drop train docs with `metadata.dolma2_qc["1"] >= t` (a fastText quality score). Only common_crawl docs
  carry it; the other sources are always kept. Writes `qc_stats.json` per source dir. Which label is "good" is
  unresolved (experiments.md 2026-10-13); pick `t` with `qc_scan.sbatch` + `plot_qc.py` (score distribution, data
  left per threshold). `0.001` keeps 34% of the bytes (6.25B tokens at `--train_frac=0.11`).

`meta.json` records the options and the dedup / qc counts. Dedup at 1B tokens: 3.371 vs 3.387, within noise
(experiments.md 2026-10-14); the qc-filtered data has not been trained on yet.
(`dedup_filter.py` / `dedup_filter.sbatch` are the old C4 pipelines, not yet ported.)

## 2. Train

```
sbatch run.sbatch --run_name=my-run                     # 8 H200 x 8 h max (= the 64 H200-hour cap), 6B tokens
sbatch --gpus-per-node=2 --cpus-per-task=24 run.sbatch --run_name=baseline --token_budget=3e9   # ~16 H200-hours
sbatch profile.sbatch --per_device_batch=8              # 20-step speed / memory test, no W&B
```

Measured on 8 H200 (job 514647, AdamW, compile + fused CE, 4 x 8192 tokens/GPU x accum 2, DDP 200 MB buckets without
find_unused): **432k tokens/s (54.0k/GPU), 69 GiB peak**, so 6B tokens take ~3.9 h (~31 H200-hours). Muon is ~4% slower,
Adam-mini about the same. Flag and DDP sweeps: experiments.md 2026-10-08.

| Flag | Default | Meaning |
|---|---|---|
| `--token_budget` | `6e9` | Training tokens; steps = budget / (global batch x 8192). No padding, so every token counts |
| `--global_batch_size` | `64` | Sequences per step (64 x 8192 = 0.5M tokens). Giving it turns off the default ramp |
| `--per_device_batch` | `4` | Max sequences per GPU per micro-batch; grad accum covers the rest |
| `--batch_ramp` | `32:0.5,64` | `none` for a fixed batch. `"B1:t1,...,Bn"`: batch Bi for the next ti tokens, the last for the rest. ti > 1 is a token count (`32:1e9,64`, or `32:1B,64` / `32:333M,64`): the switch is at the same tokens for any budget. ti ≤ 1 is a fraction of the budget (`32:0.3333,64`: 333M of 1B, 1B of 3B). Switches at the last whole step within the stage; LRs and the LR schedule are unchanged. Overrides `--global_batch_size` |
| `--learning_rate` / `--weight_decay` / `--beta2` | `3e-3` (muonium), else `6e-4` / `0.1` / `0.95` | Peak LR; with muon / muonium the AdamW (embedding, norms) part |
| `--warmup_frac` / `--decay_frac` | `0.02` / `0.1` | Linear warmup, then WSD with linear decay to 0 over the final fraction (0.1 as in MiniCPM; runs before 2026-10-14 used 0.2) |
| `--lr_schedule` / `--min_lr_ratio` | `wsd` / `0.1` | `cosine`: cosine from the peak to `min_lr_ratio` × peak (Llama 2/3). `wsd2`: warmup, stable, decay to `--stable2_lr_ratio` × peak over `--decay1_frac`, stable for `--stable2_frac`, then decay to 0 over `--decay_frac` |
| `--seed` | none | Init, data order and Sophia's label sampling (repeat-seed runs) |
| `--schedule_unit` | `steps` | `tokens`: WSD warmup / decay as fractions of `--token_budget` tokens, so the LR at a token count doesn't depend on the batch ramp (needed to branch runs from a checkpoint) |
| `--save_at_tokens` / `--ckpt_dir` | none / `/work/$USER/llama_ckpt/<run_name>` | Full checkpoints (model, optimizer, scheduler, RNG, step) at the last step at or before each token count (`100M,333M`), linked as `tokens-<count>` |
| `--resume_from` | none | Continue from a checkpoint with this run's own flags (data order and LR continue where it stopped) |
| `--adam_eps` | `1e-8` | Adam ε (Llama 2: 1e-5) |
| `--optimizer` | `muonium` | `adamw` (fused torch AdamW), `adam_mini`, `muon` (Muon for decoder-layer matrices + AdamW for embeddings / norms, `muon_adamw.py`), `muonium` (pip `muonium`, see below), `sophia` (SophiaG, `sophia.py` vendored from the official repo) or `soap` (see below); LR = `--learning_rate` |
| `--muon_lr` / `--muon_momentum` | `0.01` (muonium), `1.25e-3` (muon) / `0.95` | Muon LR for the matrices; `--learning_rate` is then the AdamW (embedding, norms) LR. `muonium` scales by sqrt(max(1, rows/cols)): best 0.01 (experiments.md 2026-10-15) |
| `--muonium_algorithm` / `--muonium_rest` / `--muon_wd` | `normuon` / `adamw` / `--weight_decay` | `--optimizer muonium`: `normuon` adds per-neuron second-moment normalization; `--muonium_rest sophia` puts SophiaG on the embedding + norms |
| `--norm_wd` / `--no-norm_wd` | on for muonium (AdamW rest), else off | Weight decay on the RMSNorm weights too (adamw, muonium, sophia, soap). Off is the HF Trainer split (matrices + embedding); muonium's AdamW group decayed everything, so on keeps the old runs' behavior |
| `--soap_impl` / `--soap_betas` / `--soap_precond_freq` | `official` / `0.95,0.95` / `10` | SOAP (`--optimizer soap`, ε = `--adam_eps`). `official`: `soap.py` (paper authors) with the work split across GPUs by `soap_dist.py`. `meta`: facebookresearch `distributed_shampoo` (`--soap_block`, `--soap_comm_dtype`), no faster at the same preconditioner. +31% step time at 0.5M batch, +7% at 2M |
| `--sophia_rho` / `--sophia_betas` / `--sophia_hess_interval` | `0.05` / `0.965,0.99` / `10` | SophiaG (official GPT-2 values). The Hessian pass every k steps reruns the step's batch with sampled labels: +1/k compute. Use weight decay ~2x AdamW's (0.2). Watch `train/sophia_win_rate` (official guidance: 0.1–0.5) |
| `--ddp_find_unused` / `--ddp_bucket_cap_mb` / `--ddp_bf16_grads` | off / `200` / off | DDP settings (+6% vs Trainer's defaults of on / 25 MB); bf16 all-reduce gives no gain with large buckets |
| `--num_evals` | `20` | Evals over the run (lab: at least every 10% of steps) plus a final eval |
| `--fused_ce` / `--torch_compile` | on in the sbatch scripts | Liger fused LM head + CE (no 4 GiB/seq fp32 logits); `torch.compile` per decoder layer |
| `--attn_implementation` | `flash_attention_3` | `flash_attention_3` (Hub kernel via `kernels`), `sdpa`, `flash_attention_4` |
| `--packing` | `doc` | `doc`: each document attends only to itself, positions restart per document (FA varlen; ~10% faster). `concat`: attention crosses document boundaries (all runs before 2026-10-14). Evals use the same mode; compare across modes with `eval_packing.py` |
| `--profile_dir` / `--profile_start` / `--profile_cpu` | off / `8` / on | torch.profiler over 5 steps |
| `--data_dir` | `/work/$USER/dolma3_llama` | `train.bin` / `val.bin`, plus `dev.bin` if present (our own dev set, logged as `eval/dev_loss`; see Data selection) |
| `--block_size` / `--config_overrides` | `8192` / none | Context; JSON applied to the Llama 3.2 1B config (`train_scorer.py` uses both) |
| `--save_dir` | `/work/$USER/llama_models/<run_name>` | Final model + tokenizer |
| `--dyn_every` | `0` (off) | Optimizer dynamics (`llama_train/optim_dynamics.py`): per block type, layer and globally, every N steps over a 2-step window: gradient norm / cosine, update size / cosine / alignment with −g, directional sharpness and curvature, σ1, stable and effective rank of W, dW and the accumulated change. Rank 0; ~5 s per measurement on the proxy. Metric definitions: "Optimizer dynamics metrics" below |
| `--dyn_spectral_every` | `--dyn_every` | Spectral / effective-rank metrics every N steps (full SVD per matrix) |
| `--dyn_erank`, `--dyn_track_delta` | on | Effective rank of dW; effective rank of W_t − W_(t−N) (keeps one copy of the matrices) |
| `--dyn_out` | `<save_dir>/optim_dynamics.jsonl` | One JSON line per measurement; W&B gets `dyn_<metric>/<block type>` on train/global_step |
| `--dyn_heatmap_every` | `10` | W&B (layer x block type) heatmap images every N measurements (0: none) |

Runs are bounded by the token budget: the step count comes from `--token_budget`, and a run ends when those steps are
done. The Slurm time limit is only a safety net. Training stops 5 min before it (read from `squeue`) and still saves
the model, but that run saw fewer tokens and is not comparable. At step 50 the log prints the projected finish (or a
WARNING if it is past the deadline), and the W&B summary records `budget_completed` and `final_tokens_seen`. Give
comparison runs enough `--time` (e.g. 1:15 for a 1.1B-token run, which takes ~50 min).

W&B: `cerulean-labs/lab5-training-llama` on `https://app.forge.coreweave.com`. The lab's required metrics are
`train/loss` (mean over the 10 logging steps), `train/grad_norm`, `train/learning_rate`, `train/tokens_per_second`
(of that step), `train/total_tokens_seen`, and `eval/perplexity`. Extras: `train/mfu`, `train/ppl`, `eval/loss`, and
`eval/dev_loss` / `eval/dev_perplexity` when the data dir has a `dev.bin`.
All train and eval charts use `train/total_tokens_seen` as their x-axis (eval rows carry that key too). To plot runs
against tokens locally (login node, reads W&B):

```bash
python plot_runs.py dvqj42sv dkyqsbe6 --out plots/wsd_vs_cosine_1h.png   # train/eval loss and ppl vs tokens seen
```

The defaults are the best run so far, `muonium-normuon0.01-adamw3e-3-ramp-3b` (doc eval ppl 15.1 at 3B tokens,
experiments.md 2026-10-16), except the batch switch: NorMuon LR 0.01 on the decoder-layer matrices, AdamW 3e-3 on the embedding + norms, wd 0.1,
WSD 2% / 10%, doc packing, batch 32 for the first half of the tokens, then 64 (that run switched at a third; the switch-point sweeps at 1B and
3B put the best at half the budget). So it is just

```
sbatch --time=5:00:00 run.sbatch --run_name=normuon-ramp-3b --token_budget=3e9
```

Branching instead of rerunning: runs that share a prefix (e.g. batch ramps that switch at different points) train
the prefix once. The trunk saves checkpoints; each branch resumes with its own flags. Use token-unit WSD in all of
them, and give the trunk the branches' micro-batch layout (here `--per_device_batch=2`: 2 x 2 accum x 8 GPUs = 32):

```
sbatch run.sbatch --run_name=trunk --token_budget=1e9 --schedule_unit=tokens --batch_ramp=32:0.5,32 --per_device_batch=2 \
    --save_at_tokens=100M,333M --ckpt_dir=/work/$USER/llama_ckpt/trunk
sbatch run.sbatch --run_name=at100M --token_budget=1e9 --schedule_unit=tokens --batch_ramp=32:100M,64 \
    --resume_from=/work/$USER/llama_ckpt/trunk/tokens-100M
```

Another `--optimizer` keeps its own LR defaults (6e-4; Muon 1.25e-3), not the muonium ones; the batch ramp still
applies unless `--batch_ramp=none` or `--global_batch_size` is given.

Compare runs' dynamics by stage (warmup / stable / decay) and block type with `analyze_dynamics.py`
(`NAME=path/optim_dynamics.jsonl ...`): a plot per metric, layer heatmaps, and `summary.md` with stage medians.

### Optimizer dynamics metrics (`--dyn_every N`)

Every N steps the logger (`llama_train/optim_dynamics.py`) watches two consecutive optimizer steps, a = b − 1 and b
(b is the logged step):

| Symbol | Meaning |
|---|---|
| g_a, g_b | Gradients at steps a and b, after clipping (what the optimizer reads) |
| v_a = W_after(a) − W_before(a) | The update the optimizer actually applied at step a, weight decay included. v_b likewise |
| W | Weights at step b, before its update |
| v̂ = v_a / ‖v_a‖ | Direction of the last update |

Every value is computed for one group of tensors:
- **Block types:** `embed` (tied with the LM head), `attn_q/k/v/o`, `mlp_gate/up/down`, `norm` (the RMSNorm weights).
- **`global`:** the whole model.
- **Per layer:** layer × block type.

Norms, cosines and curvature are exact for the group: dot products are summed over its tensors, as if the group were
one flattened vector (⟨x, y⟩ = Σ_tensors Σ_i x_i y_i), so `global` matches MiniCPM's whole-model definitions. Spectral
metrics apply to 2-D tensors only (not `norm`) and are averaged over the group's matrices.

On W&B, metric `m` of group `t` is `dyn_<m>/<t>` (e.g. `dyn_upd_rel/attn_q`, `dyn_grad_cos_prev/global`), plotted
against `train/global_step`. More W&B outputs:
- `dyn_lr/group<i>`: the LR of each optimizer param group.
- `dyn_heatmap/<m>`: a (layer × block type) image every `--dyn_heatmap_every` measurements, for `upd_rel`,
  `upd_cos_prev`, `grad_cos_prev` and `upd_srank`.

The JSONL (`--dyn_out`) has every value, per layer included, under keys `global/<m>`, `type/<t>/<m>` and
`layer/<LLL>/<t>/<m>`; load it with `load_dynamics()`.

**Gradient**

| `m` | Math | What it tells |
|---|---|---|
| `grad_norm`, `grad_max` | ‖g_b‖, max \|g_b\| | Gradient size per block. Spikes: instability; one block much larger than the others: imbalance |
| `grad_cos_prev` | ⟨g_a, g_b⟩ / (‖g_a‖ ‖g_b‖) | Oscillation (see below) |

How to read `grad_cos_prev`: on a quadratic with curvature λ along some direction, a step of size η gives
g_b = (1 − ηλ) g_a.
- < 0: ηλ > 1, the step overshoots and the iterate bounces across a sharp direction. Persistent values near −1 are
  the edge of stability [Cohen+21, Cohen+22].
- ≈ 0: gradient noise dominates.
- \> 0: steady descent; the step is short compared with the curvature (the LR could go up).

**Weights**

| `m` | Math | What it tells |
|---|---|---|
| `w_norm` | ‖W‖_F | Weight growth. With decoupled decay, ‖W‖ settles after ~1/(ηλ) steps [Kosson+23, Wang & Aitchison 24]. If it is still rising at the end, the decay is too weak for the run length |
| `w_sigma1` | σ₁(W): 6 power iterations, warm-started | Spectral norm: how much the layer can amplify its input. Growth here drives activation and logit growth (e.g. σ₁(W_q)·σ₁(W_k) bounds the attention logits) |
| `w_srank` | ‖W‖_F² / σ₁(W)² | Stable rank [Rudelson & Vershynin 07]: how many directions hold the weight's energy. Low means a few spikes dominate |

**Update**

| `m` | Math | What it tells |
|---|---|---|
| `upd_norm`, `upd_rms`, `upd_max` | ‖v_a‖, ‖v_a‖/√n, max \|v_a\| | Absolute step. RMS ≈ the LR means a sign-like update (Adam's is ~0.2–0.4 × LR [Liu+25]) |
| `upd_rel` | ‖v_a‖ / ‖W‖ | Relative step: the fraction of the weights changed per step. The most comparable step-size measure across optimizers and blocks |
| `upd_cos_prev` | ⟨v_a, v_b⟩ / (‖v_a‖ ‖v_b‖) | Smoothness of consecutive updates (momentum). Lower: the update direction changes more from step to step |
| `upd_grad_align` | ⟨v_a, −g_a⟩ / (‖v_a‖ ‖g_a‖) | 1 for plain SGD. Small values: momentum and the preconditioner turn the step away from the current gradient |
| `upd_spec_rel` | σ₁(v_a) / σ₁(W) | Step size in spectral norm, the natural norm for a hidden-layer matrix [Bernstein & Newhouse 24] |
| `upd_srank` | ‖v_a‖_F² / σ₁(v_a)² | Stable rank of the update |
| `upd_erank`, `upd_erank_frac` | exp(−Σᵢ pᵢ log pᵢ), pᵢ = σᵢ²/Σⱼ σⱼ² (full SVD of v_a); `_frac` = erank / min(rows, cols) | Effective rank [Roy & Vetterli 07; this σ² variant: Huang+26, sec. 3.2]. 1/min(rows, cols) means rank 1, 1 means every singular direction gets an equal share |
| `upd_row_cv` | std / mean over rows of ‖row_i(v_a)‖ | Variation across output neurons. High: a few neurons get most of the update (the imbalance NorMuon corrects [Li+25]) |
| `delta_rel`, `delta_erank`, `delta_erank_frac` | The same, for D = W_b − W_(b−N), the change over the last N steps | Oscillating components cancel in D, so this measures the progress that sticks, not the per-step jitter |

**Curvature along the update** (MiniCPM sec. 4.4 [Hu+24])

| `m` | Math | What it tells |
|---|---|---|
| `D1` | ⟨g_b, v̂⟩ | Slope of the loss along the last step, measured after taking it. > 0: the step went past the 1-D minimum along v̂; < 0: still downhill |
| `dir_sharpness` | ⟨g_b − g_a, v_a⟩ / ‖v_a‖² ≈ v̂ᵀ H v̂ | Hessian curvature along the update direction, by finite differences (g_b − g_a ≈ H v_a); MiniCPM's D2. Low: the optimizer steps in flat directions |
| `curvature_K` | \|D2\| / (1 + D1²)^{3/2} | Curvature of the 1-D loss curve. Here D1² ≪ 1, so K ≈ \|dir_sharpness\| |

Combining D1 and the sharpness: on a quadratic, the step length relative to the best step along v̂ is
r = 1 + D1 / ⟨−g_a, v̂⟩, where ⟨−g_a, v̂⟩ = `upd_grad_align` × ‖g_a‖.
- r = 1: an exact line search.
- 1 < r < 2: overshoots, but the loss still drops along v̂.
- r > 2: the loss rises along the step.

**Caveats**
- g_a and g_b come from different batches, so `D1` and `dir_sharpness` carry gradient noise of size about
  ‖noise‖ / ‖v_a‖. Use medians over many measurements.
- Ignore those two at the very end of the decay, where the LR is ~0 and the tiny update makes the noise term blow up.

**Cost:** ~5 s per measurement on the proxy (mostly SVDs, the 128k × 768 embedding included): about +30% at
`--dyn_every 50`. Use `--dyn_spectral_every 200`, or `--no-dyn_erank`, for routine runs.

What these showed for AdamW vs Muon vs Sophia on the proxy, and which optimizer fits each stage and block:
experiments.md 2026-10-17. References: list at the end of that entry.

## Data selection (small scoring model)

Small proxy / scoring model: `train_scorer.py` is `train.py` with a 174M config (768 hidden, 12 layers, 2048 context,
Muon 0.01 / AdamW 1e-3; any `train.py` flag overrides). Saved to `/work/$USER/llama_models/<run_name>`, not pushed.

```
sbatch --export=ALL,TRAIN_SCRIPT=train_scorer.py run.sbatch --run_name=scorer-6b --token_budget=6e9 --learning_rate=3e-3
```

Per-doc mean loss with it (`scoring.py` holds the shared code):

- `score_docs.py`: held-out docs right after `train.bin` in the shuffle; `percentiles.csv` per topic.
- `score_bin.py`: every doc of a `train.bin`, 8 GPUs; `train_scores.parquet`.
- `build_bands.py --drop_low L --drop_high H`: copy of `train.bin` without each topic's lowest L% / highest H% loss docs.
  Hurts eval perplexity (experiments.md 2026-10-15).

Soft reweighting (rows drawn with probability ~ exp(beta * z), z = loss standardized within topic; no row dropped).
Settings and paths in `config.yaml`; every stage resumes:

```
# 1. pool (~19B tokens of random rows, minus the scorer's train rows, the dev set and the official eval), tokenized
sbatch --array=0-7 --gpus-per-node=1 --cpus-per-task=12 --mem=96G --time=3:00:00 --output=logs/%x-%A_%a.out \
    --wrap="source ./.venv/bin/activate && export HF_HUB_CACHE=/work/\$USER/hf_cache/hub TOKENIZERS_PARALLELISM=false && \
    python build_pool.py manifest --task \$SLURM_ARRAY_TASK_ID --num_tasks 8"
# 2. score unique texts (first 512 tokens), 8 GPUs; 3. join, z per topic -> pool.parquet, dev.bin, pool_report.txt
sbatch --gpus-per-node=8 --cpus-per-task=96 --time=2:00:00 --wrap="source ./.venv/bin/activate && python build_pool.py score"
sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=128G --wrap="source ./.venv/bin/activate && python build_pool.py merge"
# 4. select a data dir (train.bin + val.bin + dev.bin); proxy mode: from a random 3x-budget subpool
sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=64G --wrap="source ./.venv/bin/activate && \
    python select_rows.py --beta 0.25 --budget 1e9 --seed 0 --subpool_mult 3"
```

`select_rows.py` writes `<pool>/sel_b<beta>_<budget>_s<seed>[_sub<k>]/` with `select_report.txt` (topic shares, mean z,
selected fraction per z bin); train on it with `--data_dir`. beta 0 is a uniform sample; no beta beat it at 1B tokens.

Also: `metadata_dist.py` (is `train.bin` representative of the mix), `repetition.py` (exact repeats in `train.bin`).

## 3. Push to the Hub

```
source .venv/bin/activate && source .env
python push_model.py --model_dir /work/$USER/llama_models/my-run --repo_id $MODEL_NAME --message "my-run, eval ppl ..."
```

This uploads the model, config, and tokenizer. Then set `eval_model_id` in the OJ script.

## Code layout

`train.py` is the entry point (`main(argv)`; `run.sbatch` launches it, `train_scorer.py` calls it with its defaults).
The pieces are in `llama_train/`:

| Module | Contents |
|---|---|
| `cli.py` | All flags, and the defaults that depend on other flags (muonium LRs, the default batch ramp) |
| `plan.py` | `Plan`: steps, batch ramp, `batch_at` / `tokens_seen`; `RampBatchSampler`; LR schedules (`hf_schedule`, and `lr_lambda` for wsd2 / `--schedule_unit tokens`); `--save_at_tokens` steps. No GPU needed |
| `model.py` | Llama 3.2 1B config and init, Liger fused LM head + loss, per-layer `torch.compile` |
| `data.py` | `TokenBlocks` (uint32 token files in blocks), `DocPackingCollate` (`--packing doc`), train / val / dev sets |
| `optim.py` | `build_optimizer`: Adam-mini, Muon, muonium, Sophia, SOAP (AdamW: the Trainer's own) |
| `callbacks.py` | Checkpoints at token marks, Slurm deadline, W&B token axis, step time / MFU, Sophia's Hessian pass, profiler |
| `optim_dynamics.py` | `OptimDynamicsLogger` / `OptimDynamicsCallback` (`--dyn_every`), `load_dynamics` for analysis |
| `trainer.py` | `CustomTrainer`: LR schedule, ramp sampler, dev-set eval, bf16 DDP hook, lab metrics |

Optimizer implementations stay at the top level (`muon_adamw.py`, `sophia.py`, `soap.py` / `soap_dist.py`).
`tests/plan_dump.py` prints the plan (per-step batch, tokens, LR) for 9 flag sets on the CPU; after a change to
`cli.py` / `plan.py`, diff its output against the previous commit's (`--impl old --src <old train.py>` reads a
pre-refactor `train.py`).
