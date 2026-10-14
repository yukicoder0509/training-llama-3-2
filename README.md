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
| `--global_batch_size` | `64` | Sequences per step (64 x 8192 = 0.5M tokens) |
| `--per_device_batch` | `4` | Max sequences per GPU per micro-batch; grad accum covers the rest |
| `--batch_ramp` | none | `"B1:f1,...,Bn"`: batch Bi for fraction fi of the token budget |
| `--learning_rate` / `--weight_decay` / `--beta2` | `6e-4` / `0.1` / `0.95` | AdamW (untuned starting points) |
| `--warmup_frac` / `--decay_frac` | `0.02` / `0.1` | Linear warmup, then WSD with linear decay to 0 over the final fraction (0.1 as in MiniCPM; runs before 2026-10-14 used 0.2) |
| `--lr_schedule` / `--min_lr_ratio` | `wsd` / `0.1` | `cosine`: cosine from the peak to `min_lr_ratio` × peak (Llama 2/3) |
| `--adam_eps` | `1e-8` | Adam ε (Llama 2: 1e-5) |
| `--optimizer` | `adamw` | `adamw` (fused torch AdamW), `adam_mini`, `muon` (Muon for decoder-layer matrices + AdamW for embeddings / norms, `muon_adamw.py`), `sophia` (SophiaG, `sophia.py` vendored from the official repo) or `soap` (see below); LR = `--learning_rate` |
| `--muon_lr` / `--muon_momentum` | `1.25e-3` / `0.95` | Muon (tuned on GPT-2, re-tune) |
| `--soap_impl` / `--soap_betas` / `--soap_precond_freq` | `official` / `0.95,0.95` / `10` | SOAP (`--optimizer soap`, ε = `--adam_eps`). `official`: `soap.py` (paper authors) with the work split across GPUs by `soap_dist.py`. `meta`: facebookresearch `distributed_shampoo` (`--soap_block`, `--soap_comm_dtype`), no faster at the same preconditioner. +31% step time at 0.5M batch, +7% at 2M |
| `--sophia_rho` / `--sophia_betas` / `--sophia_hess_interval` | `0.05` / `0.965,0.99` / `10` | SophiaG (official GPT-2 values). The Hessian pass every k steps reruns the step's batch with sampled labels: +1/k compute. Use weight decay ~2x AdamW's (0.2). Watch `train/sophia_win_rate` (official guidance: 0.1–0.5) |
| `--ddp_find_unused` / `--ddp_bucket_cap_mb` / `--ddp_bf16_grads` | off / `200` / off | DDP settings (+6% vs Trainer's defaults of on / 25 MB); bf16 all-reduce gives no gain with large buckets |
| `--num_evals` | `20` | Evals over the run (lab: at least every 10% of steps) plus a final eval |
| `--fused_ce` / `--torch_compile` | on in the sbatch scripts | Liger fused LM head + CE (no 4 GiB/seq fp32 logits); `torch.compile` per decoder layer |
| `--attn_implementation` | `flash_attention_3` | `flash_attention_3` (Hub kernel via `kernels`), `sdpa`, `flash_attention_4` |
| `--packing` | `doc` | `doc`: each document attends only to itself, positions restart per document (FA varlen; ~10% faster). `concat`: attention crosses document boundaries (all runs before 2026-10-14). Evals use the same mode; compare across modes with `eval_packing.py` |
| `--profile_dir` / `--profile_start` / `--profile_cpu` | off / `8` / on | torch.profiler over 5 steps |
| `--data_dir` | `/work/$USER/dolma3_llama` | `train.bin` / `val.bin` |
| `--save_dir` | `/work/$USER/llama_models/<run_name>` | Final model + tokenizer |

Runs are bounded by the token budget: the step count comes from `--token_budget`, and a run ends when those steps are
done. The Slurm time limit is only a safety net. Training stops 5 min before it (read from `squeue`) and still saves
the model, but that run saw fewer tokens and is not comparable. At step 50 the log prints the projected finish (or a
WARNING if it is past the deadline), and the W&B summary records `budget_completed` and `final_tokens_seen`. Give
comparison runs enough `--time` (e.g. 1:15 for a 1.1B-token run, which takes ~50 min).

W&B: `cerulean-labs/lab5-training-llama` on `https://app.forge.coreweave.com`. The lab's required metrics are
`train/loss` (mean over the 10 logging steps), `train/grad_norm`, `train/learning_rate`, `train/tokens_per_second`
(of that step), `train/total_tokens_seen`, and `eval/perplexity`. Extras: `train/mfu`, `train/ppl`, `eval/loss`.
All train and eval charts use `train/total_tokens_seen` as their x-axis (eval rows carry that key too). To plot runs
against tokens locally (login node, reads W&B):

```bash
python plot_runs.py dvqj42sv dkyqsbe6 --out plots/wsd_vs_cosine_1h.png   # train/eval loss and ppl vs tokens seen
```

## 3. Push to the Hub

```
source .venv/bin/activate && source .env
python push_model.py --model_dir /work/$USER/llama_models/my-run --repo_id $MODEL_NAME --message "my-run, eval ppl ..."
```

This uploads the model, config, and tokenizer. Then set `eval_model_id` in the OJ script.
