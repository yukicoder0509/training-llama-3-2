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
The tokenizer is `NousResearch/Llama-3.2-1B` (its `tokenizer.json` is byte-identical to the gated `meta-llama/Llama-3.2-1B`). The data is not cleaned.
(`dedup_filter.py` / `dedup_filter.sbatch` are the old C4 pipelines, not yet ported.)

## 2. Train

```
sbatch run.sbatch --run_name=my-run                     # 8 H200 x 8 h max (= the 64 H200-hour cap), 6B tokens
sbatch --gpus-per-node=2 --cpus-per-task=24 run.sbatch --run_name=baseline --token_budget=3e9   # ~16 H200-hours
sbatch profile.sbatch --per_device_batch=8              # 20-step speed / memory test, no W&B
```

Measured on 8 H200 (job 512020, AdamW, compile + fused CE, 4 x 8192 tokens/GPU x accum 2): **410k tokens/s
(51.3k/GPU, MFU ~55%), 80 GiB peak**, so 6B tokens take ~4.1 h (~33 H200-hours). Muon is ~4% slower, Adam-mini about
the same. Flag sweep: experiments.md 2026-10-08.

| Flag | Default | Meaning |
|---|---|---|
| `--token_budget` | `6e9` | Training tokens; steps = budget / (global batch x 8192). No padding, so every token counts |
| `--global_batch_size` | `64` | Sequences per step (64 x 8192 = 0.5M tokens) |
| `--per_device_batch` | `4` | Max sequences per GPU per micro-batch; grad accum covers the rest |
| `--batch_ramp` | none | `"B1:f1,...,Bn"`: batch Bi for fraction fi of the token budget |
| `--learning_rate` / `--weight_decay` / `--beta2` | `6e-4` / `0.1` / `0.95` | AdamW (untuned starting points) |
| `--warmup_frac` / `--decay_frac` | `0.02` / `0.2` | Linear warmup, then WSD with linear decay to 0 over the final fraction |
| `--lr_schedule` / `--min_lr_ratio` | `wsd` / `0.1` | `cosine`: cosine from the peak to `min_lr_ratio` × peak (Llama 2/3) |
| `--adam_eps` | `1e-8` | Adam ε (Llama 2: 1e-5) |
| `--optimizer` | `adamw` | `adamw` (fused torch AdamW), `adam_mini`, or `muon` (Muon for decoder-layer matrices + AdamW for embeddings / norms, `muon_adamw.py`) |
| `--muon_lr` / `--muon_momentum` | `1.25e-3` / `0.95` | Muon (tuned on GPT-2, re-tune) |
| `--num_evals` | `20` | Evals over the run (lab: at least every 10% of steps) plus a final eval |
| `--fused_ce` / `--torch_compile` | on in the sbatch scripts | Liger fused LM head + CE (no 4 GiB/seq fp32 logits); `torch.compile` per decoder layer |
| `--attn_implementation` | `sdpa` | `sdpa`, `flash_attention_3`, `flash_attention_4` |
| `--profile_dir` / `--profile_start` / `--profile_cpu` | off / `8` / on | torch.profiler over 5 steps |
| `--data_dir` | `/work/$USER/dolma3_llama` | `train.bin` / `val.bin` |
| `--save_dir` | `/work/$USER/llama_models/<run_name>` | Final model + tokenizer |

The training stops 5 min before the Slurm time limit (read from `squeue`) and still saves the model.

W&B: `cerulean-labs/lab5-training-llama` on `https://app.forge.coreweave.com`. The lab's required metrics are
`train/loss` (mean over the 10 logging steps), `train/grad_norm`, `train/learning_rate`, `train/tokens_per_second`
(of that step), `train/total_tokens_seen`, and `eval/perplexity`. Extras: `train/mfu`, `train/ppl`, `eval/loss`.

## 3. Push to the Hub

```
source .venv/bin/activate && source .env
python push_model.py --model_dir /work/$USER/llama_models/my-run --repo_id $MODEL_NAME --message "my-run, eval ppl ..."
```

This uploads the model, config, and tokenizer. Then set `eval_model_id` in the OJ script.
