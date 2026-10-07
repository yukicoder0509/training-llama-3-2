# Quick Start
1. Install virtual environments
2. Activate virtual environments
3. Login to Hugging Face and WandB
4. Create `.env` file. Put `MODEL_NAME=` variable. This is the Hub repo `push_model.py` pushes to.

# Usage

Run heavy work (data prep, training) through Slurm, not on the login node.

## 1. Prepare data

```
sbatch prepare.sbatch
```

Writes `train.bin` / `val.bin` to `$OUT_DIR` (default `/work/$USER/c4_gpt2`): the first 20 raw C4 `en`
train shards (~3B tokens, `--num_train_shards`) and 5k validation docs. This is what `run.sbatch` trains on by default.

Cleaned variants (CPU-only datatrove pipelines in `dedup_filter.py`, then tokenized; `val.bin` is copied from the
raw data so every run evaluates on the same tokens):

```
sbatch dedup_filter.sbatch dedup    # MinHash near-dedup (Jaccard ~0.8)          -> ~/c4_gpt2_dedup
sbatch dedup_filter.sbatch filter   # + mild Gopher repetition / quality filters  -> ~/c4_gpt2_dedup_filtered
sbatch run.sbatch --data_dir=$HOME/c4_gpt2_dedup --run_name=...
```

## 2. Train

```
sbatch --job-name=gpt2-c4 run.sbatch --run_name=my-run   # defaults = current best setting
```

| Flag | Default | Meaning |
|---|---|---|
| `--learning_rate` | `1.25e-3` | Peak LR |
| `--global_batch_size` | `128` | Sequences per optimizer step (per-GPU batch × grad accum × 2 GPUs); must have a measured step time in `SEC_PER_STEP` (64, 128, 256, 512) or pass `--sec_per_step` |
| `--per_device_batch` | `64` | Max sequences per GPU per micro-batch; grad accum covers the rest |
| `--pad_vocab` / `--no-pad_vocab` | on in `run.sbatch` / `profile.sbatch` (off when calling `train.py` directly) | Pad the vocab 50257 → 50304 so the LM-head matmuls use fast Hopper kernels (1.5× faster steps); trimmed back to 50257 before saving. `SEC_PER_STEP` assumes padding, so pass `--sec_per_step` with `--no-pad_vocab` |
| `--fused_ce` / `--no-fused_ce` | on in `run.sbatch` / `profile.sbatch` (off when calling `train.py` directly) | Liger fused LM head + cross-entropy (`liger-kernel`): no full fp32 logits, 1.24× faster steps, peak memory 65 → 24 GiB at 64/GPU. `SEC_PER_STEP` assumes it, so pass `--sec_per_step` with `--no-fused_ce` |
| `--torch_compile` / `--no-torch_compile` | on in `run.sbatch` / `profile.sbatch` (off when calling `train.py` directly) | `torch.compile` each transformer block: fuses LayerNorm / cast / GELU / elementwise kernels, 1.12× faster steps (~4 s compile). Compiling the whole `model.transformer` instead made attention 2.7× slower |
| `--attn_implementation` | `sdpa` | HF attention backend: `sdpa` (cuDNN flash), `flash_attention_2` / `flash_attention_3` (Hub kernels via `kernels`), `flash_attention_4` (`flash-attn-4`). FA3 / FA4 were no faster than `sdpa` here |
| `--profile_dir` / `--profile_start` / `--profile_cpu` | off / `8` / on | Profile steps start+1..start+5 with torch.profiler (also inside a full run, e.g. `--profile_start=300`); `--no-profile_cpu` = GPU kernels only |
| `--eval` / `--no-eval` | on | Periodic eval (`profile.sbatch` passes `--no-eval`) |
| `--activation` | `gelu_pytorch_tanh` | MLP activation; same formula as GPT-2's `gelu_new` but one fused kernel |
| `--optimizer` | `adamw` | `adamw`, `adam_mini`, or `muon` (Muon for the blocks' 2D weights + AdamW for the rest, `muon_adamw.py`; lost to AdamW by ~0.5 ppl at equal time). Muon runs ~5% slower per step, so size them with `--sec_per_step` (0.167 at batch 128) |
| `--muon_lr` / `--muon_momentum` | `1.25e-3` / `0.95` | Muon peak LR (`match_rms_adamw` scaling; best tested, ≥ 1e-2 diverges) and Nesterov momentum |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--beta2` | `0.95` | Adam β2 |
| `--warmup_frac` | `0.01` | Linear warmup over this fraction of the steps (1%: ppl 30.77; 10%: 32.34; 0.5%: 31.92) |
| `--decay_frac` | `0.2` | `0`: constant after warmup; `>0`: warmup-stable-decay, linear decay to 0 over this fraction of the final steps |
| `--dropout` | `0` | GPT-2's `resid_pdrop`, `attn_pdrop`, `embd_pdrop` (GPT-2 used 0.1) |
| `--data_dir` | `/work/$USER/c4_gpt2` | Directory with `train.bin` and `val.bin` |
| `--run_name` | none | W&B run name; also names the save directory |
| `--save_dir` | `~/gpt2_models/<run_name or latest>` | Where the final model is saved |

The step count fills the 30-min job limit: `(TIME_BUDGET − 37 evals × EVAL_SEC) / SEC_PER_STEP[batch]`
in `train.py` (batch 128 → 10334 steps). Warmup is 1% of the steps. If training runs slow, it stops 90 s before the limit and
still saves the model. Losses go to W&B (`cerulean-labs/gpt2-training`) and `logs/<job-name>-<job-id>.out`.

## 3. Push to the Hub

Training only saves the model locally, so the upload doesn't count against the job's time
limit. After the job finishes, push from the login node. Uploading only uses the network,
so it doesn't need Slurm:

```
source .venv/bin/activate && source .env
python push_model.py --model_dir ~/gpt2_models/my-run --repo_id $MODEL_NAME --message "my-run, eval 3.61"
```

The last lines of the training log print this command with the right paths.

| Flag | Default | Meaning |
|---|---|---|
| `--model_dir` | required | Folder written by `train.py` |
| `--repo_id` | `$MODEL_NAME` | Hub repo to push to |
| `--message` | `Upload <folder name>` | Commit message |

It uploads `config.json`, `generation_config.json` and `model.safetensors` as one commit. Each
saved model takes ~500 MB of the 100 G `/home` quota; delete old ones from `~/gpt2_models/` once pushed.

Experiment history and results: [`experiments.md`](experiments.md).
