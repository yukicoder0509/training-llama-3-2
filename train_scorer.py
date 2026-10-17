"""Small scoring model for perplexity-based filtering of the training set: train.py with a ~170M Llama 3.2 config.

    sbatch --export=ALL,TRAIN_SCRIPT=train_scorer.py,TORCHINDUCTOR_CACHE_DIR=/work/$USER/inductor_cache \
        --job-name=scorer run.sbatch --run_name=scorer-muon-1b --token_budget=1e9

Same code and settings as train.py (data, doc packing, FA3, fused CE, compile, WSD 2% / 10%, 0.5M-token steps), with:
  - the Llama 3.2 1B config shrunk to 768 hidden / 2048 FFN / 12 layers / 12 query + 4 KV heads, 2048 context, no
    long-context RoPE scaling (98.5M tied embedding + 75.5M in the layers);
  - 2048-token blocks, 256 per step (still 0.5M tokens), 32 per GPU (no grad accum on 8 GPUs);
  - Muon (muonium) for the layer matrices at LR 0.01, AdamW for the embedding and norms at LR 1e-3;
  - its own W&B project. The model is saved to /work/$USER/llama_models/<run_name> (not pushed).
Any train.py flag given on the command line overrides these defaults (argparse keeps the last value).
"""

import json
import os
import sys

SCORER_CONFIG = dict(
    hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
    num_attention_heads=12, num_key_value_heads=4, head_dim=64,
    max_position_embeddings=2048, rope_scaling=None,  # long-context scaling not needed
)
DEFAULTS = [
    "--model_name=scorer",  # local only; no Hub repo
    "--block_size=2048",
    f"--config_overrides={json.dumps(SCORER_CONFIG)}",
    "--global_batch_size=256", "--per_device_batch=32",
    "--optimizer=muonium", "--muonium_algorithm=muon", "--muonium_rest=adamw",
    "--muon_lr=0.01", "--learning_rate=1e-3", "--weight_decay=0.1",
]

# https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/llama3.2-arch-small-scoring-model
os.environ["WANDB_BASE_URL"] = "https://api.forge.coreweave.com"
os.environ["WANDB_ENTITY"] = "bendoris0509-national-yang-ming-chiao-tung-university"
os.environ["WANDB_PROJECT"] = "llama3.2-arch-small-scoring-model"

# run.sbatch always passes --model_name=$MODEL_NAME (the main 1B repo) first; drop it so the scorer is never named after
# that repo (a --model_name given later on the command line still applies)
cli = sys.argv[1:]
if cli and cli[0].startswith("--model_name="):
    cli = cli[1:]
if __name__ == "__main__":
    from train import main
    main([*DEFAULTS, *cli])
