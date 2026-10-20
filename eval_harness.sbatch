#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
# lm-evaluation-harness on saved models (separate venv: /work/$USER/venvs/lmeval, lm-eval 0.4.13, so the training
# venv's transformers is untouched). Usage: sbatch --job-name=harness eval_harness.sbatch OUT_DIR MODEL_DIR...
# BATCH (env, default 64): lm-eval batch size; 16 for the 1.2B main model (64 OOMs: fp32 logits over 128k vocab).
# Writes OUT_DIR/<model>/<suite>/results_*.json and samples_<task>_*.jsonl (per-choice log-likelihoods, for
# harness_proxy.py); suites: mmlu 0-shot / 5-shot (multiple choice, letter answer),
# mmlu_continuation 0-shot (cloze: log-likelihood of the answer text), and small-model tasks 0-shot.
set -euo pipefail
OUT=$1; shift
export HF_DATASETS_TRUST_REMOTE_CODE=1
LM=/work/$USER/venvs/lmeval/bin/lm_eval
for m in "$@"; do
  n=$(basename "$m")
  run() { $LM --model hf --model_args "pretrained=$m,dtype=bfloat16" --batch_size ${BATCH:-64} --output_path "$OUT/$n/$1" --tasks "$2" --num_fewshot "$3" --log_samples; }
  run mmlu0 mmlu 0
  run mmlu5 mmlu 5
  run mmlu_cloze mmlu_continuation 0
  run small arc_easy,arc_challenge,hellaswag,piqa,sciq,lambada_openai,winogrande 0
done
