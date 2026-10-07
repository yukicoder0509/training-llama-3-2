#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=llama-dolma3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96  # 12 per GPU (partition max); 1 CPU slowed NCCL allreduce ~5x (GPT-2 profile 499745 vs 499746)
#SBATCH --gpus-per-node=8
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=8:00:00  # lab cap: 64 H200-hours per run = 8 GPUs x 8 h
#SBATCH --output=logs/%x-%j.out

# Fewer GPUs (e.g. the 16 H200-hour baseline): sbatch --gpus-per-node=2 --cpus-per-task=24 --time=8:00:00 run.sbatch ...
# train.py stops 5 min before the job's time limit (squeue) and still saves the model.

cd $SLURM_SUBMIT_DIR
source $HOME/.bashrc
source ${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate
source $SLURM_SUBMIT_DIR/.env

# Log to the team W&B server (cerulean-labs/lab5-training-llama) instead of the personal wandb.ai account.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export WANDB_ENTITY=${WANDB_ENTITY:-cerulean-labs}
export WANDB_PROJECT=${WANDB_PROJECT:-lab5-training-llama}  # HF Trainer otherwise defaults to project "huggingface"
export HF_HUB_CACHE=/work/$USER/hf_cache/hub  # keeps the HF token at its default ~/.cache/huggingface/token

# Single-node jobs: NCCL bootstrap over loopback. On some nodes (hgpn122/123/124/151) it picked an
# unroutable interface and hung ~10 min ("No route to host"), jobs 496710-496729.
export NCCL_SOCKET_IFNAME=lo

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $(hostname)  GPUs: $NUM_GPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
echo "========================================"
START_TIME=$(date +%s)

# reference https://huggingface.co/docs/lerobot/multi_gpu_training
NUMA_PREFIX=$($SLURM_SUBMIT_DIR/numa_bind.sh)  # NUMA_BIND=local|mem to bind to the GPUs' NUMA node
echo "NUMA binding: ${NUMA_PREFIX:-none}"
$NUMA_PREFIX accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    --main_process_port=$((29500 + SLURM_JOB_ID % 1000)) \
    train.py --model_name=$MODEL_NAME --fused_ce --torch_compile "$@"  # --no-fused_ce / --no-torch_compile to disable

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s, $(awk "BEGIN {printf \"%.1f\", $((END_TIME - START_TIME)) * $NUM_GPUS / 3600}") GPU-hours"
echo "========================================"

# Note:
# hgpn003 breaks multi-GPU jobs (jobs 493252, 493275, 493290)
# hgpn001: both ranks land on the same GPU -> NCCL "Multiple Ranks are using the same GPU" (job 496208)
# hgpn157: other processes held 137.7 GiB of each GPU -> OOM at startup (job 504613)
