#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --job-name=gpt2-profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24  # 12 per GPU (partition max); 1 CPU slowed NCCL allreduce ~5x (profile 499745 vs 499746)
#SBATCH --gpus-per-node=2
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=0:15:00
#SBATCH --output=logs/%x-%j.out

# Short profiling run: same setup as run.sbatch, 20 steps, no eval / W&B / Hub. torch.profiler records
# steps 9-13 (train.py --profile_dir) and nvidia-smi samples GPU utilization every second.
# To profile inside a real full run instead: sbatch run.sbatch --profile_dir=logs/profile-<name> --profile_start=300
# (add --no-profile_cpu for GPU kernels only, lowest overhead).
# Extra args go to train.py. Compare CPU counts by overriding sbatch's --cpus-per-task, e.g.:
#   sbatch --cpus-per-task=1 --job-name=gpt2-profile-cpu1 profile.sbatch
# Output: logs/profile-<job id>/ (trace_rank*.json for chrome://tracing or ui.perfetto.dev, gpu_util.txt)

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate
source $HOME/training-gpt-2/.env

export WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=lo  # see run.sbatch

OUT=$HOME/training-gpt-2/logs/profile-$SLURM_JOB_ID
mkdir -p $OUT
echo "Node: $(hostname)  CPUs: $(nproc)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi dmon -s um -d 1 -o T > $OUT/gpu_util.txt &
DMON_PID=$!

NUMA_PREFIX=$($HOME/training-gpt-2/numa_bind.sh)  # NUMA_BIND=local|mem to bind to the GPUs' NUMA node
echo "NUMA binding: ${NUMA_PREFIX:-none}"
$NUMA_PREFIX accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=bf16 \
    --main_process_port=$((29500 + SLURM_JOB_ID % 1000)) \
    train.py --model_name=$MODEL_NAME --max_steps=20 --no-eval --profile_dir=$OUT --pad_vocab --fused_ce --torch_compile \
    --save_dir=/tmp/profile-$SLURM_JOB_ID --run_name=profile-$SLURM_JOB_ID "$@"

kill $DMON_PID
rm -rf /tmp/profile-$SLURM_JOB_ID
echo "=== nvidia-smi dmon (sm% / mem% bandwidth, fb MB used per GPU), profile output in $OUT"
cat $OUT/gpu_util.txt
