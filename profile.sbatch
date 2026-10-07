#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --job-name=llama-profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24  # 12 per GPU (partition max)
#SBATCH --gpus-per-node=2
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x-%j.out

# Short profiling / speed run: same setup as run.sbatch, 20 steps, no eval / W&B / Hub. torch.profiler records
# steps 9-13 (train.py --profile_dir) and nvidia-smi samples GPU utilization every second.
# To profile inside a real full run instead: sbatch run.sbatch --profile_dir=logs/profile-<name> --profile_start=300
# Extra args go to train.py, e.g. --per_device_batch=8 to test memory / speed per micro-batch size.
# Output: logs/profile-<job id>/ (trace_rank*.json for chrome://tracing or ui.perfetto.dev, gpu_util.txt)

cd $SLURM_SUBMIT_DIR
source $HOME/.bashrc
source ${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate
source $SLURM_SUBMIT_DIR/.env

export WANDB_MODE=disabled
export HF_HUB_CACHE=/work/$USER/hf_cache/hub
export NCCL_SOCKET_IFNAME=lo  # see run.sbatch

NUM_GPUS=$(nvidia-smi -L | wc -l)
OUT=$SLURM_SUBMIT_DIR/logs/profile-$SLURM_JOB_ID
mkdir -p $OUT
echo "Node: $(hostname)  CPUs: $(nproc)  GPUs: $NUM_GPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi dmon -s um -d 1 -o T > $OUT/gpu_util.txt &
DMON_PID=$!

NUMA_PREFIX=$($SLURM_SUBMIT_DIR/numa_bind.sh)  # NUMA_BIND=local|mem to bind to the GPUs' NUMA node
echo "NUMA binding: ${NUMA_PREFIX:-none}"
$NUMA_PREFIX accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    --main_process_port=$((29500 + SLURM_JOB_ID % 1000)) \
    train.py --model_name=$MODEL_NAME --max_steps=20 --no-eval --profile_dir=$OUT --fused_ce --torch_compile \
    --save_dir=/tmp/profile-$SLURM_JOB_ID --run_name=profile-$SLURM_JOB_ID "$@"

kill $DMON_PID
rm -rf /tmp/profile-$SLURM_JOB_ID
echo "=== nvidia-smi dmon (sm% / mem% bandwidth, fb MB used per GPU), profile output in $OUT"
cat $OUT/gpu_util.txt
