#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=gpt2-c4-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

# Tokenize C4 into $OUT_DIR/{train,val}.bin (no GPU used, but the partition requires one).
# Default: the first 20 raw train shards (~3B tokens). For deduped / filtered data use dedup_filter.sbatch.

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate

export HF_HOME=/work/$USER/hf_cache

OUT_DIR=${OUT_DIR:-/work/$USER/c4_gpt2}
python prepare_data.py --out_dir $OUT_DIR "$@"
