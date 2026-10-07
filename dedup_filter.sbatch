#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=c4-clean
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

# Clean C4 with dedup_filter.py, then tokenize the result (CPU only; the partition requires a GPU).
#   sbatch dedup_filter.sbatch dedup    # MinHash near-dedup     -> ~/c4_prep/dedup          -> ~/c4_gpt2_dedup
#   sbatch dedup_filter.sbatch filter   # mild filters on dedup  -> ~/c4_prep/dedup_filtered -> ~/c4_gpt2_dedup_filtered
# val.bin is copied from the unfiltered baseline data, so every run evaluates on identical tokens.
# Train with: sbatch run.sbatch --data_dir=$HOME/c4_gpt2_dedup[_filtered] ...

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate
export HF_HOME=/work/$USER/hf_cache
set -e
cd $HOME/training-gpt-2  # explicit cwd: a getcwd() failure on the network FS killed job 505851 between stages

STAGE=$1
case $STAGE in
    dedup)  SRC=dedup;          OUT=$HOME/c4_gpt2_dedup ;;
    filter) SRC=dedup_filtered; OUT=$HOME/c4_gpt2_dedup_filtered ;;
    *) echo "usage: sbatch dedup_filter.sbatch dedup|filter"; exit 1 ;;
esac

python dedup_filter.py --stage $STAGE
mkdir -p $OUT
cp /work/$USER/c4_gpt2/val.bin $OUT/val.bin
python prepare_data.py --input_dir $HOME/c4_prep/$SRC --out_dir $OUT
