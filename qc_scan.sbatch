#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=dolma3-qc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

# Sample the dolma2_qc scores of Dolma 3 into /work/$USER/dolma3_llama/qc_scan.npz (qc_scan.py). Plot: plot_qc.py.
# No GPU used, but the partition requires one.

cd $SLURM_SUBMIT_DIR
source $HOME/.bashrc
source ${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate
export HF_HUB_CACHE=/work/$USER/hf_cache/hub
export TOKENIZERS_PARALLELISM=false

python qc_scan.py --num_proc $SLURM_CPUS_PER_TASK "$@"
