#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate

python train.py

# nvidia-smi
# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"