#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=dolma3-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out

# Download Dolma 3 (~110 GB), reproduce the OJ's seed-42 split and tokenize into $OUT_DIR/{train,val}.bin
# (+ eval_docs.jsonl with all 50k held-out docs). No GPU used, but the partition requires one.

cd $SLURM_SUBMIT_DIR
source $HOME/.bashrc
source ${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate

export HF_HUB_CACHE=/work/$USER/hf_cache/hub  # keeps the HF token at its default ~/.cache/huggingface/token
export HF_DATASETS_CACHE=/work/$USER/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false  # one tokenizer thread per worker process

OUT_DIR=${OUT_DIR:-/work/$USER/dolma3_llama}
python prepare_data.py --out_dir $OUT_DIR "$@"
