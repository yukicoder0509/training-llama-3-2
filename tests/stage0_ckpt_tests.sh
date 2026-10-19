#!/bin/bash
# Stage 0 T2 / T3: checkpoint round trips on the 174M proxy, batch 128 x 2048 (262,144 tokens/step), W&B off.
#   bash tests/stage0_ckpt_tests.sh main     # submit the two straight runs (they save the checkpoints)
#   bash tests/stage0_ckpt_tests.sh branch   # after both finished: submit the resumed runs
# T3 (decay branches with existing flags): main run 400 steps (WSD decay over the last 40), checkpoints at steps 200
#   and 360. Branch from 360 with the branch recipe (--token_budget=(S+N) x 262144, --decay_frac=(N+0.5)/(S+N)) must
#   reproduce the main run's final eval within ~0.003. Branch from 200 with N=40 must decay linearly from the peak at
#   step 200 to 0 at 240.
# T2 (--embed_optimizer soap save -> load -> continue): 60 steps, checkpoint at 20; the resumed run must track the
#   straight run's train loss over steps 30-60 and its final eval.
set -e
cd "$(dirname "$0")/.."
TPS=262144
X="--export=ALL,TRAIN_SCRIPT=train_scorer.py,TORCHINDUCTOR_CACHE_DIR=/work/$USER/inductor_cache,WANDB_MODE=disabled"
EX=25a-hgpn001,25a-hgpn003,25a-hgpn157,25a-hgpn062,25a-hgpn083,25a-hgpn173,25a-hgpn069
COMMON="--global_batch_size=128 --per_device_batch=16 --muonium_algorithm=normuon --learning_rate=3e-3"
SOAP="--embed_optimizer=soap --learning_rate=2.8e-3 --norm_lr=3e-3"
CK=/work/$USER/llama_ckpt
sub() { name=$1; shift; echo "$(sbatch --parsable $X --exclude=$EX --time=0:40:00 --job-name=$name run.sbatch --run_name=$name $COMMON "$@") $name"; }

case "$1" in
main)
    sub s0-t3-main --token_budget=$((400 * TPS)) --save_at_tokens=$((200 * TPS)),$((360 * TPS))
    sub s0-t2-main --token_budget=$((60 * TPS)) --save_at_tokens=$((20 * TPS)) $SOAP
    ;;
branch)
    sub s0-t3-branch360 --resume_from=$CK/s0-t3-main/checkpoint-360 --token_budget=$((400 * TPS)) --decay_frac=$(python3 -c "print(40.5 / 400)")
    sub s0-t3-branch200 --resume_from=$CK/s0-t3-main/checkpoint-200 --token_budget=$((240 * TPS)) --decay_frac=$(python3 -c "print(40.5 / 240)")
    sub s0-t2-resume20 --resume_from=$CK/s0-t2-main/checkpoint-20 --token_budget=$((60 * TPS)) $SOAP
    ;;
esac
