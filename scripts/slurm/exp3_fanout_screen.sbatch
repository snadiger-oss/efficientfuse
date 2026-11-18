#!/bin/bash -l
#SBATCH --job-name=exp3_fanout
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/haripreeth/exp3_fanout_screen/slurm-%j.log

# Experiment 3 screen: shared expensive intermediate (fan-out). Submit from the repo root:
#   sbatch scripts/slurm/exp3_fanout_screen.sbatch
set -euo pipefail
module load gcc/13.2.0
export CC=$(which gcc) CXX=$(which g++) PYTHONUNBUFFERED=1
nvidia-smi
conda run --no-capture-output -n env_scalable python scripts/exp3_screen_fanout.py \
  --shapes 8192x8192 2048x32768 --k 2 4 8 12 16 24 32 \
  --rounds 5 --window 2.0 --output-dir results/haripreeth/exp3_fanout_screen
