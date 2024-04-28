#!/bin/bash

#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --job-name=rad-compare-attacks
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem=256G
#SBATCH --array=0-1

date
nvidia-smi

source env.sh
python compare_attacks.py --id $SLURM_ARRAY_TASK_ID

