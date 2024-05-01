#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu
#SBATCH --job-name=rad-compare-attacks
#SBATCH --output=%u-%j-comp.out
#SBATCH --error=%u-%j-comp.err
#SBATCH --mem=256G
#SBATCH --array=0-3

date
nvidia-smi

source env.sh
python compare_attacks.py --id $SLURM_ARRAY_TASK_ID

