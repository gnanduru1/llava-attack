#!/bin/bash
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --reservation=bi_fox_dgx
# #SBATCH --account=cs6501_sp24
# #SBATCH --partition=gpu

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=rad-training
#SBATCH --output=%u-%j-rad-train.out
#SBATCH --error=%u-%j-rad-train.err
#SBATCH --mem=256G
#SBATCH --array=0-3

date
nvidia-smi

source env.sh
python training.py --rad --id $SLURM_ARRAY_TASK_ID --n 1534

