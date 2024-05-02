#!/bin/bash
# #SBATCH --account=cs6501_sp24
# #SBATCH --partition=bii-gpu
# #SBATCH --account=bii_dsc_community
# #SBATCH --reservation=bi_fox_dgx
#SBATCH --partition=gpu

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=rad-compare-attacks
#SBATCH --output=%u-%j-comp.out
#SBATCH --error=%u-%j-comp.err
#SBATCH --mem=256G
#SBATCH --array=0-3

date
nvidia-smi

source env.sh
python compare_attacks.py --id $SLURM_ARRAY_TASK_ID --alpha=100 --n 800

