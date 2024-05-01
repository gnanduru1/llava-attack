#!/bin/bash
# #SBATCH --partition=gpu
# #SBATCH --account=cs6501_sp24
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --reservation=bi_fox_dgx

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:2
# #SBATCH --partition=gpu
#SBATCH --job-name=rad-tune-alpha
#SBATCH --output=%u-%j-tune.out
#SBATCH --error=%u-%j-tune.err
#SBATCH --mem=256G
#SBATCH --array=0-1


date
nvidia-smi

source env.sh
python tune_alpha.py --id $SLURM_ARRAY_TASK_ID --n 800

# 60573009
# 60736481