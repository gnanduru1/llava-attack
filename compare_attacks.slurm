#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:4
# #SBATCH --partition=gpu
# #SBATCH --account=cs6501_sp24
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --reservation=bi_fox_dgx
#SBATCH --job-name=rad-compare-attacks
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem=256G
# #SBATCH --exclusive

date
nvidia-smi

source env.sh
python compare_attacks.py

