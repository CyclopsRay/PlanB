#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --mem=48G
# Specify a job name:
#SBATCH -J run_main
# Specify an output file
#SBATCH -o Run_main-%J.out
#SBATCH -e Run_main-%J.err
#SBATCH -p gpu --gres=gpu:1

module load cuda/11.3.1
module load cudnn/8.2.0
module load anaconda/3-5.2.0 gcc/10.2
source activate vae_att
# Change to your own name

/gpfs/data/rsingh47/ylei29/anaconda/vae_att/bin/python ./main.py
