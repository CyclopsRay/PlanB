#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --mem=48G
# Specify a job name:
#SBATCH -J run_main
# Specify an output file
#SBATCH -o Run_cell-%J.out
#SBATCH -e Run_cell-%J.err
#SBATCH -p gpu --gres=gpu:1

module load cuda/11.3.1
module load cudnn/8.2.0
module load anaconda/3-5.2.0 gcc/10.2
source activate vae_att


/gpfs/data/rsingh47/ylei29/anaconda/vae_att/bin/python ./cell_matching.py