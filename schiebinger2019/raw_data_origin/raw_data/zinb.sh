#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=50G
#SBATCH -J k-2
#SBATCH -o Run_cell-%J.out
#SBATCH -e Run_cell-%J.err
#SBATCH -n 2

module load cuda/11.3.1
module load cudnn/8.2.0
module load anaconda/3-5.2.0 gcc/10.2
source activate Renv

/users/lyuyang/anaconda3/envs/Renv/bin/Rscript /users/lyuyang/lyy/PlanB/schiebinger2019/raw_data_origin/raw_data/zinb.R