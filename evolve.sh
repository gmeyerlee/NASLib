#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

source ../naslib_env/bin/activate
python run_evolve.py --seed 1010

