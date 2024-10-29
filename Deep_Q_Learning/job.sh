#!/usr/bin/env bash
#SBATCH --mail-user=sviswasam@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -A cs525
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem 12G
#SBATCH --job-name="P3"

#SBATCH --output=/home/sviswasam/rl/logs/logs.out
#SBATCH --error=/home/sviswasam/rl/logs/err.err

source activate myenv
python main.py --train_dqn