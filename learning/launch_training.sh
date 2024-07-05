#!/bin/bash -l

#SBATCH --job-name="SAME_AS_WANDB?"
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --mail-user=EMAIL
#SBATCH --mail-type=ALL

# Run this script with: sbatch launch_training.sh

# Print the current date for debugging
date

# Load conda and run the training script
module load miniconda3
python training.py MODEL_NAME PROJECT_NAME "Sample description of training run..." ARCHITECTURE_NAME DATA_NAME


# Print the name of the node for debugging
hostname
