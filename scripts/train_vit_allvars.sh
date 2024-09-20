#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpunode53
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/train-vit-allvars-assim3day-l1loss-%j.out
#SBATCH --error=./slurmlogs/train-vit-allvars-assim3day-l1loss-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/train.py model=vit datamodule=assim trainer=gpu trainer.devices=1 trainer.max_epochs=100 task_name=train_vit_allvars