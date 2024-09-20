#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p A100
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpunode55
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/train-4dvarnet-allvars-assim3day-l1loss-%j.out
#SBATCH --error=./slurmlogs/train-4dvarnet-allvars-assim3day-l1loss-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/train.py model=fdvarnet datamodule=assim trainer=dp trainer.devices=2 trainer.max_epochs=100 task_name=train_4dvarnet_allvars