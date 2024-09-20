#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p V100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpunode58
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/train-4dvarformer-allvars-assim3day-20240614-%j.out
#SBATCH --error=./slurmlogs/train-4dvarformer-allvars-assim3day-20240614-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/train.py model=fdvarformer model.obs_vars=[8,9,10,11,20,21] datamodule=assim trainer=gpu trainer.max_epochs=100 trainer.devices=1 task_name=train_4dvarformer_20240613