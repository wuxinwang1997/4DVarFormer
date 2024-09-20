#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpunode55
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/log-pretrain-fourcastnet-small-%j.out
#SBATCH --error=./slurmlogs/log-pretrain-fourcastnet-small-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/train.py trainer=gpu trainer.devices=1 model=fourcastnet model.net.depth=6 model.net.num_blocks=8 task_name=fourcastnet