#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p V100
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpunode55
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/eval-4dvarformer-diffobs-20240617-%j.out
#SBATCH --error=./slurmlogs/eval-4dvarformer-diffobs-20240617-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/inference/eval_medium_forecast.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_vars 8 9 10 11 20 21
#python src/inference/eval_medium_forecast.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=4 --mode=test --obs_vars 8 9 10 11 20 21
#python src/inference/eval_medium_forecast.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=5 --mode=test --obs_vars 8 9 10 11 20 21
#python src/inference/eval_medium_forecast.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_vars 8 9 10 11
#python src/inference/eval_medium_forecast.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_vars 20 21
#
srun python src/inference/eval_da_cycle_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=4 --obs_vars 8 9 10 11 20 21
srun python src/inference/eval_da_cycle_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=3 --obs_vars 8 9 10 11 20 21
srun python src/inference/eval_da_cycle_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=2 --obs_vars 8 9 10 11 20 21
srun python src/inference/eval_da_cycle_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=1 --obs_vars 8 9 10 11 20 21

python src/inference/eval_medium_forecast_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=4 --obs_vars 8 9 10 11 20 21
python src/inference/eval_medium_forecast_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=3 --obs_vars 8 9 10 11 20 21
python src/inference/eval_medium_forecast_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=2 --obs_vars 8 9 10 11 20 21
python src/inference/eval_medium_forecast_diffobs.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --forecast_step=28 --decorrelation_step=20 --init_time=3 --mode=test --obs_num=1 --obs_vars 8 9 10 11 20 21

#srun python src/inference/eval_da_cycle.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=4 --mode=test --obs_vars 8 9 10 11 20 21
#srun python src/inference/eval_da_cycle.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=5 --mode=test --obs_vars 8 9 10 11 20 21
#srun python src/inference/eval_da_cycle.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_vars 8 9 10 11
#srun python src/inference/eval_da_cycle.py --pretrain_dir=../ckpts/4dvarformer_base.ckpt --da_method=4dvarformer --assim_step=120 --decorrelation_step=20 --init_time=3 --mode=test --obs_vars 20 21
