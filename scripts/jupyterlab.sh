#!/bin/bash

#SBATCH -p normal
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --job-name jupyter
#SBATCH --cpus-per-task=4
#SBATCH --mem 100G
#SBATCH --output jupyterlab-log-%J.log

# get tunneling info
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$USER
cluster=$(hostname -f | awk -F"." '{print $2}')

# 填入服务器地址
clusterurl="27.132.92.74"

export PATH=$PATH:~/.local/bin

#module load conda/anaconda/2022.05
#source activate torch-deepda

echo -e "\n"
echo    "  Paste ssh command in a terminal on local host (i.e., laptop)"
echo    "  ------------------------------------------------------------"
echo -e "  ssh -N -L ${port}:${node}:${port} ${user}@${clusterurl}\n"
echo    "  Open this address in a browser on local host; see token below"
echo    "  ------------------------------------------------------------"
echo -e "  localhost:${port}"

jupyter-lab --no-browser --port=${port} --ip=${node}
