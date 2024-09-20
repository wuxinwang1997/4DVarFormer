import sys
sys.path.append(".")
from pathlib import Path
import pickle
import numpy as np
import torch
from src.datamodules.components.era5_datasets import ERA5
from src.models.fourcastnet_module import FourCastNetLitModule
from src.inference.inference import autoregressive_inference
import matplotlib as plt


data_dir = "../data/era5_6_hourly/ffr_era5"
mode = "test"
modelname = "fourcastnet"
pretrain_dir = "../ckpts/"
output_dir = "../output/forecast"
ckpt = "fourcastnet.ckpt"
var_idx_dir = "../data/era5_6_hourly/ffr_era5/var_idx.npy"
device = "cuda:0"
prediction_days = 7
decorrelation_step = 20
start_id = 0
dt = 6

module = FourCastNetLitModule.load_from_checkpoint(f"{pretrain_dir}/{ckpt}")
module.net.to(device).eval()

var_idx = module.var_idx
mult = module.mult.to(device, dtype=torch.float32)
clim = module.clims[2].to(device, dtype=torch.float32)

eval_dataset = ERA5(data_dir=data_dir, split=mode, modelname=modelname)

# 取初始场
prediction_length = 24 * prediction_days
n_samples = eval_dataset.__len__() - prediction_length // 6
stop = n_samples
ics = np.arange(start_id, stop, decorrelation_step)

fcs = []
val_pred, val_real, val_rmse, val_acc = [], [], [], []

for i, ic in enumerate(ics):
    seq_real, seq_pred, seq_rmse, seq_acc = autoregressive_inference(ic,
                                                                    eval_dataset,
                                                                    module,
                                                                    dt,
                                                                    prediction_length,
                                                                    clim,
                                                                    mult,
                                                                    device)
    if i == 0:
        val_pred = seq_pred
        val_rmse = seq_rmse
        val_acc = seq_acc
    else:
        val_pred = np.concatenate((val_pred, seq_pred), axis=0)
        val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
        val_acc = np.concatenate((val_acc, seq_acc), axis=0)

for i in range(val_rmse.shape[-1]):
    print(f"RMSE of {var_idx[i]} is： {np.mean(val_rmse, axis=0)[:, i]}")
    print(f"ACC of {var_idx[i]} is： {np.mean(val_acc, axis=0)[:, i]}")

np.save(f"{output_dir}/rmse_fourcastnet.npy", val_rmse)
np.save(f"{output_dir}/acc_fourcastnet.npy", val_acc)

