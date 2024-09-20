import sys
sys.path.append(".")
from pathlib import Path
import pickle
import numpy as np
import torch
from src.datamodules.components.assim_datasets import Assim
from src.datamodules.assim_datamodule import AssimDataModule
from src.models.fourcastnet_module import FourCastNetLitModule
from src.models.map_module import MapLitModule
from src.inference.inference import var4d_medium_forecast
import matplotlib as plt
import argparse

def laplace(img):
    n, r, c = img.shape
    new_image = np.zeros((n, r-2, c-2))
    L_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    for k in range(n):
        for i in range(r-3):
            for j in range(c-3):
                new_image[k, i+1, j+1] = abs(np.sum(img[k, i:i+3, j:j+3] * L_lap))

    return new_image

def GC_1(r):
    return 1 - (5/3)*r**2 + (5/8)*r**3 + (1/2)*r**4 - (1/4)*r**5

def GC_2(r):
    return 4 - 5*r + (5/3)*r**2 + (5/8)*r**3 -(1/2)*r**4 + (1/12)*r**5 - 2/(3*r+1e-9)

def medium_forecast_inference(data_dir,
                                pretrain_ckpt,
                                output_dir,
                                init_time,
                                dt,
                                spin_up_step,
                                forecast_step,
                                decorrelation_step,
                                mode,
                                obs_vars,
                                device):

    data_dir_nmc = f"{data_dir}/assim_dir_{init_time-1}day"
    data_dir_run = f"{data_dir}/assim_dir_{init_time}day"

    spin_up_hours = 6 * spin_up_step
    forecast_hours = 6 * forecast_step
    decorrelation_step = decorrelation_step
    start_id = 0

    afnonet = FourCastNetLitModule.load_from_checkpoint(f"/public/home/wangwuxing01/research/fundation/ckpts/fourcastnet.ckpt")
    afnonet = afnonet.net.to(device).eval()
    module = MapLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    module.reset_mask(obs_vars)
    var_idx = module.var_idx
    mult = module.mult
    mask = module.mask
    clim = module.clims[2]

    module.requires_grad = False
    module.reset_mask(obs_vars)
    module.to(device, dtype=torch.float32).eval()

    eval_dataset_nmc = Assim(data_dir=data_dir_nmc, split=mode, modelname="fourcastnet")
    eval_dataset = Assim(data_dir=data_dir_run, split=mode, modelname="fourcastnet")
    valid_data_nmc = eval_dataset_nmc.__getitem__(np.arange(4, 29*4).tolist())
    valid_data_all = eval_dataset.__getitem__(np.arange(28*4).tolist())

    diff = []
    for i in range(len(valid_data_nmc)):
        xb1 = valid_data_nmc[i][1]
        xb2 = valid_data_all[i][1]
        diff.append((mult ** 2) * np.mean((xb2 - xb1) ** 2, axis=(-2, -1), keepdims=True))
    diff = np.concatenate(diff, axis=0)
    B_half = np.sqrt(np.mean(diff, axis=0))

    # S, L, B_half = [], [], []
    # Nx = xb1.shape[-2] * xb1.shape[-1]
    # row, col = np.meshgrid(np.arange(Nx), np.arange(Nx))
    # dist_matrix = np.abs(row - col)
    # dist_matrix = dist_matrix // xb1.shape[-1] + dist_matrix % xb1.shape[-1]
    # for i in range(s.shape[-1]):
    #     S.append(s[0, i] * np.triu(np.ones((Nx, Nx)), 0))
    #     L.append(np.sqrt(np.sqrt(8 * np.var(xb1) / np.var(laplace(xb1)))))
    #     d_matrix = dist_matrix / (L[i] * xb1.shape[-1] // 2)
    #     corr_matrix = ((dist_matrix < 1) * GC_1(d_matrix) +
    #                    (dist_matrix < 2) * GC_2(d_matrix / 1) -
    #                    (dist_matrix < 1) * GC_2(d_matrix)) * \
    #                   (1 - ((dist_matrix < 1) * GC_1(d_matrix > 2)))
    #     B_half.append((S[i] * np.exp(-corr_matrix / L[i])).astype(np.float32))

    obserr = [0, 0, 0, 0, 14, 10, 13, 13, 2.0, 0.5, 1.4, 2.2, 0, 0, 0, 0, 0, 0, 0, 0, 1.43, 1.40, 0, 0]
    obserr_ = np.ones((1, 24))
    for i in range(24):
        obserr_[:, i] = obserr[i] * obserr_[:, i]
    obserr = obserr_ ** 2
    R_inv = torch.Tensor(np.where(obserr == 0, 0, 1 / obserr).astype(np.float32))

    # 取初始场
    n_samples = eval_dataset.__len__() - (spin_up_step + forecast_step)
    stop = n_samples
    ics = np.arange(start_id, stop, decorrelation_step)

    val_assim, val_real, val_rmse, val_acc = [], [], [], []

    for i, ic in enumerate(ics):
        seq_real, seq_assim, seq_rmse, seq_acc = var4d_medium_forecast(ic,
                                                                        eval_dataset,
                                                                        B_half,
                                                                        R_inv,
                                                                        1,
                                                                        afnonet,
                                                                        dt,
                                                                        spin_up_hours,
                                                                        forecast_hours,
                                                                        clim,
                                                                        mult,
                                                                        mask,
                                                                        var_idx,
                                                                        device)
        if i == 0:
            val_assim = seq_assim
            val_rmse = seq_rmse
            val_acc = seq_acc
        else:
            val_assim = np.concatenate((val_assim, seq_assim), axis=0)
            val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
            val_acc = np.concatenate((val_acc, seq_acc), axis=0)

    for i in range(val_rmse.shape[-1]):
        print(f"RMSE of {var_idx[i]} is： {np.mean(val_rmse, axis=0)[:, i]}")
        print(f"ACC of {var_idx[i]} is： {np.mean(val_acc, axis=0)[:, i]}")

    # np.save(f"{output_dir}/rmse_4dvar_obs{obs_vars}.npy", val_rmse)
    # np.save(f"{output_dir}/acc_4dvar_obs{obs_vars}.npy", val_acc)

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--data_dir',
        type=str,
        help='path of the validation data',
        default="/public/home/wangwuxing01/research/fundation/data/era5_6_hourly"
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help='path for pretrain prediction models',
        default='/public/home/wangwuxing01/research/fundation/ckpts/vit_base.ckpt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path for output',
        default='/public/home/wangwuxing01/research/fundation/output/medium_forecast'
    )

    parser.add_argument(
        '--init_time',
        type=int,
        help='init lead times [d] of the first background fields',
        default=3
    )

    parser.add_argument(
        '--dt',
        type=int,
        help='one time step of the forecast model',
        default=6
    )

    parser.add_argument(
        '--spin_up_step',
        type=int,
        help='length of the assimilation window [d]',
        default=1
    )

    parser.add_argument(
        '--forecast_step',
        type=int,
        help='length of the assimilation window [d]',
        default=10
    )

    parser.add_argument(
        '--decorrelation_step',
        type=int,
        help='decoorelation between each initial time [d]',
        default=20
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='mode of data',
        default='val'
    )

    parser.add_argument(
        '--obs_vars',
        type=int,
        nargs="+",
        help='observation variables used to do assimilation, '
             'avail_obs=["r@50", "r@500", "r@850", "r@1000", "t@50", "t@500", "t@850", "t@1000", "u10", "v10"]',
        # avail_obs=["r@50", "r@500", "r@850", "r@1000", "t@50", "t@500", "t@850", "t@1000", "u10", "v10"]
        required=True
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    pretrain_ckpt = args.pretrain_dir
    output_dir = args.output_dir
    init_time = args.init_time
    dt = args.dt
    spin_up_step = args.spin_up_step
    forecast_step = args.forecast_step
    decorrelation_step = args.decorrelation_step
    mode = args.mode
    obs_vars = args.obs_vars
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    medium_forecast_inference(data_dir,
                            pretrain_ckpt,
                            output_dir,
                            init_time,
                            dt,
                            spin_up_step,
                            forecast_step,
                            decorrelation_step,
                            mode,
                            obs_vars,
                            device)