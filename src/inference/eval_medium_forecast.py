import sys
sys.path.append(".")
from pathlib import Path
import pickle
import numpy as np
import torch
from src.datamodules.components.assim_datasets import Assim
from src.datamodules.assim_datamodule import AssimDataModule
from src.models.fdvarformer_module import FDVarFormerLitModule
from src.models.fdvarnet_module import FDVarNetLitModule
from src.models.fourcastnet_module import FourCastNetLitModule
from src.models.map_module import MapLitModule
from src.inference.inference import medium_forecast
from src.da_methods.var4d import Solve_Var4D
import matplotlib as plt
import argparse

def medium_forecast_inference(data_dir,
                                pretrain_ckpt,
                                output_dir,
                                init_time,
                                dt,
                                spin_up_step,
                                forecast_step,
                                decorrelation_step,
                                mode,
                                da_method,
                                obs_vars,
                                device):

    data_dir = f"{data_dir}/assim_dir_{init_time}day"

    spin_up_hours = 6 * spin_up_step
    forecast_hours = 6 * forecast_step
    decorrelation_step = decorrelation_step
    start_id = 20 - init_time * 4

    afnonet = FourCastNetLitModule.load_from_checkpoint(f"/public/home/wangwuxing01/research/fundation/ckpts/fourcastnet.ckpt")
    afnonet = afnonet.net.to(device).eval()

    if da_method == "4dvarformer":
        module = FDVarFormerLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    elif da_method == "4dvarnet":
        module = FDVarNetLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    elif da_method == "vit":
        module = MapLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    module.reset_mask(obs_vars)
    # mean = module.mean
    # std = module.std
    var_idx = module.var_idx
    mult = module.mult
    mask = module.mask
    clim = module.clims[2]

    module.requires_grad = False
    module.reset_mask(obs_vars)
    module.to(device, dtype=torch.float32).eval()

    eval_dataset = Assim(data_dir=data_dir, split=mode, modelname="fourcastnet")

    # 取初始场
    n_samples = eval_dataset.__len__() - (spin_up_step + forecast_step)
    stop = n_samples
    ics = np.arange(start_id, stop, decorrelation_step)

    val_assim, val_real, val_rmse, val_acc = [], [], [], []

    for i, ic in enumerate(ics):
        seq_real, seq_assim, seq_rmse, seq_acc = medium_forecast(ic,
                                                                eval_dataset,
                                                                da_method,
                                                                module,
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

    np.save(f"{output_dir}/rmse_{da_method}_obs{obs_vars}_xb_{init_time}day.npy", val_rmse)
    np.save(f"{output_dir}/acc_{da_method}_obs{obs_vars}_xb_{init_time}day.npy", val_acc)

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
        # default="logs/train/runs/2023-10-16_15-52-49/checkpoints/last.ckpt"
        default='/public/home/wangwuxing01/research/fundation/ckpts/4dvarformer_base.ckpt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path for output',
        default='/public/home/wangwuxing01/research/fundation/output_npj/medium_forecast'
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
        '--da_method',
        type=str,
        help='method used to do assimilation',
        default='4dvarformer'
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
    da_method = args.da_method
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
                            da_method,
                            obs_vars,
                            device)