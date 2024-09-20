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
from src.inference.inference import assim_one_step, assim_vit_step, assim_4dvarnet_step
from src.da_methods.var4d import Solve_Var4D
import matplotlib as plt
import argparse
import xarray as xr

LEVELS = [50, 500, 850, 1000]

def dataset_to_sample(raw_data, mean, std):
    tmpdata = (raw_data - mean) / std
    xt0 = tmpdata

    return xt0

def sample_to_dataset(raw_data, mean, std):
    xt0 = raw_data * std + mean

    return xt0

def medium_forecast_inference(data_dir,
                                pretrain_ckpt,
                                output_dir,
                                init_time,
                                year,
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
    start_id = 0

    afnonet = FourCastNetLitModule.load_from_checkpoint(f"/public/home/wangwuxing01/research/fundation/ckpts/fourcastnet.ckpt")
    afnonet = afnonet.net.to(device).eval()

    if da_method == "4dvarformer":
        module = FDVarFormerLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    elif da_method == "4dvarnet":
        module = FDVarNetLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    elif da_method == "vit":
        module = MapLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    elif da_method == "era5":
        module = FDVarFormerLitModule.load_from_checkpoint(f"{pretrain_ckpt}")

    module.reset_mask(obs_vars)
    var_idx = module.var_idx
    mult = module.mult
    mask = module.mask
    clim = module.clims[2]

    normalize_mean = np.load(
        f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/ffr_era5/normalize_mean.npz")
    normalize_std = np.load(
        f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/ffr_era5/normalize_std.npz")

    mean_list = [normalize_mean[k] for k in var_idx]
    std_list = [normalize_std[k] for k in var_idx]

    mean = np.ones((1, 24, 1, 1))
    std = np.ones((1, 24, 1, 1))

    for i in range(24):
        mean[:, i] *= mean_list[i]
        std[:, i] *= std_list[i]

    module.requires_grad = False
    module.reset_mask(obs_vars)
    module.to(device, dtype=torch.float32).eval()

    with open(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/ibtrack_{year}.pickle", "rb") as file:
        ibtrack = pickle.load(file)
        time = ibtrack["time"]
        name = ibtrack["name"]
        lat = ibtrack["lat"]
        lon = ibtrack["lon"]
        idx = ibtrack["idx"]

    era5s = xr.open_mfdataset(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/xt_{year}.nc",
                                 combine="by_coords", parallel=True)
    xbs = xr.open_mfdataset(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/xb_{year}_pred3day.nc",
                                 combine="by_coords", parallel=True)
    obs_ts = xr.open_mfdataset(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/obs_t_{year}.nc",
                                 combine="by_coords", parallel=True)
    obs_u10s = xr.open_mfdataset(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/obs_u10.nc",
                                  combine="by_coords", parallel=True)
    obs_v10s = xr.open_mfdataset(f"/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir/obs_v10.nc",
                                combine="by_coords", parallel=True)

    seq_pred = np.zeros((len(idx), 29, 24, 160, 160))
    save_name = []
    start_times = []

    for i, id in enumerate(idx):
        if int(time[id][0][-8:-6]) != 0:
            start_time = str(time[id][8 - (int(time[id][0][-8:-6]) % 24) // 3], encoding="utf-8")
            start_times.append(time[id][8 - (int(time[id][0][-8:-6]) % 24) // 3])
            end_time = start_time.replace(" 00", " 18")
            xb = np.expand_dims(xbs.sel(time=start_time)["background"].values, axis=0)
            era5 = np.expand_dims(era5s.sel(time=start_time)["era5"].values, axis=0)
            obs_u10 = obs_u10s.sel(time=slice(start_time, end_time))["u10"].values[:, :-1, :-1]
            obs_u10 = dataset_to_sample(obs_u10, normalize_mean["u10"], normalize_std["u10"])
            obs_v10 = obs_v10s.sel(time=slice(start_time, end_time))["v10"].values[:, :-1, :-1]
            obs_v10 = dataset_to_sample(obs_v10, normalize_mean["v10"], normalize_std["v10"])
            obs_wind = np.expand_dims(np.stack([obs_u10, obs_v10], axis=1), axis=0)
            obs_t = np.expand_dims(obs_ts.sel(time=slice(start_time, end_time))["t"].values[:, :, :-1, :-1], axis=0)
            for j, level in enumerate(LEVELS):
                obs_t[:,:,j] = dataset_to_sample(obs_t[:,:,j], normalize_mean[f"t@{level}"], normalize_std[f"t@{level}"])
        else:
            start_time = str(time[id][0], encoding="utf-8")
            start_times.append(time[id][0])
            end_time = start_time.replace(" 00", " 18")
            xb = np.expand_dims(xbs.sel(time=start_time)["background"].values, axis=0)
            era5 = np.expand_dims(era5s.sel(time=start_time)["era5"].values, axis=0)
            obs_u10 = obs_u10s.sel(time=slice(start_time, end_time))["u10"].values[:, :-1, :-1]
            obs_u10 = dataset_to_sample(obs_u10, normalize_mean["u10"], normalize_std["u10"])
            obs_v10 = obs_v10s.sel(time=slice(start_time, end_time))["v10"].values[:, :-1, :-1]
            obs_v10 = dataset_to_sample(obs_v10, normalize_mean["u10"], normalize_std["u10"])
            obs_wind = np.expand_dims(np.stack([obs_u10, obs_v10], axis=1), axis=0)
            obs_t = np.expand_dims(obs_ts.sel(time=slice(start_time, end_time))["t"].values[:, :, :-1, :-1], axis=0)
            for j, level in enumerate(LEVELS):
                obs_t[:,:,j] = dataset_to_sample(obs_t[:,:,j], normalize_mean[f"t@{level}"], normalize_std[f"t@{level}"])

        print(xb.shape, obs_wind.shape, obs_t.shape)

        if da_method == "4dvarformer":
            xa = assim_one_step(torch.from_numpy((xb - mean) / std),
                                module, torch.from_numpy(obs_wind),
                                torch.from_numpy(obs_t),
                                torch.from_numpy(obs_t), mask, var_idx, device)
        elif da_method == "vit":
            xa = assim_vit_step(torch.from_numpy((xb - mean) / std),
                                module, torch.from_numpy(obs_wind),
                                torch.from_numpy(obs_t),
                                torch.from_numpy(obs_t), mask, var_idx, device)
        elif da_method == "4dvarnet":
            xa = assim_4dvarnet_step(torch.from_numpy((xb - mean) / std),
                                     module, torch.from_numpy(obs_wind),
                                     torch.from_numpy(obs_t),
                                     torch.from_numpy(obs_t), mask, var_idx, device)

        elif da_method == "era5":
            xa = torch.from_numpy((era5 - mean) / std).to(device, dtype=torch.float32)

        seq_pred[i, 0:1] = (xa.cpu().detach() * std) + mean

        for j in range(1, 29):
            x_pred = afnonet(torch.from_numpy((seq_pred[i, j-1:j] - mean) / std).to(device, dtype=torch.float32)).cpu().detach()
            seq_pred[i, j:j+1] = x_pred * std + mean

        save_name.append(name[id])

    seq_pred = xr.DataArray(
        seq_pred,
        dims=['start_time', 'time', 'var', 'lat', 'lon'],
        coords={
            'start_time': start_times,
            'time': np.arange(0, int(29*6), 6),
            'var': [var_idx[i] for i in range(24)],
            'lat': xbs["lat"].values,
            'lon': xbs["lon"].values
        },
        name='tc_pred'
    )

    # print(seq_pred)

    seq_pred.to_netcdf(f"{output_dir}/pred_{da_method}_{year}.nc")


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
        default='/public/home/wangwuxing01/research/fundation/ckpts/4dvarformer_base.ckpt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path for output',
        default='/public/home/wangwuxing01/research/fundation/data/era5_6_hourly/tc_dir'
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
        '--year',
        type=int,
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
    year = args.year
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
                            year,
                            dt,
                            spin_up_step,
                            forecast_step,
                            decorrelation_step,
                            mode,
                            da_method,
                            obs_vars,
                            device)