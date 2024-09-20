import dask
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
from pathlib import Path
import sys
sys.path.append(".")
from ffrecord import FileWriter
import os
import glob
from src.models.fourcastnet_module import FourCastNetLitModule
from src.utils.weighted_acc_rmse import weighted_rmse
from src.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR, PRESSURE_LEVEL_VARS, SINGLE_LEVEL_VARS
import torch
import argparse

np.random.seed(2023)

path = f'../data/era5_6_hourly'

DATAMAP = {
    'geopotential': 'z',
    'relative_humidity': 'r',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_temperature': 't2m',
    'mean_sea_level_pressure': 'msl',
    'total_precipitation': 'tp'
}

DATAVARS = ['z', 'r', 't', 'u', 'v', 'u10', 'v10', 't2m', 'msl', 'tp']

LEVELS = [50, 500, 850, 1000]

dt = 6
modelname = "fourcastnet"
pretrain_dir = "../ckpts/"
ckpt = "fourcastnet.ckpt"
device = "cuda"

var_idx = []

def dataset_to_sample(raw_data, mean, std):
    tmpdata = (raw_data - mean) / std

    xt0 = tmpdata[:,:,:-1,:-1]
    return xt0

def sample_to_dataset(raw_data, mean, std):
    xt0 = raw_data * std + mean

    return xt0

def load_ndf(year, time_scale, variables):
    np_vars = {}

    # non-constant fields
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            ps = glob.glob(os.path.join(path, f"{year}.nc"))
        else:
            ps = glob.glob(os.path.join(path, f"all_surface.nc"))
        ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True).sel(time=time_scale)  # dataset for a single variable
        code = NAME_TO_VAR[var]

        if len(ds[code].shape) == 3:  # surface level variables
            ds[code] = ds[code].expand_dims("val", axis=1)
            # remove the last 24 hours if this year has 366 days
            np_vars[NAME_TO_VAR[var]] = ds[code].to_numpy()

        else:  # multiple-level variables, only use a subset
            assert len(ds[code].shape) == 4
            all_levels = ds["level"][:].to_numpy()
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            for level in all_levels:
                ds_level = ds.sel(level=[level])
                level = int(level)
                # remove the last 24 hours if this year has 366 days
                np_vars[f"{NAME_TO_VAR[var]}@{level}"] = ds_level[code].to_numpy()

    return np_vars

def load_windobs(time_scale):
    datas = []
    for k in ['obs_u10', 'obs_v10']:
        tmp = xr.open_mfdataset(f'{path}/{k}.nc', combine='by_coords').sel(time=time_scale)
        datas.append(tmp)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data

def load_level_obs(year, time_scale, var):
    datas = []
    tmp = xr.open_mfdataset(f'{path}/{var}_{year}.nc', combine='by_coords').sel(time=time_scale)
    datas.append(tmp)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data

def fetch_dataset(args, cursor_time, normalize_mean, normalize_std, variables):
    # load weather data
    step = (cursor_time.year - 2010) * 12 + (cursor_time.month - 1) + 1
    start = cursor_time.strftime('%Y-%m-%d %H:%M:%S')
    end = (cursor_time + relativedelta(months=11, days=30, hours=23)).strftime('%Y-%m-%d %H:%M:%S')

    print(f'Step {step} | from {start} to {end}')

    time_scale = slice(start, end)
    data = load_ndf(cursor_time.year, time_scale, variables)

    var_idx = [k for k in data.keys()]

    X0 = np.concatenate([dataset_to_sample(data[k].astype(np.float32),
                                           normalize_mean[k],
                                           normalize_std[k]) for k in var_idx], axis=1)

    X_out = np.zeros_like(X0)[args.lead_time * 24 // dt:]

    for i in np.arange(X0.shape[0] - args.lead_time * 24 // dt):
        x = X0[i : i + 1]
        for j in np.arange(args.lead_time * 24 // dt):
            if j == 0:
                x = module(torch.from_numpy(x).to(device, dtype=torch.float32))
            else:
                x = module(x)
        X_out[i: i + 1] = x.detach().cpu().numpy()

    weighted_rmse_ = weighted_rmse(X_out, X0[args.lead_time * 24 // dt:])
    rmse = module.mult * weighted_rmse_

    for i in range(X_out.shape[1]):
        print(f"Rmse of {var_idx[i]} for lead time {args.lead_time} day is： {rmse[i]}")

    ps = glob.glob(os.path.join(path, f"{cursor_time.year}.nc"))
    x = xr.open_mfdataset(ps, combine="by_coords", parallel=True).sel(time=time_scale)
    time = x["time"].values
    lat = x["latitude"].values
    lon = x["longitude"].values

    Xt_nc = xr.DataArray(
        np.concatenate([sample_to_dataset(X0[args.lead_time * 24 // dt:, i:i + 1].astype(np.float32),
                                          normalize_mean[k],
                                          normalize_std[k]) for i, k in enumerate(var_idx)], axis=1),
        dims=['time', 'var', 'lat', 'lon'],
        coords={
            'time': time[args.lead_time * 24 // dt:],
            'var': [var_idx[i] for i in range(X_out.shape[1])],
            'lat': lat[:-1],
            'lon': lon[:-1]
        },
        name='era5'
    )

    xb = np.concatenate([sample_to_dataset(X_out[:, i:i+1].astype(np.float32),
                                        normalize_mean[k],
                                        normalize_std[k]) for i, k in enumerate(var_idx)], axis=1)

    Xb_nc = xr.DataArray(
        xb,
        dims=['time', 'var', 'lat', 'lon'],
        coords={
            'time': time[args.lead_time * 24 // dt:],
            'var': [var_idx[i] for i in range(X_out.shape[1])],
            'lat': lat[:-1],
            'lon': lon[:-1]
        },
        name='background'
    )

    Xb_nc.to_netcdf(f"{path}/tc_dir/xb_{cursor_time.year}_pred{args.lead_time}day.nc")
    Xt_nc.to_netcdf(f"{path}/tc_dir/xt_{cursor_time.year}_pred{args.lead_time}day.nc")

def dump_era5(args, normalize_mean, normalize_std, variables):
    start_time = datetime(2021, 1, 1, 0, 0)
    end_time = datetime(2023, 1, 1, 0, 0)

    cursor_time = start_time
    while True:
        if cursor_time >= end_time:
            break

        fetch_dataset(args, cursor_time, normalize_mean, normalize_std, variables)
        cursor_time += relativedelta(years=1)

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--lead_time',
        type=int,
        help='forecast lead time',
        default=3
    )

    return parser

if __name__ == "__main__":

    parser = prepare_parser()
    args = parser.parse_args()

    variables = [
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure"
    ]

    module = FourCastNetLitModule.load_from_checkpoint(f"{pretrain_dir}/{ckpt}")
    module.net.to(device).eval()
    normalize_mean = np.load(f"{path}/ffr_era5/normalize_mean.npz")
    normalize_std = np.load(f"{path}/ffr_era5/normalize_std.npz")

    dump_era5(args, normalize_mean, normalize_std, variables)
