import dask
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
from pathlib import Path

from ffrecord import FileWriter
# from data_factory.graph_tools import fetch_time_features
import argparse
import sys
sys.path.append(".")
import os
import glob
from src.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR, PRESSURE_LEVEL_VARS, SINGLE_LEVEL_VARS
np.random.seed(2023)

DATADIR = f'../data/era5_6_hourly'

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

var_idx = []
obs_idx = []

def dataset_to_sample(raw_data, mean, std):
    tmpdata = (raw_data - mean) / std
    if len(tmpdata.shape) == 3:
        xt0 = tmpdata[:, :-1, :-1]
    elif len(tmpdata.shape) == 4:
        xt0 = tmpdata[:,:,:-1,:-1]
    return xt0

def sample_to_dataset(raw_data, mean, std):
    xt0 = raw_data * std + mean

    return xt0

def xb_to_sample(raw_data, mean, std):
    tmpdata = (raw_data - mean) / std
    xt = tmpdata

    return xt

def write_dataset(X_era5, Xb, Obs_wind, Obs_t, Obs_r, out_file):
    n_sample = X_era5.shape[0]

    # 初始化ffrecord
    writer = FileWriter(out_file, n_sample)

    for item in zip(X_era5, Xb, Obs_wind, Obs_t, Obs_r):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()

def load_xb(args, year, month, time_scale):
    # non-constant fields
    ps = glob.glob(os.path.join(DATADIR, "background_dir", f"{year}year_{month}month_pred{args.lead_time}day.nc"))
    xb = xr.open_mfdataset(ps, combine="by_coords", parallel=True).sel(time=time_scale)  # dataset for a single variable

    return xb

def load_era5(year, time_scale, variables):
    np_vars = {}

    # non-constant fields
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            if year == 2021:
                ps = glob.glob(os.path.join(DATADIR, f"202*.nc"))
            else:
                ps = glob.glob(os.path.join(DATADIR, f"{year}.nc"))
        else:
            ps = glob.glob(os.path.join(DATADIR, f"all_surface.nc"))
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

def load_windobs(year, time_scale):
    datas = []
    for k in ['obs_u10', 'obs_v10']:
        tmp = xr.open_mfdataset(f'{DATADIR}/{k}.nc', combine='by_coords').sel(time=time_scale)
        datas.append(tmp)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data

def load_level_obs(year, time_scale, var):
    datas = []
    if year == 2021:
        tmp = xr.open_mfdataset(f'{DATADIR}/{var}_202*.nc', combine='by_coords').sel(time=time_scale)
    else:
        tmp = xr.open_mfdataset(f'{DATADIR}/{var}_{year}.nc', combine='by_coords').sel(time=time_scale)
    datas.append(tmp)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data

def fetch_dataset(args, cursor_time, out_dir, normalize_mean, normalize_std, variables):
    # load weather data
    step = (cursor_time.year - 2010) * 12 + (cursor_time.month - 1) + 1
    start = (cursor_time + relativedelta(days=args.lead_time)).strftime('%Y-%m-%d %H:%M:%S')
    if (cursor_time.year == 2022) and (cursor_time.month == 12):
        end = (cursor_time + relativedelta(days=30, hours=23)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        end = (cursor_time + relativedelta(months=1, days=args.lead_time, hours=23)).strftime('%Y-%m-%d %H:%M:%S')

    print(f'Step {step} | from {start} to {end}')

    time_scale = slice(start, end)
    era5 = load_era5(cursor_time.year, time_scale, variables)

    var_idx = [k for k in era5.keys()]

    X_era5 = np.concatenate([dataset_to_sample(era5[k].astype(np.float32),
                                           normalize_mean[k],
                                           normalize_std[k]) for k in var_idx], axis=1)

    #
    print(f"X_era5.shape: {X_era5.shape}\n")

    xb = load_xb(args, cursor_time.year, cursor_time.month, time_scale)
    Xb = []
    for name in list(xb["var"].values):
        raw = xb["background"].sel(var=name).values
        Xb.append(xb_to_sample(raw, normalize_mean[name], normalize_std[name]))
    Xb = np.stack(Xb, axis=1)
    print(f"Xb.shape: {Xb.shape}\n")

    windobs = load_windobs(cursor_time.year, time_scale)
    Obs_wind = []
    for name in ["u10", "v10"]:
        raw = windobs[name]
        obs = dataset_to_sample(raw, normalize_mean[name], normalize_std[name])
        Obs_wind.append(obs)
        obs_idx.append(name)

    Obs_wind = np.stack(Obs_wind, axis=1)
    print(f"Obs_wind.shape: {Obs_wind.shape}\n")

    obs_t = load_level_obs(cursor_time.year, time_scale, "obs_t")
    Obs_t = []
    for name in ["t"]:
        for level in LEVELS:
            raw = obs_t[name].sel(level=level)
            obs = dataset_to_sample(raw, normalize_mean[f'{name}@{level}'], normalize_std[f'{name}@{level}'])
            Obs_t.append(obs)
            obs_idx.append(f'{name}@{level}')

    Obs_t = np.stack(Obs_t, axis=1)
    print(f"Obs_t.shape: {Obs_t.shape}\n")

    obs_r = load_level_obs(cursor_time.year, time_scale, "obs_r")
    Obs_r = []
    for name in ["r"]:
        for level in LEVELS:
            raw = obs_r[name].sel(level=level)
            obs = dataset_to_sample(raw, normalize_mean[f'{name}@{level}'], normalize_std[f'{name}@{level}'])
            Obs_r.append(obs)
            obs_idx.append(f'{name}@{level}')

    Obs_r = np.stack(Obs_r, axis=1)
    print(f"Obs_r.shape: {Obs_r.shape}\n")

    Xb_save = Xb[: -4]
    print(f"Xb_save.shape: {Xb_save.shape}\n")
    Obs_wind_save = np.stack([Obs_wind[:-4], Obs_wind[1:-3], Obs_wind[2:-2], Obs_wind[3:-1]], axis=1)
    print(f"Obs_wind_save.shape: {Obs_wind_save.shape}\n")
    Obs_t_save = np.stack([Obs_t[:-4], Obs_t[1:-3], Obs_t[2:-2], Obs_t[3:-1]], axis=1)
    print(f"Obs_t_save.shape: {Obs_t_save.shape}\n")
    Obs_r_save = np.stack([Obs_r[:-4], Obs_r[1:-3], Obs_r[2:-2], Obs_r[3:-1]], axis=1)
    print(f"Obs_r_save.shape: {Obs_r_save.shape}\n")
    X_era5_save = np.stack([X_era5[:-4], X_era5[1:-3], X_era5[2:-2], X_era5[3:-1], X_era5[4:]], axis=1)
    print(f"X_era5_save.shape: {X_era5_save.shape}\n")

    write_dataset(X_era5_save, Xb_save, Obs_wind_save, Obs_t_save, Obs_r_save, out_dir / f"{step:03d}.ffr")

    np.save(f"../data/era5_6_hourly/assim_dir_{args.lead_time}day/var_idx.npy", np.asarray(var_idx))
    np.save(f"../data/era5_6_hourly/assim_dir_{args.lead_time}day/obs_idx.npy", np.asarray(obs_idx))

def dump_assim(args, out_dir, split, normalize_mean, normalize_std, variables):
    out_dir.mkdir(exist_ok=True, parents=True)

    if split == 'train':
        start_time = datetime(2010, 1, 1, 0, 0)
        end_time = datetime(2020, 1, 1, 0, 0)
    elif split == 'val':
        start_time = datetime(2020, 1, 1, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 0)
    else:
        start_time = datetime(2021, 1, 1, 0, 0)
        end_time = datetime(2023, 1, 1, 0, 0)

    cursor_time = start_time
    while True:
        if cursor_time >= end_time:
            break

        fetch_dataset(args, cursor_time, out_dir, normalize_mean, normalize_std, variables)
        cursor_time += relativedelta(months=1)

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

    normalize_mean = np.load(f"{DATADIR}/ffr_era5/normalize_mean.npz")
    normalize_std = np.load(f"{DATADIR}/ffr_era5/normalize_std.npz")

    # for spit in ['val', 'test', 'train']:
    #     out_dir = Path(f"../data/era5_6_hourly/assim_dir_{args.lead_time}day/{spit}.ffr")
    #     dump_assim(args, out_dir, spit, normalize_mean, normalize_std, variables)
    for spit in ['test']:
        out_dir = Path(f"../data/era5_6_hourly/assim_dir_{args.lead_time}day/{spit}.ffr")
        dump_assim(args, out_dir, spit, normalize_mean, normalize_std, variables)
