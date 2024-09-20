import dask
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
from pathlib import Path
import os
import glob
import sys
sys.path.append(".")
from src.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR, PRESSURE_LEVEL_VARS, SINGLE_LEVEL_VARS
from ffrecord import FileWriter
# from data_factory.graph_tools import fetch_time_features

np.random.seed(2023)
path = f'../data/era5_6_hourly'

def write_dataset(xt0, xt1, y, out_file):
    n_sample = xt0.shape[0]

    # 初始化ffrecord
    writer = FileWriter(out_file, n_sample)

    for item in zip(xt0, xt1, y):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()

def load_ndf(year, split, time_scale, climatology, normalize_mean, normalize_std, variables):
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

            if split == "train":  # compute mean and std of each var in each year
                var_mean_monthly = np_vars[NAME_TO_VAR[var]].mean(axis=(0, 2, 3))
                var_std_monthly = np_vars[NAME_TO_VAR[var]].std(axis=(0, 2, 3))
                if var not in normalize_mean:
                    normalize_mean[NAME_TO_VAR[var]] = [var_mean_monthly]
                    normalize_std[NAME_TO_VAR[var]] = [var_std_monthly]
                else:
                    normalize_mean[NAME_TO_VAR[var]].append(var_mean_monthly)
                    normalize_std[NAME_TO_VAR[var]].append(var_std_monthly)

            clim_monthly = np_vars[NAME_TO_VAR[var]].mean(axis=0)
            if var not in climatology:
                climatology[NAME_TO_VAR[var]] = [clim_monthly]
            else:
                climatology[NAME_TO_VAR[var]].append(clim_monthly)

        else:  # multiple-level variables, only use a subset
            assert len(ds[code].shape) == 4
            all_levels = ds["level"][:].to_numpy()
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            for level in all_levels:
                ds_level = ds.sel(level=[level])
                level = int(level)
                # remove the last 24 hours if this year has 366 days
                np_vars[f"{NAME_TO_VAR[var]}@{level}"] = ds_level[code].to_numpy()

                if split == "train":  # compute mean and std of each var in each year
                    var_mean_monthly = np_vars[f"{NAME_TO_VAR[var]}@{level}"].mean(axis=(0, 2, 3))
                    var_std_monthly = np_vars[f"{NAME_TO_VAR[var]}@{level}"].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[f"{NAME_TO_VAR[var]}@{level}"] = [var_mean_monthly]
                        normalize_std[f"{NAME_TO_VAR[var]}@{level}"] = [var_std_monthly]
                    else:
                        normalize_mean[f"{NAME_TO_VAR[var]}@{level}"].append(var_mean_monthly)
                        normalize_std[f"{NAME_TO_VAR[var]}@{level}"].append(var_std_monthly)

                clim_monthly = np_vars[f"{NAME_TO_VAR[var]}@{level}"].mean(axis=0)
                if f"{NAME_TO_VAR[var]}@{level}" not in climatology:
                    climatology[f"{NAME_TO_VAR[var]}@{level}"] = [clim_monthly]
                else:
                    climatology[f"{NAME_TO_VAR[var]}@{level}"].append(clim_monthly)

    return np_vars, climatology, normalize_mean, normalize_std

def fetch_dataset(cursor_time, out_dir, split, climatology, normalize_mean, normalize_std, variables):
    # load weather data
    step = (cursor_time.year - 2010) * 12 + (cursor_time.month - 1) + 1
    start = cursor_time.strftime('%Y-%m-%d %H:%M:%S')
    if cursor_time.month == 12:
        end = (cursor_time + relativedelta(days=30, hours=23)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        end = (cursor_time + relativedelta(months=1, hours=0)).strftime('%Y-%m-%d %H:%M:%S')

    print(f'Step {step} | from {start} to {end}')

    time_scale = slice(start, end)
    data, climatology, normalize_mean, normalize_std = load_ndf(cursor_time.year, split, time_scale, climatology, normalize_mean, normalize_std, variables)

    var_idx = [k for k in data.keys()]

    np_vars = np.concatenate([data[k].astype(np.float32) for k in var_idx], axis=1)[:,:,:-1,:-1]

    print(f"np_vars.shape: {np_vars.shape}\n")

    write_dataset(np_vars[:-2], np_vars[1:-1], np_vars[2:], out_dir / f"{step:03d}.ffr")
    np.save(f"{path}/ffr_era5/var_idx.npy", np.asarray(var_idx))
    return climatology, normalize_mean, normalize_std

def dump_era5(out_dir, split, variables):
    out_dir.mkdir(exist_ok=True, parents=True)
    normalize_mean = {}
    normalize_std = {}
    climatology = {}
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

        climatology, normalize_mean, normalize_std = fetch_dataset(cursor_time, out_dir, split, climatology, normalize_mean, normalize_std, variables)
        cursor_time += relativedelta(months=1)

    if split == "train":
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            mean, std = normalize_mean[var], normalize_std[var]
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean.mean(axis=0)
            normalize_mean[var] = mean
            normalize_std[var] = std

        np.savez(os.path.join(f"{path}/ffr_era5", "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(f"{path}/ffr_era5", "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(f"{path}/ffr_era5/{spit}.ffr", "climatology.npz"),
        **climatology,
    )

if __name__ == "__main__":
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
    for spit in ['val', 'test', 'train']:
        out_dir = Path(f"{path}/ffr_era5/{spit}.ffr")
        dump_era5(out_dir, spit, variables)
