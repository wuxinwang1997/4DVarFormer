import xarray as xr
import pickle
import numpy as np

DATADIR = '../data/era5_6_hourly/'

DATANAMES = {
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_temperature': 't2m',
    'mean_sea_level_pressure': 'msl',
    'total_precipitation': 'tp',
    'geopotential': 'z',
    'relative_humidity': 'r',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v'}

DATAVARS = ['u10', 'v10', 't2m', 'msl', 'tp', 'z', 'r', 't', 'u', 'v']

LEVELS = [50, 500, 850, 1000]

deg = 180 / np.pi
rad = np.pi / 180

if __name__ == '__main__':
    raw = xr.open_mfdataset((f'{DATADIR}/all_surface.nc'), combine='by_coords')
    u10 = raw['u10']
    v10 = raw['v10']
    rawu10 = raw['u10'].values
    rawv10 = raw['v10'].values
    wspd = np.sqrt(u10.values**2 + v10.values**2)
    wdir = 180 + np.arctan2(u10.values, v10.values)*deg
    wspd_err_low = 2 * np.random.randn(wspd.shape[0], wspd.shape[1], wspd.shape[2])
    wspd_err_high = 0.1 * wspd * np.random.randn(wspd.shape[0], wspd.shape[1], wspd.shape[2])
    mask_err_low = np.where(wspd < 20, 1, 0)
    mask_err_high = np.where(wspd >= 20, 1, 0)
    wspd = (wspd + wspd_err_low) * mask_err_low + (wspd + wspd_err_high) * mask_err_high
    u10.values = -wspd * np.sin(wdir*rad)
    print(f"RMSE of u10: {np.sqrt(np.mean((u10.values-rawu10)**2))}")
    v10.values = -wspd * np.cos(wdir*rad)
    print(f"RMSE of v10: {np.sqrt(np.mean((v10.values-rawv10)**2))}")