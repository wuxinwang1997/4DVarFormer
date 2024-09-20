import xarray as xr
import pickle
import numpy as np
import copy

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
    for year in range(2010, 2023):
        raw = xr.open_mfdataset((f'{DATADIR}/{year}.nc'), combine='by_coords')
        t_raw = raw['t']
        longitude = raw["longitude"]
        latitude = raw["latitude"]
        level = raw["level"]
        time = raw["time"]
        t_l_value = t_raw.values
        error = [2, 0.5, 1.4, 2.2]
        for i, _ in enumerate(LEVELS):
            t_l_value[:,i] = t_l_value[:,i] + error[i] * np.random.randn(t_l_value.shape[0], t_l_value.shape[2], t_l_value.shape[3])

        t_obs = xr.DataArray(
                        t_l_value,
                        dims=[ "time", "level", "latitude", "longitude"],
                        coords={
                            'time': time,
                            'level': level,
                            'latitude': latitude,
                            'longitude': longitude
                        },
                        name='t'
                    )

        t_obs.to_netcdf(f'{DATADIR}/obs_t_{year}.nc')