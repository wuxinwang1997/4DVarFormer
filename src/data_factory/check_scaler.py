import xarray as xr
import pickle
import numpy as np

DATADIR = '../data/era5_6_hourly/'

DATANAMES = {
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_temperature': 't2m',
    'mean_sea_level_pressure': 'msl',
    'geopotential': 'z',
    'relative_humidity': 'r',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'total_precipitation': 'tp'
}

DATAVARS = ['u10', 'v10', 't2m', 'msl', 'z', 'r', 't', 'u', 'v', 'tp']

LEVELS = [50, 500, 850, 1000]

if __name__ == '__main__':

    Mean, Std, Climatology = {}, {}, {}
    print('start computing scaler!')
    for k, v in DATANAMES.items():
        if k in ['geopotential',
                   'relative_humidity', 
                   'temperature', 
                   'u_component_of_wind', 
                   'v_component_of_wind']:
            for level in LEVELS:
                raw = xr.open_mfdataset((f'{DATADIR}/2*.nc'), combine='by_coords')
                print(raw[DATANAMES[k]])

        elif k in ['10m_u_component_of_wind',
                   '10m_v_component_of_wind',
                   '2m_temperature',
                   'mean_sea_level_pressure']:
            raw = xr.open_mfdataset((f'{DATADIR}/all_surface.nc'), combine='by_coords')
            print(raw[DATANAMES[k]])

        elif k == 'total_precipitation':
            raw = xr.open_mfdataset((f'{DATADIR}/all_surface.nc'), combine='by_coords')
            print(raw[DATANAMES[k]])