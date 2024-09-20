import xarray as xr
import pickle
import numpy as np

DATADIR = '../data/era5_6_hourly/'

DATANAMES = {
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
                meanv, stdv, clim, samples = 0, 0, 0, 0
                for year in np.arange(2010, 2023):
                    raw = xr.open_mfdataset((f'{DATADIR}/{year}.nc'), combine='by_coords')
                    data = raw.sel(level=level)[v].values
                    sample_per_year = data.shape[0]
                    np.nan_to_num(data, copy=False)
                    samples += sample_per_year
                    meanv += sample_per_year * np.mean(data)
                    stdv += sample_per_year * (np.std(data))**2
                    clim += sample_per_year * np.mean(data, axis=0)
                meanv /= samples
                stdv = np.sqrt(stdv / samples)
                clim /= samples
                Mean[f'{DATANAMES[k]}@{level}'] = meanv
                Std[f'{DATANAMES[k]}@{level}'] = stdv
                Climatology[f'{DATANAMES[k]}@{level}'] = clim
                print(f"Var: {k} | mean: {Mean[f'{DATANAMES[k]}@{level}']}, "
                      f"std: {Std[f'{DATANAMES[k]}@{level}']}, "
                      f"clim: {Climatology[f'{DATANAMES[k]}@{level}']}")

        elif k in ['10m_u_component_of_wind',
                   '10m_v_component_of_wind',
                   '2m_temperature',
                   'mean_sea_level_pressure']:
            raw = xr.open_mfdataset((f'{DATADIR}/all_surface.nc'), combine='by_coords')
            data = raw[v].values
            np.nan_to_num(data, copy=False)
            Mean[DATANAMES[k]] = np.mean(data)
            Std[DATANAMES[k]] = np.std(data)
            Climatology[DATANAMES[k]] = np.mean(data, axis=0)
            print(f'Var: {k} | mean: {Mean[DATANAMES[k]]}, '
                  f'std: {Std[DATANAMES[k]]}, '
                  f'clim: {Climatology[DATANAMES[k]]}')

        elif k == 'total_precipitation':
            raw = xr.open_mfdataset((f'{DATADIR}/all_surface.nc'), combine='by_coords')
            data = raw[v].values * 1e9
            np.nan_to_num(data, copy=False)
            Mean[DATANAMES[k]] = np.mean(data) / 1e9
            Std[DATANAMES[k]] = np.std(data) / 1e9
            Climatology[DATANAMES[k]] = np.mean(data, axis=0) / 1e9
            print(f"Var: {k} | mean: {Mean[DATANAMES[k]]}, "
                  f"std: {Std[DATANAMES[k]]}, "
                  f"clim: {Climatology[DATANAMES[k]]}")

    with open(f"{DATADIR}/scaler.pkl", "wb") as f:
        pickle.dump({
            "mean": Mean,
            "std": Std,
            "clim": Climatology
        }, f)