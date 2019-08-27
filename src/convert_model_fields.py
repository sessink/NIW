# Standard Library
from datetime import datetime

# Scientific Computing
import numpy as np
import xarray as xr

# Plotting
import seaborn as sns

# %%
dn2dt_vec = np.vectorize(lambda x: datenum2datetime(x))


def convert_cfs(infile, landfile, outfile):
    data = load_matfile(str(infile))

    ds = xr.Dataset({  # define wanted variables here!
        'pres': (['lon', 'lat', 'time'], data['pres_sfc']),
        'temp': (['lon', 'lat', 'time'], data['tmp_sfc']),
        'u10': (['lon', 'lat', 'time'], data['u_wind_10m']),
        'v10': (['lon', 'lat', 'time'], data['v_wind_10m']),
        },

        coords={'lat': (data['Lat'].flatten()),
                'lon': (data['Lon'].flatten()),
                'time': data['Jday_gmt']},
                    )

    ds['pres'] = ds.pres / 100
    ds = ds.dropna(dim='time', how='all')
    # convert to datetime
    ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))
    ds['wspeed'] = np.sqrt(ds.u10**2 + ds.v10**2)

    land = xr.open_dataset(str(landfile))
    land = xr.open_dataset()
    land_mask = land.LAND_L1.isel(time=0).sel(
        lon=slice(ds.lon.min(), ds.lon.max()),
        lat=slice(ds.lat.max(), ds.lat.min()))
    ds = ds.where(land_mask == 0)

    ds.to_netcdf(str(outfile))


convert_cfs(snakemake.input[0], snakemake.input[1], snakemake.output)

# %% TESTING
ds
# infile = 'data/CFS/CFSv2_wind_rh_t_p_2016_2018.mat'
# outfile = 'data/CFS/CFSv2_wind_rh_t_p_2016_2018.nc'
