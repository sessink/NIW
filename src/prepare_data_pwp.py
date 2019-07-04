# %% imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 7])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)
# %% read data
path = './data/xarray/xr_7788a_grid.nc'
dat = xr.open_dataset(path)
met = xr.open_dataset('data/metdata/float_cfs_hourly.nc')
float = path.split('_')[1]
fluxes = met.sel(floatid=float)
# %%

fluxes = fluxes.drop(['tau', 'lat', 'lon', 'Qnet', 'floatid'])
fluxes['precip'] = ('time', np.zeros_like(fluxes.lw))
# fluxes.reset_coords('precip')
fluxes

fluxes = fluxes.assign_coords(time=(fluxes.time - fluxes.time[0]) /
                              np.timedelta64(1, 'D'))

fluxes = fluxes.sel(time=slice(0, 50))

fluxes.to_netcdf('./src/pwp/input_data/niw_' + float + '_met.nc')

#template_flux = xr.open_dataset('./src/pwp/input_data/beaufort_met.nc')
# template_flux

# %%
dat
dat.S.plot(vmin=34, vmax=35)
# dat.rho0.plot(vmin=1022,vmax=1027)

# %%
t = 160
f, ax = plt.subplots(1, 3, sharey=True)
dat.isel(time=t).T.plot(y='z', ax=ax[0])
dat.isel(time=t).S.plot(y='z', ax=ax[1])
dat.isel(time=t).rho0.plot(y='z', ax=ax[2])

profile = dat.isel(time=t)
profile = profile.drop(['eps', 'chi', 'u', 'v', 'sigma', 'dudz', 'dvdz', 'n2', 'time',
                        'pressure', 'lon'])
profile = profile.rename({'T': 't', 'S': 's', 'rho0': 'dens'})

profile = profile.assign_coords(z=profile.z * (-1))

profile.to_netcdf('./src/pwp/input_data/niw_' + float + '_profile.nc')
