import glob
import os
from datetime import datetime, timedelta

# # mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cmocean import cm

# %% set up figures
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 7])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
dir = glob.glob('./data/ml/ml_????a*.nc')
all = []
floatid = []
for file in dir:
    data = xr.open_dataset(file)
    floatid.append(file.split('/')[3].split('_')[1][0:][:4])

    Tf = 2 * np.pi / gsw.f(40) / 3600
    data['latl'] = data.lat
    data['lonl'] = data.lon
    dat = data.resample(time='18h').mean()
    dat = dat.assign_coords(lon=dat.lonl)
    dat = dat.assign_coords(lat=dat.latl)
    # dat = dat.assign_coords(floatid=floatid)
    dat = dat.drop(['latl', 'lonl'])

    # dat['ang_lowpass'] = np.arctan2(dat.v_lowpass,dat.u_lowpass)
    dat['ang_resid'] = np.arctan2(dat.v_resid, dat.u_resid)
    all.append(dat)

allt = xr.concat(all, dim='floatid')

# %%
allt['floatid'] = allt.floatid
allt = allt.assign_coords(floatid=np.arange(6))
allt
# %%
f, ax = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11))
allt.hke.sortby('floatid').plot(ax=ax[0], cmap=cm.amp)
ax[0].set_xlabel('')
ax[0].set_yticks(np.arange(0, 7))
ax[0].set_yticklabels(floatid)
allt.hke_lowpass.sortby('floatid').plot(ax=ax[1], cmap=cm.amp)
ax[1].set_xlabel('')
ax[1].set_yticks(np.arange(0, 7))
ax[1].set_yticklabels(floatid)
allt.hke_resid.sortby('floatid').plot(ax=ax[2], cmap=cm.amp)
ax[2].set_yticks(np.arange(0, 7))
ax[2].set_yticklabels(floatid)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.savefig('./figures/hke_all_floats_a_ts.pdf')
plt.show()

# %%
temp = allt[['u', 'v']]
temp
temp = temp.dropna('time', how='all')
temp
# %%
store = []
for t in range(allt.time.size):
    if allt.isel(time=t).isnull().sum().u == 2:
        store.append(t)

# %%
color = sns.color_palette("GnBu_d", n_colors=len(store))

f, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True, sharex=True)
for i, t in enumerate(store):
    temp = allt.isel(time=t)
    ax.quiver(np.arange(6),
              np.zeros(6) + 2 * t,
              temp.u_resid,
              temp.v_resid,
              color=color[i])
    ax.set_xlim(1, 6)

    # ax.set_ylim(-0.02,0.02)
plt.show()
