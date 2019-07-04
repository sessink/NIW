# %% imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cmocean import cm

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 7])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% FUNCTIONS
def vertical_line(axis, x, ymax=1.2):
    for i, ax in enumerate(axis):
        if i == 0:
            ax.axvline(x=x1,
                       ymin=0,
                       ymax=1,
                       c="black",
                       linewidth=1,
                       zorder=0,
                       clip_on=False)
        else:
            ax.axvline(x=x1,
                       ymin=0,
                       ymax=ymax,
                       c="black",
                       linewidth=1,
                       zorder=0,
                       clip_on=False)


# %% read data
path = './data/ml/mlall_9h_6Tf.nc'
dat = xr.open_dataset(path)
met = xr.open_dataset('data/metdata/float_cfs_hourly.nc')

# %%
dat
# %%
f, ax = plt.subplots(3, 1, sharex=True)
for float in dat.float:

    # met.sel(float=float).tx.plot(ax=ax[0])
    # met.sel(float=float).ty.plot(ax=ax[0])
    # ax[0].set_ylim(0,1)
    # dat.eps.pipe(np.log10).plot(ax=ax[1])
    # dat.hke.plot(ax=ax[1],label='HKE')
    # dat.hke_lowpass.plot(ax=ax[1],label='HKE Lowpass',marker='.')
    dat.sel(float=float).hke_resid.plot(ax=ax[1], label='HKE Butterworth')
    dat.sel(float=float).hke_ni.plot(ax=ax[1], label='HKE NI')
    ax[1].legend(loc='best')

    dat.sel(float=float).mld.plot(ax=ax[2])

    x1 = dat.time.isel(time=20).values
    # vertical_line(ax, x1)

# dat.mld.plot(ax=ax[3])
