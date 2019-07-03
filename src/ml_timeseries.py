# %% imports
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# set up figure params
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=100, figsize=[10,7])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

# %% read data
path = './data/ml/ml_7788a_filt_5Tf.nc'
dat = xr.open_dataset(path)
met = xr.open_dataset('data/metdata/float_cfs_hourly.nc')
float = path.split('_')[1]

# %%
f,ax = plt.subplots(2,1,sharex=True)
met.sel(floatid=float).tx.plot(ax=ax[0])
met.sel(floatid=float).ty.plot(ax=ax[0])
ax[0].set_ylim(0,1)
# dat.eps.pipe(np.log10).plot(ax=ax[1])
# dat.hke.plot(ax=ax[1],label='HKE')
# dat.hke_lowpass.plot(ax=ax[1],label='HKE Lowpass',marker='.')
dat.hke_resid.plot(ax=ax[1],label='HKE Residual')
ax[1].legend(loc='best')
# dat.mld.plot(ax=ax[3])
