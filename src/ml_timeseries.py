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
path = './data/ml/ml_7786b_filt_5Tf.nc'
dat = xr.open_dataset(path)
met = xr.open_dataset('data/metdata/float_cfs_hourly.nc')
float = path.split('_')[1]
#%%


met.sel(floatid=float).dswrf.plot()
met.sel(floatid=float).uswrf.plot()

met.sel(floatid=float).dlwrf.plot()
met.sel(floatid=float).ulwrf.plot()

met.sel(floatid=float).lhtfl.plot()
met.sel(floatid=float).shtfl.plot()

met['lw'] = met.dlwrf-met.ulwrf
met['sw'] = met.dswrf-met.uswrf
met['Qnet'] = -met.lhtfl-met.shtfl-\
              met.uswrf+met.dswrf-\
              met.ulwrf+met.dlwrf

met.sel(floatid=float).Qnet.plot()
met.sel(floatid=float).Qsfc.plot()

f,ax = plt.subplots(4,1,sharex=True)
met.sel(floatid=float).tau.plot(ax=ax[0])
ax[0].set_ylim(0,0.5)
dat.eps.pipe(np.log10).plot(ax=ax[1])
dat.hke.plot(ax=ax[2],label='HKE')
dat.hke_lowpass.plot(ax=ax[2],label='HKE Lowpass',marker='.')
dat.hke_resid.plot(ax=ax[2],label='HKE Residual')
ax[2].legend(loc=1)
dat.mld.plot(ax=ax[3])
