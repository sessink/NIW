# %% IMPORTS
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cmocean import cm

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=120, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% FUNCTIONs
def plot_data(infile, outfile):
    data = xr.open_dataset(str(infile))
    id = str(infile).split('_')[1].split('.')[0]
    data['rho0'] = data.rho0-1000

    var = ['u_lowpass', 'u_resid', 'uni', 'v_lowpass', 'v_resid', 'vni']
    f, ax = plt.subplots(len(var), 1, sharex=True)
    for i, ax in enumerate(ax):
        if 'rho' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot(ax=ax, rasterized=True,
                              cbar_kwargs={'pad': 0.01},
                              vmin=22.5, vmax=27.5,
                              cmap=cm.dense)
        elif 'n2' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].pipe(np.log10).plot(ax=ax, rasterized=True,
                                             cbar_kwargs={'pad': 0.01},
                                             vmin=-5, cmap=cm.amp)
        elif 'resid' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot(ax=ax, rasterized=True,
                              cbar_kwargs={'pad': 0.01}, vmin=-.3, vmax=.3,
                              cmap='RdBu_r')
        elif 'lowpass' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot(ax=ax, rasterized=True,
                              cbar_kwargs={'pad': 0.01},
                              vmin=-.75, vmax=.75, cmap='RdBu_r')

        elif 'ni' in var[i]:
                    data.mld.plot(ax=ax, color='k')
                    data[var[i]].plot(ax=ax, rasterized=True,
                                      cbar_kwargs={'pad': 0.01},
                                      vmin=-.3, vmax=.3, cmap='RdBu_r')
        ax.set_xticks(pd.date_range(data.time.min().values,
                      data.time.max().values, freq='M',))
        ax.set(ylim=[-500, 0], title=var[i], xlabel=None)
        ax.set_title('')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(str(outfile))


# # %% MAIN
plot_data(snakemake.input, snakemake.output)

# %% TESTING
# 
# infile = 'data/filtered/filt_7780b_9h_6Tf.nc'
# data = xr.open_dataset(str(infile))
#
# data.uni.plot()
# plot_data(infile,'test.pdf')
