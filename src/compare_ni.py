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
def plot_currents(infile, outfile):
    data = xr.open_dataset(str(infile))
    id = str(infile).split('_')[1].split('.')[0]
    data['rho0'] = data.rho0 - 1000

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


def plot_hke(infile, outfile):
    mpl.rc('figure', dpi=120, figsize=[8.5, 5.5])
    data = xr.open_dataset(str(infile))
    id = str(infile).split('_')[1].split('.')[0]
    data['rho0'] = data.rho0 - 1000

    data['hke_lowpass'] = 0.5 * (data.u_lowpass**2 + data.v_lowpass**2)
    data['hke_resid'] = 0.5 * (data.u_resid**2 + data.v_resid**2)
    data['hke_ni'] = 0.5 * (data.uni**2 + data.vni**2)

    var = ['hke_lowpass', 'hke_resid', 'hke_ni']
    f, ax = plt.subplots(len(var), 1, sharex=True)
    for i, ax in enumerate(ax):

        if 'lowpass' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot(ax=ax, rasterized=True,
                              cbar_kwargs={'pad': 0.01},
                              vmin=0, vmax=.3, cmap=cm.amp)

        else:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot(ax=ax, rasterized=True,
                              cbar_kwargs={'pad': 0.01},
                              vmin=0, vmax=.3, cmap=cm.amp)

        ax.set_xticks(pd.date_range(data.time.min().values,
                                    data.time.max().values, freq='M',))
        ax.set(ylim=[-500, 0], title=var[i], xlabel=None)
        ax.set_title('')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(str(outfile))


# # %% MAIN
plot_currents(snakemake.input, snakemake.output[0])
plot_hke(snakemake.input, snakemake.output[1])

# %% TESTING
#
# infile = 'data/filtered/filt_7780b_9h_6Tf.nc'
# data = xr.open_dataset(str(infile))
#
# data.uni.plot()
# plot_data(infile,'test.pdf')
