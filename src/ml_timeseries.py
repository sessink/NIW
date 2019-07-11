# %% imports
# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_timeseries(input, output):

    dat = xr.open_dataset(str(input[0]))
    met = xr.open_dataset(str(input[1]))
    float = str(input[0]).split('_')[1]

    f, ax = plt.subplots(5, 1, sharex=True)
    if float != '7782b':

        met = met.sel(floatid=float)
        met.tx.plot(ax=ax[0], label=r'$\tau_x$')
        met.ty.plot(ax=ax[0], label=r'$\tau_y$')
        ax[0].legend()
        ax[0].set_xlabel(None)

        met = met.resample(time='6h', skipna=True).mean()
        quiveropts = dict(headlength=0,
                          headwidth=1,
                          scale_units='y',
                          scale=15,
                          color='k')
        ax[1].quiver(met.time.values, np.zeros(met.time.shape), met.tx, met.ty,
                     **quiveropts)
        ax[1].set_xlabel(None)

    dat.hke.plot(ax=ax[2], label='total hke')
    dat.hke_lowpass.plot(ax=ax[2], label='lowpass hke')
    ax[2].legend()
    ax[2].set_xlabel(None)

    dat.hke_resid.plot(ax=ax[3], label='resid hke')
    dat.hke_ni.plot(ax=ax[3], label='ni hke')
    ax[3].legend()
    ax[3].set_xlabel(None)

    dat.mld.plot(ax=ax[4], label=r'mld (0.03kgm$^{-3}$ from $\rho_{10m}$)')
    ax[4].legend()
    ax[4].set_xlabel(None)

    ax[4].set_xlim(dat.mld.time.values.min(), dat.mld.time.values.max())
    plt.tight_layout()
    plt.savefig(str(output))


# %% MAIN
plot_timeseries(snakemake.input, snakemake.output)

# %% testing
# dat = xr.open_dataset('data/ml/ml_7785b_9h_6Tf.nc')
#
# plt.figure()
# dat.mld.plot()
# plt.xlim( dat.mld.time.values.min(),dat.mld.time.values.max() )
# dat = xr.open_dataset('data/ml/ml_7788a_9h_6Tf.nc')
# met = xr.open_dataset('data/metdata/float_cfs_hourly.nc')
# float = '7788a'
#
#
# plt.quiver(met.time.values, np.zeros(met.time.shape), met.tx, met.ty,
#            **quiveropts)
# plt.xlim(dat.mld.time.values.min(), dat.mld.time.values.max())
