# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumtrapz

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 15])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def plot_timeseries(dat, met, float):

    f, ax = plt.subplots(11, 1, sharex=True)
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

    dat.mld.plot(ax=ax[2], label=r'mld (0.03kgm$^{-3}$ from $\rho_{10m}$)')
    ax[2].legend()
    ax[2].set_xlabel(None)

    dat.hke.plot(ax=ax[3],marker='.', label='total hke')
    dat.hke_lowpass.plot(ax=ax[3],marker='.', label='lowpass hke')
    ax[3].legend()
    ax[3].set_xlabel(None)

    dat.hke_resid.plot(ax=ax[4], label='resid hke')
    dat.hke_ni.plot(ax=ax[4], label='ni hke')
    ax[4].legend()
    ax[4].set_xlabel(None)

    dat.SHKEdt_total.plot(ax=ax[5], label='total hke')
    dat.SHKEdt_lowpass.plot(ax=ax[5], label='lowpass hke')
    ax[5].legend()
    ax[5].set_xlabel(None)

    dat.SHKEdt_resid.plot(ax=ax[6], label='resid hke')
    dat.SHKEdt_ni.plot(ax=ax[6], label='ni hke')
    ax[6].legend()
    ax[6].set_xlabel(None)


    dat.dHKEdt_total.pipe(np.log10).plot(ax=ax[7],
                                         lw=0,
                                         marker='.',
                                         label='total dhke/dt')
    dat.dHKEdt_lowpass.pipe(np.log10).plot(ax=ax[7],
                                           lw=0,
                                           marker='.',
                                           label='lowpass dhke/dt')
    ax[7].legend()
    ax[7].set_xlabel(None)

    dat.dHKEdt_resid.pipe(np.log10).plot(ax=ax[8],
                                         lw=0,
                                         marker='.',
                                         label='resid dhke/dt')
    dat.dHKEdt_ni.pipe(np.log10).plot(ax=ax[8],
                                      lw=0,
                                      marker='.',
                                      label='ni dhke/dt')
    ax[8].legend()
    ax[8].set_xlabel(None)

    dat.eps.pipe(np.log10).plot(ax=ax[9],
                                marker='.',
                                lw=0,
                                label=r'$log_{10}(\epsilon)$')
    # ax[7].set_ylim(-8,0)
    ax[9].legend()
    ax[9].set_xlabel(None)


    dat.u_resid.plot(ax=ax[10],marker='.',label='total dhke/dt')
    dat.v_resid.plot(ax=ax[10],marker='.',label='total dhke/dt')

    ax[10].legend()
    ax[10].set_xlabel(None)

    ax[10].set_xlim(dat.mld.time.values.min(), dat.mld.time.values.max())
    plt.tight_layout()
    plt.show()
    # plt.savefig(str(output))


# %%
infile = 'data/ml/ml_7784b_9h_6Tf.nc'
metfile = 'data/metdata/float_cfs_hourly.nc'
data = xr.open_dataset(infile)
met = xr.open_dataset(metfile)

data['dHKEdt_resid'] = data.hke_resid.differentiate('time', datetime_unit='s')
data['dHKEdt_ni'] = data.hke_ni.differentiate('time', datetime_unit='s')
data['dHKEdt_lowpass'] = data.hke_lowpass.differentiate('time', datetime_unit='s')
data['dHKEdt_total'] = data.hke.differentiate('time', datetime_unit='s')

timeslice = slice('2017-10-20', '2017-11-10')

stormdat = data.sel(time=timeslice)
stormmet = met.sel(time=timeslice)

stormdat['SHKEdt_resid'] = ('time',
                            cumtrapz(stormdat.hke_resid,
                                     dx=9 * 3600,
                                     initial=0))
stormdat['SHKEdt_ni'] = ('time',
                         cumtrapz(stormdat.hke_ni, dx=9 * 3600, initial=0))
stormdat['SHKEdt_lowpass'] = ('time',
                              cumtrapz(stormdat.hke_lowpass,
                                       dx=9 * 3600,
                                       initial=0))
stormdat['SHKEdt_total'] = ('time',
                            cumtrapz(stormdat.hke, dx=9 * 3600, initial=0))

# %%

plot_timeseries(stormdat, stormmet, '7784b')
