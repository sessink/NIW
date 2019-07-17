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
mpl.rc('figure', dpi=100, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def plot_timeseries(dat, met, float):

    f, ax = plt.subplots(6, 1, sharex=True)
    if float != '7782b':

        met = met.sel(floatid=float)
        met['tau'] = np.sqrt(met.tx**2 + met.ty**2)
        met.tx.plot(ax=ax[0], label=r'$\tau_x$')
        met.ty.plot(ax=ax[0], label=r'$\tau_y$')
        met.tau.plot(ax=ax[0], label=r'$\tau$')
        ax[0].legend()
        ax[0].set_xlabel(None)
        ax[0].set_ylabel(r'wind stress [Nm$-2$]')

        met = met.resample(time='9h', skipna=True).mean()
        quiveropts = dict(headlength=0,
                          headwidth=1,
                          scale_units='y',
                          scale=1,
                          color='k')
        qv = ax[1].quiver(met.time.values, np.zeros(met.time.shape), met.tx, met.ty,met.tau,
                     **quiveropts)
        plt.quiverkey(qv, 0.9, 0.8, 0.5, '0.5 Nm$^{-2}$', coordinates='axes')
        ax[1].set_xlabel(None)
        ax[1].set_ylabel(r'wind stress vectors')
        ax[1].set_ylim(-1,1)

    dat.mld.plot(ax=ax[2], label=r'mld (0.03kgm$^{-3}$ from $\rho_{10m}$)')
    ax[2].legend()
    ax[2].set_xlabel(None)
    ax[2].set_ylabel('Mixed layer depth [m]')

    dat.hke.plot(ax=ax[3],marker='.',lw=0, label='total hke')
    dat.hke_lowpass.plot(ax=ax[3],marker='.',lw=0, label='lowpass hke')
    dat.hke_resid.plot(ax=ax[3], label='resid hke')
    dat.hke_ni.plot(ax=ax[3], label='ni hke')
    ax[3].legend()
    ax[3].set_xlabel(None)
    ax[3].set_ylabel('HKE [m$^2$s$^2$]')

    dat.dHKEdt_total.pipe(np.abs).pipe(np.log10).plot(ax=ax[4],
                                         lw=0,
                                         marker='.',
                                         label='total dhke/dt')
    dat.dHKEdt_lowpass.pipe(np.abs).pipe(np.log10).plot(ax=ax[4],
                                           lw=0,
                                           marker='.',
                                           label='lowpass dhke/dt')
    dat.dHKEdt_resid.pipe(np.abs).pipe(np.log10).plot(ax=ax[4],
                                         marker='.',
                                         label='resid dhke/dt')
    dat.dHKEdt_ni.pipe(np.abs).pipe(np.log10).plot(ax=ax[4],
                                      marker='.',
                                      label='ni dhke/dt')
    ax[4].legend()
    ax[4].set_xlabel(None)
    ax[4].set_ylabel(r'$\frac{\partial HKE}{\partial t}$ [m$^2$s$^3$]')

    dat.eps.pipe(np.abs).pipe(np.log10).plot(ax=ax[5],
                                marker='.',
                                lw=0.1,
                                label=r'$log_{10}(\epsilon)$')
    # ax[7].set_ylim(-8,0)
    ax[5].legend()
    ax[5].set_xlabel(None)
    ax[5].set_ylabel(r'$\epsilon$ [m$^2$s$^3$]')

    ax[-1].set_xlim(dat.mld.time.values.min(), dat.mld.time.values.max())
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f'figures/storms/nov2017/{float:s}.pdf')
    plt.close()

# %%
def slice_metfile(metfile,timeslice):
    met = xr.open_dataset(metfile)
    stormmet = met.sel(time=timeslice)
    return stormmet

def read_float_mlavg(float, timeslice):
    infile = f'data/ml/ml_{float:s}_9h_6Tf.nc'

    data = xr.open_dataset(infile)


    data['dHKEdt_resid'] = data.hke_resid.differentiate('time', datetime_unit='s')
    data['dHKEdt_ni'] = data.hke_ni.differentiate('time', datetime_unit='s')
    data['dHKEdt_lowpass'] = data.hke_lowpass.differentiate('time', datetime_unit='s')
    data['dHKEdt_total'] = data.hke.differentiate('time', datetime_unit='s')


    stormdat = data.sel(time=timeslice)


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
    return stormdat, stormmet

# %%
metfile = 'data/metdata/float_cfs_hourly.nc'
floatlist = ['7700b','7701b','7785b','7784b','7780b']
timeslice = slice('2017-10-18', '2017-11-10')

stormmet = slice_metfile(metfile, timeslice)
for float in floatlist:
    stormdat, stormmet = read_float_mlavg(float, timeslice)
    plot_timeseries(stormdat, stormmet, float)
