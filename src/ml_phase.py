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

    f, ax = plt.subplots(4, 1, sharex=True)
    if float != '7782b':

        met = met.sel(floatid=float)
        met['tau'] = np.sqrt(met.tx**2 + met.ty**2)
        # met.tx.plot(ax=ax[0], label=r'$\tau_x$')
        # met.ty.plot(ax=ax[0], label=r'$\tau_y$')
        # met.tau.plot(ax=ax[0], label=r'$\tau$')
        # ax[0].legend()
        # ax[0].set_xlabel(None)
        # ax[0].set_ylabel(r'wind stress [Nm$-2$]')

        met = met.resample(time='9h', skipna=True).mean()
        quiveropts = dict(headlength=0,
                          headwidth=1,
                          scale_units='y',
                          scale=1,
                          color='k')
        qv = ax[0].quiver(met.time.values, np.zeros(met.time.shape), met.tx,
                          met.ty, met.tau, **quiveropts)
        plt.quiverkey(qv, 0.9, 0.8, 0.5, '0.5 Nm$^{-2}$', coordinates='axes')
        ax[0].set_xlabel(None)
        ax[0].set_ylabel(r'$\tau$')
        ax[0].set_ylim(-1, 1)

    quiveropts = dict(headlength=0,
                      headwidth=1,
                      scale_units='y',
                      scale=1,
                      color='k')
    qv = ax[1].quiver(dat.time.values, np.zeros(dat.time.shape), dat.u_resid,
                      dat.v_resid, **quiveropts)
    plt.quiverkey(qv, 0.9, 0.8, 0.5, '0.5 Nm$^{-2}$', coordinates='axes')
    ax[1].legend()
    ax[1].set_xlabel(None)
    # ax[1].set_ylabel('Mixed layer depth [m]')

    dat.ang.plot(ax=ax[2], label=r'ANGLE')
    ax[2].legend()
    ax[2].set_xlabel(None)
    # ax[2].set_ylabel('Mixed layer depth [m]')

    dat.ang_br.plot(ax=ax[3], label=r'ANGLE BR')
    ax[3].legend()
    ax[3].set_xlabel(None)
    # ax[3].set_ylabel('Mixed layer depth [m]')

    ax[-1].set_xlim(dat.mld.time.values.min(), dat.mld.time.values.max())
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f'figures/storms/nov152017/phase_{float:s}.pdf')
    plt.close()


# %%
def slice_metfile(metfile, timeslice):
    met = xr.open_dataset(metfile)
    stormmet = met.sel(time=timeslice)
    return stormmet


def read_float_mlavg(float, timeslice):
    import gsw
    from datetime import timedelta, datetime

    infile = f'data/ml/ml_{float:s}_3h_2Tf.nc'

    data = xr.open_dataset(infile)

    data['ang'] = np.arctan2(data.v_resid, data.u_resid)

    ref_time = datetime.strptime('1/1/2000', '%m/%d/%Y')

    t = (data.time - data.time[0]) / pd.to_timedelta(1, unit='s')
    f = gsw.f(40)
    data['u_br'] = np.real(data.u_resid * np.exp(1j * f * t))
    data['v_br'] = np.real(data.v_resid * np.exp(1j * f * t))
    data['ang_br'] = np.arctan2(data.v_br, data.u_br)

    stormdat = data.sel(time=timeslice)

    return stormdat, stormmet


# %%
metfile = 'data/metdata/float_cfs_hourly.nc'
floatlist = ['7700b', '7701b', '7785b', '7784b', '7780b']
timeslice = slice('2017-10-18', '2017-11-02')

stormmet = slice_metfile(metfile, timeslice)
for float in floatlist:
    stormdat, stormmet = read_float_mlavg(float, timeslice)
    plot_timeseries(stormdat, stormmet, float)

# %%

f, ax = plt.subplots(2, 1, sharex=True)
stormmet = slice_metfile(metfile, timeslice)
for float in floatlist:
    stormdat, stormmet = read_float_mlavg(float, timeslice)
    # if float != '7782b':
    #     stormmet['tau'] = np.sqrt( stormmet.tx**2+stormmet.ty**2 )
    #     stormmet.tau.plot(ax=ax[0])
    stormdat.ang.plot(ax=ax[0])
    ax[0].set_xlabel(None)
    stormdat.ang_br.plot(ax=ax[1])
plt.tight_layout()
plt.savefig('figures/storms/nov152017/br_phases_all_floats.pdf')
