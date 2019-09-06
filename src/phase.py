# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

import gsw

# My Stuff
from src.tools import alphabet

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=120, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% FUNCTIONs
def plot_prof_ts(data, outfile):
    '''plot profile timeseries'''
    import matplotlib.colors as colors

    var = ['n2', 'hke_resid', 'ang_band', 'ang_band_br']
    f, axs = plt.subplots(len(var), 1, sharex=True)
    for i, ax in enumerate(axs):
        # TODO: Add attributes to variables for automatic plotting labels
        data['n2'].attrs = {'long_name': r'N$^2$', 'units': r's$^2$'}
        data['hke_resid'].attrs = {
            'long_name': 'HKE (high-passed)',
            'units': r'm$^2$/s$^2$'
        }
        data['ang'].attrs = {'long_name': r'$\theta$', 'units': r'deg'}
        data['ang_br'].attrs = {
            'long_name': r'Backrotated $\theta$',
            'units': r'deg'
        }
        max, min = data[var[i]].max(), data[var[i]].min()

        if 'n2' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].pipe(np.log10).plot.pcolormesh(
                ax=ax,
                rasterized=True,
                cbar_kwargs={
                    'pad': 0.01,
                    'label': r'N$^2$ [1/s$^{-2}$]'
                },
                vmin=-6,
                vmax=-3,
                cmap='viridis',
                robust=True,
            )

        if 'hke' in var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].pipe(np.log10).plot.pcolormesh(
                ax=ax,
                rasterized=True,
                cbar_kwargs={
                    'pad': 0.01,
                    'label': r'HKE (high-passed ) [m$^2$ s$^{-2}$]'
                },
                vmin=-2,
                vmax=-1,
                cmap='viridis',
                robust=True,
            )

        if 'ang' is var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot.pcolormesh(ax=ax,
                                         rasterized=True,
                                         vmin=-180,
                                         vmax=180,
                                         cmap='RdBu_r',
                                         cbar_kwargs={
                                             'label':
                                             r'Backrotated $\theta$ [deg]',
                                             'pad': 0.01
                                         })

        if 'ang_br' is var[i]:
            data.mld.plot(ax=ax, color='k')
            data[var[i]].plot.pcolormesh(ax=ax,
                                         rasterized=True,
                                         vmin=-180,
                                         vmax=180,
                                         cmap='RdBu_r',
                                         cbar_kwargs={
                                             'label':
                                             r'Backrotated $\theta$ [deg]',
                                             'pad': 0.01
                                         })

        ax.set_xticks(
            pd.date_range(data.time.min().values,
                          data.time.max().values,
                          freq='M'))
        ax.set(ylim=[-500, 0], title=var[i], xlabel=None)
        ax.set_title('')

    alphabet(axs)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(str(outfile))


def plot_ts_per_depth(data, outfile):
    '''plot timeseries for several depths'''
    depths = [10, 20, 50, 100]
    f, axs = plt.subplots(len(depths), 1, sharex=True)

    for ax, depth in zip(axs, depths):
        zdat = data.interp(z=-depth)
        ax.fill_between(zdat.time.values, 0, zdat.hke_resid)
        ax.set_yscale('log')
        ax.set_title(f'z = {zdat.z.values}', weight='bold')
        ax.set_ylabel(r'HKE (high-passed)')
        ax2 = ax.twinx()
        zdat.ang_band_br.plot(ax=ax2,
                              lw=0,
                              marker='.',
                              color='black',
                              ylim=(-180, 180))
        ax2.set_yscale('linear')
        ax2.set_ylabel(r'$\theta$ (backrotated)')
        ax2.set_title(None)

    alphabet(axs)
    ax.set_xticks(
        pd.date_range(data.time.min().values, data.time.max().values,
                      freq='M'))
    # plt.subplots_adjust(hspace=0.1)
    plt.savefig(str(outfile))


def backrotate_phase(data):
    import gsw
    data['ang'] = np.arctan2(data.v_resid, data.u_resid)
    data['s_ang'] = np.arctan2(data.v_resid_dz, data.u_resid_dz)

    data['ang_band'] = np.arctan2(data.v_band, data.u_band)
    data['s_ang_band'] = np.arctan2(data.v_band_dz, data.u_band_dz)


    def backrotate(data,var):
        newvar = var+'_br'
        ref_time = pd.to_datetime('1/1/2016')
        timestamp = pd.to_datetime(data.time.values)
        Tf = 2 * np.pi / gsw.f(40)
        dt = (timestamp - ref_time) / pd.to_timedelta(1, unit='s') % Tf

        phase_add = (dt.values * gsw.f(40)).astype('float')
        phase_add[phase_add > np.pi] = phase_add[phase_add > np.pi] - 2 * np.pi

        data[newvar] = data[var] + phase_add
        data[newvar] = xr.where(
            data[newvar] > np.pi,
            (data[newvar].where(data[newvar] > np.pi) - np.pi), data[newvar])
        data[newvar] = xr.where(
            data[newvar] < -np.pi,
            2 * np.pi + (data[newvar].where(data[newvar] > np.pi)),
            data[newvar])
        return data

    data = backrotate(data,'ang')
    data = backrotate(data,'ang_band')
    data = backrotate(data,'s_ang')
    data = backrotate(data,'s_ang_band')

    return data


def plot_data(infile, outfile):
    from datetime import datetime
    import gsw

    data = xr.open_dataset(str(infile))
    id = str(infile).split('_')[1].split('.')[0]

    # TODO: need to calculate those earlier somewhere else.
    data['rho0'] = data.rho0 - 1000
    data['hke_resid'] = 0.5 * np.sqrt(data.u_resid**2 + data.v_resid**2)

    data = backrotate_phase(data)

    plot_prof_ts(data, str(outfile[0]))
    plot_ts_per_depth(data, str(outfile[1]))


# # %% MAIN
plot_data(snakemake.input, snakemake.output)

# # %% TESTING
infile = 'data/filtered/filt_sh_7781a_1h_2Tf.nc'
raw = 'data/xarray/xr_7784b_grid.nc'
data = xr.open_dataset(infile)
data = backrotate_phase(data)

data.s_ang_br.plot()

# data
# rawdata = xr.open_dataset(raw)
# # plot_data(infile, 'test.pdf')
# data
#
#
# data['uplus'] = data.u+0.2
# data.uplus.interp(z=-100,method='linear').plot(label='interp')
# rawdata.u.interp(z=-100,method='linear').plot(label='raw')
# plt.legend()
# plt.savefig(
