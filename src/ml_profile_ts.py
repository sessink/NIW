# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

from tools import alphabet

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=120, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% FUNCTIONs



def plot_prof_ts(data, outfile):
    '''plot profile timeseries'''
    import matplotlib.colors as colors

    var = ['n2', 'hke_resid', 'ang', 'ang_br']
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
        zdat.ang_br.plot(ax=ax2,
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


def plot_data(infile, outfile):
    from datetime import datetime
    import gsw

    data = xr.open_dataset(str(infile))
    id = str(infile).split('_')[1].split('.')[0]

    # TODO: need to calculate those earlier somewhere else.
    data['rho0'] = data.rho0 - 1000
    data['hke_resid'] = 0.5 * np.sqrt(data.u_resid**2 + data.v_resid**2)
    data['ang'] = np.arctan2(data.v_resid, data.u_resid)

    ref_time = pd.to_datetime('1/1/2016')
    timestamp = pd.to_datetime(data.time.values)

    Tf = 2 * np.pi / gsw.f(40)
    dt = (timestamp - ref_time) / pd.to_timedelta(1, unit='s') % Tf

    phase_add = (dt.values * gsw.f(40)).astype('float')
    phase_add[phase_add > np.pi] = phase_add[phase_add > np.pi] - 2 * np.pi

    data['ang_br'] = data['ang'] + phase_add
    data['ang_br'] = xr.where(
        data.ang_br > np.pi,
        (data['ang_br'].where(data.ang_br > np.pi) - np.pi), data['ang_br'])
    data['ang_br'] = xr.where(
        data.ang_br < -np.pi,
        2 * np.pi + (data['ang_br'].where(data.ang_br > np.pi)),
        data['ang_br'])

    data['ang'] = np.degrees(data['ang'])
    data['ang_br'] = np.degrees(data['ang_br'])

    # # %%
    #     f,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    #     data.ang.plot(ax=ax[0],ylim=(-500,0))
    #     data.ang_br.plot(ax=ax[1],ylim=(-500,0))
    # %%
    plot_prof_ts(data, str(outfile[0]))
    plot_ts_per_depth(data, str(outfile[1]))


# # %% MAIN
plot_data(snakemake.input, snakemake.output)

# # %% TESTING
# infile = 'data/filtered/filt_7784b_3h_2Tf.nc'
# raw = 'data/xarray/xr_7784b_grid.nc'
# data = xr.open_dataset(infile)

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
