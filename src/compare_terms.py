# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# My Stuff
from tools import alphabet

# set up figure params
sns.set(style='ticks', context='paper', palette='colorblind')
mpl.rc('figure', dpi=100, figsize=[8, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% compute some stuff
def load_data(infiles):
    # data = xr.open_dataset('data/ml/ml_7700b_3h_2Tf.nc')

    # data2 = xr.open_dataset('data/xarray/qc_ww_7700b.nc')
    data = xr.open_dataset(str(infiles[0]))
    data2 = xr.open_dataset(str(infiles[1]))

    # FIXME: This needs to go somewhere else!
    data2['mldepth'] = ('mettime', data2.mld.interp(time=data2.mettime))

    data2['taudotu'] = data2.utau / 1020 / data2.mldepth
    data2['tau'] = np.sqrt(data2.tx**2 + data2.ty**2)
    data['dHKEdt_resid'] = data.hke_resid.differentiate('time',
                                                        datetime_unit='s')
    data['dHKEdt_lowpass'] = data.hke_lowpass.differentiate('time',
                                                            datetime_unit='s')
    data['dHKEdt_total'] = data.hke.differentiate('time', datetime_unit='s')
    data['taudotu'] = ('time', data2.taudotu.interp(mettime=data.time))

    data['floatid'] = data2.floatid

    return data, data2


def scatterplots(data, outfile):
    hke_thres = 0.05

    data['dHKEdt_abs'] = np.sqrt(data.dHKEdt_resid**2)
    data['taudotu_abs'] = np.sqrt(data.taudotu**2)
    f, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].scatter(x='eps', y='dHKEdt_abs', data=data, label='no storm')
    ax[0].scatter(x='eps',
                  y='dHKEdt_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r',
                  label='storm')
    ax[0].set(xlim=(1e-9, 1e-2), ylim=(1e-9, 1e-2), yscale='log', xscale='log')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel(r'$\overline{\epsilon}$')
    ax[0].set_ylabel(r'$\overline{d\mathit{HKE}}$')
    ax[0].legend()

    ax[1].scatter(x='eps', y='taudotu_abs', data=data)
    ax[1].scatter(x='eps',
                  y='taudotu_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r')
    ax[1].set(xlim=(1e-9, 1e-2), ylim=(1e-9, 1e-2), yscale='log', xscale='log')
    ax[1].set_aspect('equal')
    ax[1].set_xlabel(r'$\overline{\epsilon}$')
    ax[1].set_ylabel(r'$\overline{\mathbf{\tau}\cdot\mathbf{u}}$')

    ax[2].scatter(x='dHKEdt_abs', y='taudotu_abs', data=data)
    ax[2].scatter(x='dHKEdt_abs',
                  y='taudotu_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r')
    ax[2].set(xlim=(1e-9, 1e-4), ylim=(1e-9, 1e-4), yscale='log', xscale='log')
    ax[2].set_aspect('equal')
    ax[2].set_xlabel(r'$\overline{d\mathit{HKE}}$')
    ax[2].set_ylabel(r'$\overline{\mathbf{\tau}\cdot\mathbf{u}}$')

    plt.suptitle(f'Float {data.floatid.values}')
    alphabet(ax)
    plt.savefig(str(outfile))


def plot_timeseries(data, data2, outfile):
    f = plt.figure(constrained_layout=True)
    widths = [2]
    heights = [1, 2]
    spec = f.add_gridspec(ncols=1,
                          nrows=2,
                          width_ratios=widths,
                          height_ratios=heights)
    # f,ax = plt.subplots(2,1,sharex=True)

    ax1 = f.add_subplot(spec[0, 0])
    data.hke_resid.plot(ax=ax1,
                        lw=1,
                        marker='.',
                        label=r'$\widetilde{HKE}$',
                        markersize=3)
    ax1.fill_between(data.time.values,
                     0.3,
                     0,
                     where=data.hke_resid > 0.05,
                     color='r',
                     alpha=0.6,
                     label=r'$\widetilde{HKE}$ >0.05')
    ax1.set_ylim(0, 0.3)
    ax1.set_ylabel(r'$\widetilde{HKE}$ [m$^{2}~s^{-2}$]')
    ax1.set_title(f'Float {data2.floatid}')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.annotate('A', (0, 1.02), xycoords='axes fraction', weight='bold')

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 2)
    ax2.invert_yaxis()

    data2.tau.plot(lw=0,
                   marker='.',
                   markersize=3,
                   color='g',
                   label=r'$|\tau|$')
    ax2.set_ylabel(r'wind stress $|\tau|$')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
               labels + labels2,
               loc='upper left',
               bbox_to_anchor=(1.1, 1))

    ax = f.add_subplot(spec[1, 0], sharex=ax1)
    data.eps.plot(lw=0, marker='+', label=r'$\epsilon$', ax=ax, markersize=3)
    data.dHKEdt_resid.where(data.dHKEdt_resid > 0).plot(
        lw=0,
        marker='o',
        label=r'$\frac{\partial \widetilde{HKE}}{\partial t}$>0',
        fillstyle='none',
        ax=ax,
        markersize=3)
    data.dHKEdt_resid.where(data.dHKEdt_resid < 0).pipe(np.abs).plot(
        lw=0,
        marker='o',
        label=r'$\frac{\partial \widetilde{HKE}}{\partial t}$<0',
        fillstyle='none',
        ax=ax,
        markersize=3)
    data.taudotu.where(data.taudotu > 0).plot(lw=0,
                                              marker='.',
                                              label=r'$u\cdot\tau>0$',
                                              ax=ax,
                                              markersize=3)
    data.taudotu.where(data.taudotu < 0).pipe(np.abs).plot(
        lw=0, marker='.', label=r'$u\cdot\tau<0$', ax=ax, markersize=3)

    ax.fill_between(data.time.values,
                    1e-1,
                    0,
                    where=data.hke_resid > 0.05,
                    color='r',
                    alpha=0.6)

    ax.annotate('B', (0, 1.02), xycoords='axes fraction', weight='bold')

    ax.set_yscale('log')
    ax.set_ylabel(r'wind work [m$^{2}~s^{-3}$]')
    ax.set_ylim(1e-9, 1e-4)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.legend(bbox_to_anchor=())
    plt.setp([ax1, ax], xlabel=None)
    plt.tight_layout()
    plt.savefig(str(outfile))


# %%
data, data2 = load_data(snakemake.input)
data.to_netcdf(snakemake.output[0])
plot_timeseries(data, data2, snakemake.output[1])
# scatterplots(data, snakemake.output[2])
