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

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[8, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %% compute some stuff

def load_data(infiles):
    data = xr.open_dataset('data/ml/ml_7700b_3h_2Tf.nc')
    # data2 = xr.open_dataset('data/xarray/qc_ww_7700b.nc')
    data = xr.open_dataset(str(infiles[0]))
    data2 = xr.open_dataset(str(infiles[1]))
    data2['mldepth'] = ('mettime', data2.mld.interp(time=data2.mettime))

    data2['taudotu'] = data2.utau / 1020/data2.mldepth
    data2['tau'] = np.sqrt( data2.tx**2+data2.ty**2 )
    data['dHKEdt_resid'] = data.hke_resid.differentiate('time', datetime_unit='s')
    data['dHKEdt_lowpass'] = data.hke_lowpass.differentiate('time',
                                                            datetime_unit='s')
    data['dHKEdt_total'] = data.hke.differentiate('time', datetime_unit='s')
    return data,data2

def plot_timeseries(data,data2,outfile):
    f = plt.figure(constrained_layout=True)
    widths = [1.61]
    heights = [1, 1.61]
    spec = f.add_gridspec(ncols=1, nrows=2, width_ratios=widths, height_ratios=heights)
    # f,ax = plt.subplots(2,1,sharex=True)

    ax1 = f.add_subplot(spec[0, 0])
    data.hke_resid.plot(ax=ax1,lw=1,marker='.',label=r'$\widetilde{HKE}$',markersize=3)
    ax1.fill_between(data.time.values,
                     0.3,
                     0,
                     where=data.hke_resid > 0.05,
                     color='r',
                     alpha=0.6,label=r'$\widetilde{HKE}$ >0.05')
    ax1.set_ylim(0,0.3)
    ax1.set_ylabel(r'$\widetilde{HKE}$ [m$^{2}~s^{-2}$]')
    ax1.set_title(f'Float {data2.floatid}')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.annotate('A', (0, 1.02),
                 xycoords='axes fraction',
                 weight='bold')

    ax2 = ax1.twinx()
    ax2.set_ylim(0,2)
    ax2.invert_yaxis()

    data2.tau.plot(lw=0,marker='.',markersize=3,color='g',label=r'$|\tau|$')
    ax2.set_ylabel(r'wind stress $|\tau|$')


    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.2,1.2))

    ax = f.add_subplot(spec[1, 0], sharex=ax1)
    data.eps.plot(lw=0, marker='+', label=r'$\epsilon$',ax=ax,markersize=3)
    data.dHKEdt_resid.where(data.dHKEdt_resid > 0).plot(lw=0,
                                                        marker='o',
                                                        label=r'$\frac{\partial \widetilde{HKE}}{\partial t}$>0',
                                                        fillstyle='none',
                                                        ax=ax,markersize=3)
    data.dHKEdt_resid.where(data.dHKEdt_resid < 0).pipe(np.abs).plot(
        lw=0, marker='o', label=r'$\frac{\partial \widetilde{HKE}}{\partial t}$<0',
        fillstyle='none',ax=ax,markersize=3)
    data2.taudotu.where(data2.taudotu > 0).plot(lw=0,
                                          marker='.',
                                          label=r'$u\cdot\tau>0$',
                                          ax=ax,markersize=3)
    data2.taudotu.where(data2.taudotu < 0).pipe(np.abs).plot(lw=0,
                                                       marker='.',
                                                       label=r'$u\cdot\tau<0$',
                                                       ax=ax,markersize=3)

    ax.fill_between(data.time.values,
                     1e-1,
                     0,
                     where=data.hke_resid > 0.05,
                     color='r',
                     alpha=0.6)

    ax.annotate('B', (0, 1.02),
                 xycoords='axes fraction',
                 weight='bold')

    ax.set_yscale('log')
    ax.set_ylabel(r'wind work [m$^{2}~s^{-3}$]')
    ax.set_ylim(1e-9, 1e-1)
    ax.legend(bbox_to_anchor=(1,1))
    plt.setp([ax1, ax], xlabel=None)
    plt.savefig(str(outfile))

# %%
data,data2 = load_data(snakemake.input)
plot_timeseries(data,data2,snakemake.output)
