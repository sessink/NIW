# Scientific Computing
# Standard Library
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import circmean

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

import gsw
from src.tools import alphabet

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=120, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def backrotate_phase(data):
    import gsw
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
    return data

# %%
years=['a','b']
for year in years:
    dir = glob('data/ml/ml_????'+year+'_1h_2Tf.nc')

    f,axs = plt.subplots(nrows=len(dir),figsize=(8, 11),sharex=True)
    for ax, file in zip(axs, dir):
        mldata = xr.open_dataset(file)
        mldata = backrotate_phase(mldata)

        ax.fill_between(mldata.time.values, 0, mldata.hke_resid)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1e0)
        # ax.set_title(f'z = {mldata.z.values}', weight='bold')
        ax.set_ylabel(r'HKE (high-passed)')
        ax2 = ax.twinx()

        mldata.ang_br.where(mldata.hke > 0).dropna('time').rolling(time=3).reduce(
             circmean, low=-180, high=180).plot(color='k', ax=ax2, lw=1)
        mldata.where(mldata.hke_resid > 0).ang_br.dropna('time').plot(
            ax=ax2, lw=0, marker='.', ms=2, color='black', ylim=(-180, 180))
        ax2.set_yscale('linear')
        ax2.set_ylabel(r'$\theta$ (backrotated) [deg]')
        ax2.set_title(None)

        if 'b' in year:
            Tf = 2 * np.pi / gsw.f(40)
            tt = mldata.time.sel(time=slice('9/1/2017', '10/1/2017'))
            dt = (tt - tt[0]) / pd.to_timedelta(1, unit='s')
            ax2.plot(tt, dt * gsw.f(40) * 0.98, color='r')

            tt = mldata.time.sel(time=slice('12/1/2017', '1/1/2018'))
            dt = (tt - tt[0]) / pd.to_timedelta(1, unit='s')
            ax2.plot(tt, -dt * gsw.f(40) * 3, color='r')

        ax2.annotate(file.split('_')[1], (0.01, 0.9),
                     xycoords='axes fraction',
                     weight='bold')

        plt.close()
    f.savefig(f'figures/phase/mlavg_{year}.pdf')
