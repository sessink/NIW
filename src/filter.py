# Scientific Computing
import numpy as np
# import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt, lfilter, iirfilter

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import gsw
from tools import compute_mld

sns.set(style='ticks', context='paper')
plt.style.use('sebstyle')

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# sns.set(style='ticks', context='paper')
# mpl.rc('figure', dpi=120, figsize=[10, 7])
# mpl.rc('savefig', dpi=500, bbox='tight')
# mpl.rc('legend', frameon=False)


# %%

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    # y = filtfilt(b, a, data)
    y = filtfilt(b, a, data)
    return y


def make_test_plots(data, outfile):
    f, axs = plt.subplots(6, 1, sharex=True)

    vars = ['u','u_lowpass','u_resid','v','v_lowpass','v_resid']
    for ax,var in zip(axs,vars):
        data[var].dropna(dim='time', how='all').plot(ylim=(-500, 0),
                                               ax=ax,
                                               rasterized=True,
                                               vmin=-1,
                                               vmax=1,
                                               cmap='RdBu_r')
        ax.set_xlabel(None)
    plt.savefig(outfile)


def filter_variables(data, resample_period, filter_period,
                     order):
    '''
    apply filter to multiple variables
    '''

    sampling_period = np.int(resample_period.split('h')[0])
    fs = 1 / (3600 * sampling_period)  # sample rate, Hz
    f = gsw.f(39)  # inertial frequency
    Tf = 2 * np.pi / f  # inertial period
    # desired cutoff frequency of the filter, Hz
    cutoff = 1 / (Tf * filter_period)
    # loop over depths
    bucket = []
    for z in range(len(data.z)):
        dat = data.isel(z=z).dropna(dim='time')
        # temp = dat[var].fillna(0)
        if dat.count() > 21:
            filtered = butter_lowpass_filter(dat, cutoff, fs, order)
            bucket.append(
                xr.DataArray(filtered, coords=[dat.time], dims=['time']))
        else:
            bucket.append(
                xr.DataArray(np.ones(dat.time.size) * np.nan,
                             coords=[dat.time],
                             dims=['time']))
    ds = xr.concat(bucket, data.z)
    return ds

def filter_wrapper(input, figureoutput, dataoutput, resample_period,
                   filter_period):
    '''
    Apply to all floata
    '''
    order = 3
    data = xr.open_dataset(str(input))

    # filter u and v

    data['u_lowpass'] = filter_variables(data['u'], resample_period, filter_period, order=order)
    data['v_lowpass'] = filter_variables(data['v'], resample_period, filter_period, order=order)

    data['u_resid'] = data.u.where(data.u.notnull()) - data.u_lowpass.where(data.u_lowpass.notnull())
    data['v_resid'] = data.v.where(data.v.notnull()) - data.v_lowpass.where(data.v_lowpass.notnull())

    data = data.dropna(dim='time', how='all')
    data = compute_mld(data)

    make_test_plots(data, str(figureoutput))
    data.to_netcdf(str(dataoutput))


# %%
filter_wrapper(snakemake.input, snakemake.output[0], snakemake.output[1],
               snakemake.config['resample_period'],
               snakemake.config['filter_period'])

# # %% TESTING
#
# try:
#     res = res.drop(['u_lowpass', 'v_lowpass'])
# except:
#     print('variables not found.')
#
# resample_period = '3h'
# filter_period = 4
# order = 6
# vars = ['u', 'v']
# for var in vars:
#     res = filter_variables(res, var, resample_period, filter_period, order=order)
#
#
# res.u.plot(ylim=(-500, 0),figsize=(10, 3),vmin=-1,vmax=1,cmap='RdBu_r')
# res.u_lowpass.plot(ylim=(-500, 0),figsize=(10, 3),vmin=-1,vmax=1,cmap='RdBu_r')
# res.u_resid.plot(ylim=(-500, 0),figsize=(10, 3),vmin=-1,vmax=1,cmap='RdBu_r')
