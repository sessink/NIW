import gsw
import numpy as np
# import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt

from tools import compute_mld

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# sns.set(style='ticks', context='paper')
# mpl.rc('figure', dpi=120, figsize=[10, 7])
# mpl.rc('savefig', dpi=500, bbox='tight')
# mpl.rc('legend', frameon=False)


# %%
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def filter_variables(data_resampled, var, resample_period, filter_period):
    '''
    apply filter to multiple variables
    '''
    # Filter requirements.
    order = 6
    sampling_period = np.int(resample_period.split('h')[0])
    fs = 1 / (3600 * sampling_period)  # sample rate, Hz
    f = gsw.f(39)  # inertial frequency
    Tf = 2 * np.pi / f  # inertial period
    # desired cutoff frequency of the filter, Hz
    cutoff = 1 / (Tf * filter_period)

    # loop over depths
    bucket = []
    for z in range(len(data_resampled.z)):
        dat = data_resampled.isel(z=z).dropna(dim='time')
        if dat[var].size > 21:
            filtered = butter_lowpass_filter(dat[var], cutoff, fs, order)
            bucket.append(
                xr.DataArray(filtered, coords=[dat.time], dims=['time']))
        else:
            bucket.append(
                xr.DataArray(np.ones(dat.time.size) * np.nan,
                             coords=[dat.time],
                             dims=['time']))
    ds = xr.concat(bucket, data_resampled.z)

    # new variable
    data_resampled[var + '_lowpass'] = ds
    data_resampled[var + '_resid'] = data_resampled[var] - \
                                     data_resampled[var + '_lowpass']
    return data_resampled


def filter_wrapper(input, output, resample_period, filter_period):
    '''
    Apply to all floata
    '''
    file = str(input)
    data_resampled = xr.open_dataset(file)

    # filter u and v
    vars = ['u', 'v']
    for var in vars:
        data_resampled = filter_variables(data_resampled, var, resample_period,
                                          filter_period)
    data_resampled = data_resampled.dropna(dim='time', how='all')
    data_resampled = compute_mld(data_resampled)
    data_resampled.to_netcdf(str(output))


# %%
filter_wrapper(snakemake.input, snakemake.output,
               snakemake.config['resample_period'],
               snakemake.config['filter_period'])
