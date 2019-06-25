import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import butter, freqz, filtfilt
import gsw

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,7])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

# %% design butterworth Filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# %% apply filter to multiple variables
def filter_variables(data_resampled,var,filter_period):
    # Filter requirements.
    order = 6
    fs = 1/(3600*12)       # sample rate, Hz
    f = gsw.f(40)   # inertial frequency
    Tf = 2*np.pi/f # inertial period
    cutoff = 1/(Tf*filter_period)  # desired cutoff frequency of the filter, Hz

    # loop over depths
    bucket=[]
    for z in range(len(data_resampled.z)):
        dat = data_resampled.isel(z=z).dropna(dim='time')
        if dat[var].size>21:
            filtered = butter_lowpass_filter(dat[var], cutoff, fs, order)
            bucket.append(xr.DataArray(filtered, coords=[dat.time] ,dims=['time']))
        else:
            bucket.append( xr.DataArray(
                np.ones(dat.time.size)*np.nan,
                coords=[dat.time] ,
                dims=['time']))
    ds = xr.concat(bucket,data_resampled.z)

    # new variable
    data_resampled[var+'_lowpass'] = ds
    data_resampled[var+'_resid'] = data_resampled[var]-data_resampled[var+'_lowpass']
    return data_resampled

# %% Recompute shears

# def recompute_shears(data_resampled,var):
#     exts = ['_lowpass','_resid']
#     for ext in exts:
#         var = var+ext
#
#
#     return data_resampled

# %% Apply to all floata
def filter_wrapper(input,output,resample_period,filter_period):
    file =  str(input)
    data = xr.open_dataset(file)
    data['latl'] = data.lat
    data['lonl'] = data.lon
    data_resampled = data.resample(time=resample_period).mean().transpose()
    data_resampled = data_resampled.assign_coords(lon=data_resampled.lonl)
    data_resampled = data_resampled.assign_coords(lat=data_resampled.latl)
    data_resampled = data_resampled.drop(['latl','lonl'])

    # filter u and v
    vars = ['u','v']
    for var in vars:
        data_resampled = filter_variables(data_resampled,var,filter_period)
        # data_resampled = recompute_shears(data_resampled,var)

    data_resampled.to_netcdf(str(output))

# %% testing
# file = './data/xarray/xr_7785b_grid.nc'
# data = xr.open_dataset(file)
# data
# resample_period='12h'
#
# data['latl'] = data.lat
# data['lonl'] = data.lon
# data_resampled = data.resample(time=resample_period).mean().transpose()
# data_resampled = data_resampled.assign_coords(lon=data_resampled.lonl)
# data_resampled = data_resampled.assign_coords(lat=data_resampled.latl)
# data_resampled = data_resampled.drop(['latl','lonl'])
#
# # filter u and v
# vars = ['u','v']
# for var in vars:
#     data_resampled = filter_variables(data_resampled,var,filter_period)


# %% MAIN
filter_period = 5 # non-dim, multiples of the inertial period Tf
resample_period = '12h' # datetime format, for xarray resample

filter_wrapper(snakemake.input,
                snakemake.output,
                resample_period,
                filter_period)

# %% Plot the frequency response.
if False:
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    w, h = freqz(b, a, worN=8000)
    plt.plot(3600*0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff*3600, 0.5*np.sqrt(2), 'ko')
    plt.axvline(3600*cutoff, color='k')
    plt.xlim(0, 3600*0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel(r'Frequency [hours$^{-1}$]')
    plt.grid()
    plt.show()
