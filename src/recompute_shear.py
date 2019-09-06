# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt, iirfilter, lfilter, savgol_filter

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import gsw

sns.set(style='ticks', context='paper')
plt.style.use('sebstyle')


# %%
def derivative(data, var):
    window_length = 5
    polyorder = 3
    test = []
    for i in range(data.time.size):
        z = data[var].isel(time=i).dropna('z').z
        if data[var].isel(time=i).dropna('z').size > window_length:
            savgol = savgol_filter(data[var].isel(time=i).dropna('z'),
                                   window_length,
                                   polyorder,
                                   deriv=1)

            test.append(xr.Dataset({var + '_dz': (['z'], savgol)}, coords={'z':
                                                                           z}))
        else:
            test.append(xr.Dataset({var + '_dz': (['z'], z*np.nan)}, coords={'z':
                                                                           z}))

    data = data.merge(xr.concat(test, dim=data[var].time).transpose())
    return data


def deriv_wrapper(infile, outfile):
    data = xr.open_dataset(str(infile))
    data = derivative(data, 'u_band')
    data = derivative(data, 'u_resid')
    data = derivative(data, 'u_lowpass')

    data = derivative(data, 'v_band')
    data = derivative(data, 'v_resid')
    data = derivative(data, 'v_lowpass')
    return data.to_netcdf(str(outfile))


deriv_wrapper(snakemake.input, snakemake.output)
# %%
# f,ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
# data.u_band_dz.plot(ylim=(-500,0),vmin=-0.01,vmax=0.01,cmap='RdBu_r',ax=ax[0])
# data.u_resid_dz.plot(ylim=(-500,0),vmin=-0.01,vmax=0.01,cmap='RdBu_r',ax=ax[1])
# data.u_lowpass_dz.plot(ylim=(-500,0),vmin=-0.01,vmax=0.01,cmap='RdBu_r',ax=ax[2])
