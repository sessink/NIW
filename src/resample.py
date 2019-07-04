import numpy as np
# import pandas as pd
import xarray as xr

# from scipy.signal import butter, filtfilt
# import gsw

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# sns.set(style='ticks', context='paper')
# mpl.rc('figure', dpi=120, figsize=[10, 7])
# mpl.rc('savefig', dpi=500, bbox='tight')
# mpl.rc('legend', frameon=False)


# %% FUNCTIONS
def compute_ni_currents(data):

    uarray = np.zeros(data.u.shape)
    varray = np.zeros(data.u.shape)
    for t, _ in enumerate(data.time[:-1]):
        uarray[:, t] = 0.5*(data.u.isel(time=t)-data.u.isel(time=t+1))
        varray[:, t] = 0.5*(data.v.isel(time=t)-data.v.isel(time=t+1))

    data['uni'] = (('z', 'time'), uarray)
    data['vni'] = (('z', 'time'), varray)
    return data


def resample_wrapper(input, output, resample_period):
    '''
    Apply to all floata
    '''
    file = str(input)
    data = xr.open_dataset(file)
    data['latl'] = data.lat
    data['lonl'] = data.lon
    data_resampled = data.resample(time=resample_period).mean().transpose()
    data_resampled = data_resampled.assign_coords(lon=data_resampled.lonl)
    data_resampled = data_resampled.assign_coords(lat=data_resampled.latl)
    data_resampled = data_resampled.drop(['latl', 'lonl'])

    # compute near inertial currents (Ren-Chieh's idea)
    data_resampled = compute_ni_currents(data_resampled)

    data_resampled.to_netcdf(str(output))


# %%
resample_wrapper(snakemake.input, snakemake.output,
                 snakemake.config['resample_period'])
