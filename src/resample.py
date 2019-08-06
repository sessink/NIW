# Scientific Computing
import numpy as np
import xarray as xr

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', context='paper')
plt.style.use('sebstyle')


# %% FUNCTIONS
def compute_ni_currents(data):
    '''
    Compute NI currents by taking 0.5*(u_t - u_t+1)
    '''
    uarray = np.zeros(data.u.shape)
    varray = np.zeros(data.u.shape)
    for t, _ in enumerate(data.time[:-1]):
        uarray[:, t] = 0.5 * (data.u.isel(time=t) - data.u.isel(time=t + 1))
        varray[:, t] = 0.5 * (data.v.isel(time=t) - data.v.isel(time=t + 1))

    data['uni'] = (('z', 'time'), uarray)
    data['vni'] = (('z', 'time'), varray)
    return data


def make_test_plots(data, outfile):
    f, ax = plt.subplots(2, 1)
    data.u.dropna(dim='z', how='all').plot(ylim=(-500, 0),
                                           ax=ax[0],
                                           rasterized=True,
                                           vmin=-1,
                                           vmax=1,
                                           cmap='RdBu_r')
    data.v.dropna(dim='z', how='all').plot(ylim=(-500, 0),
                                           ax=ax[1],
                                           rasterized=True,
                                           vmin=-1,
                                           vmax=1,
                                           cmap='RdBu_r')
    plt.savefig(outfile)


def resample_wrapper(infile, figureoutput, dataoutput, resample_period):
    '''
    Apply to all floats
    '''
    data = xr.open_dataset(str(infile))
    data['latl'] = data.lat
    data['lonl'] = data.lon
    data_resampled = data.resample(time=resample_period).mean().transpose()
    data_resampled = data_resampled.assign_coords(lon=data_resampled.lonl)
    data_resampled = data_resampled.assign_coords(lat=data_resampled.latl)
    data_resampled = data_resampled.drop(['latl', 'lonl'])

    # compute near inertial currents (Ren-Chieh's idea)
    data_resampled = compute_ni_currents(data_resampled)

    make_test_plots(data_resampled, str(figureoutput))
    data_resampled.to_netcdf(str(dataoutput))


# %%
resample_wrapper(snakemake.input, snakemake.output[0], snakemake.output[1],
                 snakemake.config['resample_period'])
