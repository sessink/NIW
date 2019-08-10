# Scientific Computing
import numpy as np
# import scipy.io as sio
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# My Stuff
from tools import datenum2datetime, load_matfile

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 7])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)
# %% define functions
# vectorize datenum2datetime
dn2dt_vec = np.vectorize(lambda x: datenum2datetime(x))


def convert_metdata(input, output):
    '''RC saved this weirdly, so need to read the two experiments
    separately and then merge'''
    path = './data/metdata/float_cfs_hourly.mat'
    data = load_matfile(str(path))

    dsa = xr.Dataset({  # define wanted variables here!
        'lat': (['floatid', 'time'], data['lat_grid'][:6]),
        'lon': (['floatid', 'time'], data['lon_grid'][:6]),
        'ty': (['floatid', 'time'], data['vtau_grid'][:6]),
        'tx': (['floatid', 'time'], data['utau_grid'][:6]),
        'qlat': (['floatid', 'time'], data['lhtfl_grid'][:6]),
        'qsens': (['floatid', 'time'], data['shtfl_grid'][:6]),
        'uswrf': (['floatid', 'time'], data['uswrf_grid'][:6]),
        'ulwrf': (['floatid', 'time'], data['ulwrf_grid'][:6]),
        'dswrf': (['floatid', 'time'], data['dswrf_grid'][:6]),
        'dlwrf': (['floatid', 'time'], data['dlwrf_grid'][:6]),
    },
        coords={'floatid': data['Floats'][:6],
                'time': data['Jday_gmt_grid'][0,:]}
    )

    dsb = xr.Dataset({  # define wanted variables here!
        'lat': (['floatid', 'time'], data['lat_grid'][6:]),
        'lon': (['floatid', 'time'], data['lon_grid'][6:]),
        'ty': (['floatid', 'time'], data['vtau_grid'][6:]),
        'tx': (['floatid', 'time'], data['utau_grid'][6:]),
        'qlat': (['floatid', 'time'], data['lhtfl_grid'][6:]),
        'qsens': (['floatid', 'time'], data['shtfl_grid'][6:]),
        'uswrf': (['floatid', 'time'], data['uswrf_grid'][6:]),
        'ulwrf': (['floatid', 'time'], data['ulwrf_grid'][6:]),
        'dswrf': (['floatid', 'time'], data['dswrf_grid'][6:]),
        'dlwrf': (['floatid', 'time'], data['dlwrf_grid'][6:]),
    },
        coords={'floatid': data['Floats'][6:],
                'time': data['Jday_gmt_grid'][6,:]}
    )

    dsa['lw'] = dsa.dlwrf - dsa.ulwrf
    dsa['sw'] = dsa.dswrf - dsa.uswrf
    dsa['Qnet'] = -dsa.qlat - dsa.qsens + dsa.sw + dsa.lw
    dsa = dsa.drop(['dlwrf', 'ulwrf', 'dswrf', 'uswrf'])

    dsb['lw'] = dsb.dlwrf - dsb.ulwrf
    dsb['sw'] = dsb.dswrf - dsb.uswrf
    dsb['Qnet'] = -dsb.qlat - dsb.qsens + dsb.sw + dsb.lw
    dsb = dsb.drop(['dlwrf', 'ulwrf', 'dswrf', 'uswrf'])

    dsa = dsa.dropna(dim='time', how='all')
    dsa = dsa.assign_coords(time=(dn2dt_vec(dsa.time)))
    dsa['tau'] = 0.5 * (dsa.tx**2 + dsa.ty**2)

    dsb = dsb.dropna(dim='time', how='all')
    dsb = dsb.assign_coords(time=(dn2dt_vec(dsb.time)))
    dsb['tau'] = 0.5 * (dsb.tx**2 + dsb.ty**2)

    merge = xr.merge([dsa, dsb])
    merge.to_netcdf(str(output))


convert_metdata(snakemake.input, snakemake.output)

# %% plot timeseries of tau

# path = './data/metdata/float_cfs_hourly.mat'
# data = load_matfile(path)
#
# data['Jday_gmt_grid'][].shape
#
# datenum2datetime( np.nanmax( data['Jday_gmt_grid'][6,:].flatten() ))
#
# data

# f,ax = plt.subplots(2,1,sharex=False)
# for i in dsa.floatid:
#     dsa.tau.isel(floatid=i).plot(label=dsa.float[i].values,ax=ax[0])
# ax[0].set_title('2016')
# ax[0].legend()
# ax[0].set_ylim(0,2)
# ax[0].set_xlabel(None)
#
# for i in dsb.floatid:
#     dsb.tau.isel(floatid=i).plot(label=dsb.float[i].values,ax=ax[1])
# ax[1].legend()
# ax[1].set_title('2017')
# ax[1].set_ylim(0,2)
# ax[1].set_xlabel(None)
#
# plt.subplots_adjust()
# plt.savefig('./figures/tau_timeseries.pdf')
# plt.show()

# %% plot timeseries of Q
# f,ax = plt.subplots(2,1,sharex=False)
# for i in dsa.floatid:
#     dsa.Qsfc.isel(floatid=i).plot(label=dsa.float[i].values,ax=ax[0])
# ax[0].set_title('2016')
# ax[0].legend()
# # ax[0].set_ylim(0,2)
# ax[0].set_xlabel(None)
# ax[0].axhline(0,color='k')
#
# for i in dsb.floatid:
#     dsb.Qsfc.isel(floatid=i).plot(label=dsb.float[i].values,ax=ax[1])
#     # dsb.Qsfc.rolling(time=50).mean().isel(floatid=i).plot()
# ax[1].legend()
# ax[1].set_title('2017')
# # ax[1].set_ylim(0,2)
# ax[1].set_xlabel(None)
# ax[1].axhline(0,color='k')
#
# plt.subplots_adjust()
# plt.savefig('./figures/Qsfc_timeseries.pdf')
# plt.show()
