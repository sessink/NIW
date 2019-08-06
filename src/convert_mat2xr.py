# Standard Library
import glob
import os
from datetime import datetime, timedelta

# Scientific Computing
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

import gsw
from tools import datenum2datetime, load_matfile

# %%
# vectorize datenum2datetime
dn2dt_vec = np.vectorize(lambda x: datenum2datetime(x))


def save_as_xr(input, output):
    '''Read float files, compose xarray dataset, convert variables,
        and save as netcdf last updated: june 11, 2019
        '''
    a = load_matfile(str(input))
    output = str(output)
    # u = 0.5 * (a['A']['u1'].flatten()[0] + a['A']['u2'].flatten()[0])
    # v = 0.5 * (a['A']['v1'].flatten()[0] + a['A']['v2'].flatten()[0])
    # dudz = 0.5 * (a['A']['du1dz'].flatten()[0] + a['A']['du2dz'].flatten()[0])
    # dvdz = 0.5 * (a['A']['dv1dz'].flatten()[0] + a['A']['dv2dz'].flatten()[0])
    # eps = np.nanmedian(np.dstack((a['A']['eps1'].flatten()[0],
    #                               a['A']['eps2'].flatten()[0])), axis=2)
    # chi = np.nanmedian(np.dstack((a['A']['chi1'].flatten()[0],
    #                               a['A']['chi2'].flatten()[0])), axis=2)

    # compose dataset object
    ds = xr.Dataset({  # define wanted variables here!
        'sigma': (['z', 'time'], a['A']['Sigma'].flatten()[0]),
        'u1': (['z', 'time'], a['A']['u1'].flatten()[0] ),
        'u2': (['z', 'time'], a['A']['u2'].flatten()[0] ),
        'v1': (['z', 'time'], a['A']['v1'].flatten()[0] ),
        'v2': (['z', 'time'], a['A']['v2'].flatten()[0] ),

        'du1dz': (['z', 'time'], a['A']['du1dz'].flatten()[0] ),
        'du2dz': (['z', 'time'], a['A']['du2dz'].flatten()[0] ),
        'dv1dz': (['z', 'time'], a['A']['dv1dz'].flatten()[0] ),
        'dv2dz': (['z', 'time'], a['A']['dv2dz'].flatten()[0] ),

        'eps1': (['z', 'time'], a['A']['eps1'].flatten()[0] ),
        'eps2': (['z', 'time'], a['A']['eps2'].flatten()[0] ),
        'chi1': (['z', 'time'], a['A']['chi1'].flatten()[0] ),
        'chi2': (['z', 'time'], a['A']['chi2'].flatten()[0] ),

        # variables needed for QC
        'RotP': (['z', 'time'], a['A']['RotP'].flatten()[0]),
        'W': (['z', 'time'], a['A']['W'].flatten()[0]),
        'verr1': (['z', 'time'], a['A']['verr1'].flatten()[0]),
        'verr2': (['z', 'time'], a['A']['verr2'].flatten()[0]),

        'kT1': (['z', 'time'], a['A']['kT1'].flatten()[0]),
        'kT2': (['z', 'time'], a['A']['kT2'].flatten()[0]),

        'T': (['z', 'time'], a['A']['T'].flatten()[0]),
        'S': (['z', 'time'], a['A']['S'].flatten()[0]),
        'n2': (['z', 'time'], a['A']['N2'].flatten()[0])},

        coords={'pressure': (['z'], a['A']['Pr'].flatten()[0].astype(float)),
                'z': a['A']['Pr'].flatten()[0].astype(float),
                'lat': (['time'], a['A']['lat'].flatten()[0]),
                'lon': (['time'], a['A']['lon'].flatten()[0]),
                'time': a['A']['Jday_gmt'].flatten()[0].astype(float)},

        attrs={'floatid': output.split('_')[1].split('.')[0]}
    )
    # remove nans
    ds = ds.dropna(dim='time', how='all')
    # convert to datetime
    ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))

    # comvert pressure to depth
    ds = ds.assign_coords(z=(gsw.z_from_p(ds.z, ds.lat.mean()).astype(float)))
    # ds = ds.assign_coords(z=-ds.z)

    _, index = np.unique(ds.time, return_index=True)
    ds = ds.isel(time=index)

    # convert variables
    p, lon, lat = xr.broadcast(ds.pressure, ds.lon, ds.lat)
    ds['S'] = xr.DataArray(gsw.SA_from_SP(
        ds.S, p, lon, lat), dims=['z', 'time'])
    ds['T'] = xr.DataArray(gsw.CT_from_t(ds.S, ds.T, p), dims=['z', 'time'])
    ds['rho0'] = xr.DataArray(gsw.sigma0(
        ds.S, ds.T) + 1000, dims=['z', 'time'])

    # save as netcdf
    ds.to_netcdf(str(output))


# %%
save_as_xr(snakemake.input, snakemake.output)

# %% testing
# input =  './data/NIWmatdata/7700b_grid.mat'
# a = load_matfile(str(input))
#
# a['A']
#
# u = 0.5*(a['A']['u1'].flatten()[0]+a['A']['u2'].flatten()[0])
# v = 0.5*(a['A']['v1'].flatten()[0]+a['A']['v2'].flatten()[0])
# dudz = 0.5*(a['A']['du1dz'].flatten()[0]+a['A']['du2dz'].flatten()[0])
# dvdz = 0.5*(a['A']['dv1dz'].flatten()[0]+a['A']['dv2dz'].flatten()[0])
# eps =  np.nanmedian( np.dstack( (a['A']['eps1'].flatten()[0],a['A']['eps2'].flatten()[0]) ) ,axis=2)
# chi =  np.nanmedian( np.dstack( (a['A']['chi1'].flatten()[0],a['A']['chi2'].flatten()[0]) ) ,axis=2)
#
# ds = xr.Dataset({ # define wanted variables here!
#     'sigma': (['z','time'],a['A']['Sigma'].flatten()[0]),
#     'u': (['z','time'],u),
#     'v': (['z','time'],v),
#     'dudz': (['z','time'],dudz),
#     'dvdz': (['z','time'],dvdz),
#     'eps': (['z','time'],eps),
#     'chi': (['z','time'],chi),
#     'T': (['z','time'],a['A']['T'].flatten()[0]),
#     'S': (['z','time'],a['A']['S'].flatten()[0]),
#     'n2': (['z','time'],a['A']['N2'].flatten()[0])},
#
#     coords = {'pressure':(['z'],a['A']['Pr'].flatten()[0].astype(float)),
#                 'z': a['A']['Pr'].flatten()[0].astype(float),
#                 'lat':(['time'],a['A']['lat'].flatten()[0]),
#                 'lon':(['time'],a['A']['lon'].flatten()[0]),
#                 'time':a['A']['Jday_gmt'].flatten()[0].astype(float)},
#
#     attrs= {'floatid':a['A']['float'].flatten()[0]}
#     )
# # remove nans
# ds = ds.dropna(dim='time',how='all')
# # convert to datetime
# ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))
#
# # comvert pressure to depth
# ds = ds.assign_coords(z=(gsw.z_from_p(ds.z,ds.lat.mean()).astype(float)))
# # ds = ds.assign_coords(z=-ds.z)
#
# _, index = np.unique(ds.time, return_index=True)
# ds = ds.isel(time=index)
#
# p,lon,lat = xr.broadcast(ds.pressure,ds.lon,ds.lat)
#
# ds['S'] = xr.DataArray(gsw.SA_from_SP(ds.S,p,lon,lat),dims=['z','time'])
# ds['T'] = xr.DataArray(gsw.CT_from_t(ds.S,ds.T,p),dims=['z','time'])
# ds['sigma0'] = xr.DataArray(gsw.sigma0(ds.S,ds.T),dims=['z','time'])

# # compose dataset object
# u = 0.5*(a['A']['u1'].flatten()[0]+a['A']['u2'].flatten()[0])
# v = 0.5*(a['A']['v1'].flatten()[0]+a['A']['v2'].flatten()[0])
# dudz = 0.5*(a['A']['du1dz'].flatten()[0]+a['A']['du2dz'].flatten()[0])
# dvdz = 0.5*(a['A']['dv1dz'].flatten()[0]+a['A']['dv2dz'].flatten()[0])
# eps =  np.nanmedian( np.dstack( (a['A']['eps1'].flatten()[0],a['A']['eps2'].flatten()[0]) ) ,axis=2)
# chi =  np.nanmedian( np.dstack( (a['A']['chi1'].flatten()[0],a['A']['chi2'].flatten()[0]) ) ,axis=2)
#
# ds = xr.Dataset({ # define wanted variables here!
#     'sigma': (['z','time'],a['A']['Sigma'].flatten()[0]-1000),
#     'u': (['z','time'],u),
#     'v': (['z','time'],v),
#     'dudz': (['z','time'],dudz),
#     'dvdz': (['z','time'],dvdz),
#     'eps': (['z','time'],eps),
#     'chi': (['z','time'],chi),
#     'n2': (['z','time'],a['A']['N2'].flatten()[0])},
#
#     coords = {'pressure':(['z'],a['A']['Pr'].flatten()[0].astype(float)),
#                 'z': a['A']['Pr'].flatten()[0].astype(float),
#                 'lat':(['time'],a['A']['lat'].flatten()[0]),
#                 'lon':(['time'],a['A']['lon'].flatten()[0]),
#                 'time':a['A']['Jday_gmt'].flatten()[0].astype(float)},
#
#     attrs= {'floatid':a['A']['float'].flatten()[0]}
#     )
# # remove nans
# ds = ds.dropna(dim='time',how='all')
# # convert to datetime
# ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))
#
# # comvert pressure to depth
# ds = ds.assign_coords(z=(gsw.z_from_p(ds.z,ds.lat.mean()).astype(float)))
# ds = ds.assign_coords(z=-ds.z)
#
# _, index = np.unique(data.time, return_index=True)
# data = data.isel(time=index)
#
# # save as netcdf
# ds.to_netcdf(str(output))
