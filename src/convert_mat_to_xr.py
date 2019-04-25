import os,glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as sio
import gsw
from datetime import datetime,timedelta

def load_matfile(file):
    '''Read Matlab structure files and convert to numpy arrays'''
    return sio.loadmat(file,struct_as_record=True,squeeze_me=True)

def datenum2datetime(datenum):
    '''Convert Matlab datenum to Python Datetime'''
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum%1) - timedelta(days = 366)
# vectorize datenum2datetime
dn2dt_vec = np.vectorize(lambda x: datenum2datetime(x))

def save_as_xr(input,output):
    '''Read float files, compose xarray dataset, convert variables, and save as netcdf
    last updated: april 24, 2019
        '''
    a = load_matfile(str(input))
    # compose dataset object
    ds = xr.Dataset({ # define wanted variables here!
        'sigma': (['z','time'],a['A']['Sigma'].flatten()[0]-1000), 
        'u1': (['z','time'],a['A']['u1'].flatten()[0]),
        'u2': (['z','time'],a['A']['u2'].flatten()[0]),
        'v1': (['z','time'],a['A']['v1'].flatten()[0]),
        'v2': (['z','time'],a['A']['v2'].flatten()[0]),
        'n2': (['z','time'],a['A']['N2'].flatten()[0])},
        coords = {'pressure':(['z'],a['A']['Pr'].flatten()[0].astype(float)),
                    'z': a['A']['Pr'].flatten()[0].astype(float),
                    'lat':(['time'],a['A']['lat'].flatten()[0]),
                    'lon':(['time'],a['A']['lon'].flatten()[0]),
                    'time':a['A']['Jday_gmt'].flatten()[0].astype(float)},

        attrs= {'floatid':a['A']['float'].flatten()[0]}
        )
    # remove nans
    ds = ds.dropna(dim='time',how='all')
    # convert to datetime
    ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))
    ds = ds.assign_coords(z=-ds.z)
    # comvert pressure to depth
    ds = ds.assign_coords(z=-(gsw.z_from_p(ds.z,ds.lat.mean()).astype(float)))
    # save as netcdf
    ds.to_netcdf(str(output))
    
save_as_xr(snakemake.input,snakemake.output)
				 