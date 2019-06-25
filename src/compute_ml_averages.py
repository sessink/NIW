# %% imports
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from tools import compute_mld

import matplotlib.pyplot as plt
# %% Apply to all floata
def integrate_columns(data):
    '''
        Integrate each profile over mixed layer.
    '''
    data = data.where(data.z>=data.mld)
    array = []
    for t in range( data.time.size ):
        array.append( data.isel(time=t).dropna('z').integrate('z'))
    return xr.concat(array,dim='time')

def mlavg_wrapper(input,output):
    file =  str(input)
    data = xr.open_dataset(file)

    # compute ML and averages
    data = compute_mld(data)

    # compute some variables
    data['hke'] = 0.5*( data.u**2 + data.v**2 )
    data['hke_lowpass'] = 0.5*( data.u_lowpass**2 + data.v_lowpass**2 )
    data['hke_resid'] = 0.5*( data.u_resid**2 + data.v_resid**2 )
    #ata['H'] =

    mlavg = integrate_columns(data)

    mlavg.to_netcdf(str(output))

# %% MAIN
mlavg_wrapper(snakemake.input,snakemake.output)

# %% testing
# file = './data/filtered/xr_7788a_grid_filt_5Tf.nc'
# data = xr.open_dataset(file)
#
# # compute ML and averages
# data = compute_mld(data)
#
# # compute some variables
# data['hke'] = 0.5*( data.u**2 + data.v**2 )
# data['hke_lowpass'] = 0.5*( data.u_lowpass**2 + data.v_lowpass**2 )
# data['hke_resid'] = 0.5*( data.u_resid**2 + data.v_resid**2 )
#
#
#
# integrated =
