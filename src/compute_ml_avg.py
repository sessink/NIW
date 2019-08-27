# %% imports
# import numpy as np
# import pandas as pd
import xarray as xr
import numpy as np

# import gsw
# from tools import compute_mld
import matplotlib.pyplot as plt


# %% Apply to all floata
def integrate_columns(data,mld):
    '''
        Integrate each profile over mixed layer.
    '''
    # mld=data.mld
    data = data.where(data.z >= mld)
    data['z'] = data.z * (-1)
    array = []
    for t in range(data.time.size):
        # TODO: could do better here with simpson's rule
        array.append(data.isel(time=t).dropna('z').integrate('z'))

    return xr.concat(array, dim='time')

def mlavg_wrapper(input, output):
    file = str(input)
    data = xr.open_dataset(file)

    # compute some variables
    # TODO: compute these earlier!
    data['hke'] = 0.5 * (data.u**2 + data.v**2)
    data['hke_lowpass'] = 0.5 * (data.u_lowpass**2 + data.v_lowpass**2)
    data['hke_resid'] = 0.5 * (data.u_resid**2 + data.v_resid**2)

    mlavg = xr.Dataset()
    vars = ['hke','hke_resid','hke_lowpass','eps']
    for var in vars:
        mlavg[var] = integrate_columns(data[var],data.mld)

    vars = ['u','u_resid','u_lowpass','v','v_resid','v_lowpass']
    for var in vars:
        mlavg[var] = data[var].mean(axis=0)

    mlavg['mld'] = data.mld

    mlavg['hke'] = -mlavg['hke']/mlavg.mld
    mlavg['hke_resid'] = -mlavg['hke_resid']/mlavg.mld
    mlavg['hke_lowpass'] = -mlavg['hke_lowpass']/mlavg.mld
    mlavg['eps'] = -mlavg['eps']/mlavg.mld

    mlavg.to_netcdf(str(output))


# %% MAIN
mlavg_wrapper(snakemake.input, snakemake.output)

# %% testing
#file = './data/filtered/filt_7700b_3h_2Tf.nc'
#data = xr.open_dataset(file)
#
# # compute some variables
# data['hke'] = 0.5 * (data.u**2 + data.v**2)
# data['hke_lowpass'] = 0.5 * (data.u_lowpass**2 + data.v_lowpass**2)
# data['hke_resid'] = 0.5 * (data.u_resid**2 + data.v_resid**2)
# data['hke_ni'] = 0.5 * (data.uni**2 + data.vni**2)
#
# mlavg = integrate_columns(data)
# mlavg['mld'] = data.mld
#
# f,ax=plt.subplots(2,1,sharex=True)
# mlavg.hke_ni.plot(ax=ax[0])
# mlavg.hke_resid.plot(ax=ax[0])
# mlavg.mld.plot(ax=ax[1])