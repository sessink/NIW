# %% IMPORTS
import numpy as np
import xarray as xr


# %% MAIN
def open_xr(infile):
    return xr.open_dataset(str(infile))


# %% MAIN
files = [open_xr(file) for file in snakemake.input]
floatnames = [file.split('_')[1] for file in snakemake.input]
data = xr.concat(files, dim='float')
data = data.assign_coords(float=floatnames)
data = data.assign_coords(floatid=np.arange(len(files)))
data.to_netcdf(str(snakemake.output))

# %% TESTING
# infile = ['data/ml/ml_7700b_9h_6Tf.nc',
#         'data/ml/ml_7701b_9h_6Tf.nc', 'data/ml/ml_7779a_9h_6Tf.nc']
#
# floatnames = [file.split('_')[1] for file in infile]
#
# files = [open_xr(file) for file in infile]
# data = xr.concat(files,dim='float')
#
#
#
# data.sel(float='7700b').sigma
