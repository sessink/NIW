import numpy as np
import pandas as pd
import xarray as xr

def filter_timeseries(input,output):
    file =  str(input)
    data = xr.open_dataset(file)

    vars = ['u','v','dudz','dvdz']
    for var in vars:
        data[var] = data[var].rolling(time=10).mean()

    data.to_netcdf(str(output))

filter_timeseries(snakemake.input,snakemake.output)
