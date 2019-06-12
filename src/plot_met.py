import os,glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as sio
import gsw
from datetime import datetime,timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,5])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)


fname = 'pres.sfc.2015.nc'
data = xr.open_dataset('./data/ncepncar/'+fname)

bdry = [30,45,140,160]

data.isel(time=110).sel(lon=slice(120,140),lat=slice(45,35)).pres.plot();
