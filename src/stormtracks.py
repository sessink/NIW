import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,7])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

path = './data/stormtracks/IBTrACS.ALL.v04r00.nc'

data = xr.open_dataset(path)


plt.plot( data.isel(date_time=0).lon,data.isel(date_time=0).lat,'.',ms=0.1)
