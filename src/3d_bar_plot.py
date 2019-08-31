# Scientific Computing
import numpy as np
# import scipy.io as sio
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# My Stuff
from src.tools import datenum2datetime, load_matfile

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 7])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
data = xr.open_dataset('data/xarray/qc_ww_7787a.nc')
# %%

utau = np.abs( data.utau.interp(mettime=data.time) )

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(data.lon, data.lat, np.zeros_like(data.lon), 0.05,0.05,np.abs( data.mld ),zsort='average',shade=False)
ax.view_init(elev=50., azim=5.)
ax.set_zlim(0,100)
