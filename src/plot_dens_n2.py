import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cmocean import cm
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,7])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

from scipy.interpolate import interp1d
from scipy.signal import medfilt
from tools import compute_mld

def plot_velocities(input,output):
	file =  str(input)
	id = str(output[0]).split('_')[1].split('.')[0]
	data = xr.open_dataset(file)
	data = compute_mld(data)
	data['sigma'] = data.sigma-1000

	var = ['sigma','n2']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
		if i==0:
			data.mld.plot(ax=ax,color='k')
			h = data[var[i]].plot(ax=ax,rasterized=True,cbar_kwargs={'pad':0.01},vmin=22,vmax=27,cmap=cm.dense)
		elif i==1:
			data.mld.plot(ax=ax,color='k')
			h = data[var[i]].pipe(np.log10).plot(ax=ax,rasterized=True,cbar_kwargs={'pad':0.01},vmin=-5,vmax=1,cmap=cm.amp)
		ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
		ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(id,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[0]))

plot_velocities(snakemake.input,snakemake.output)



# input = './data/xarray/xr_7785b_grid.nc'
# file =  str(input)
# # id = str(output[0]).split('_')[1].split('.')[0]
# data = xr.open_dataset(file)
# data = compute_mld(data)
#
# test plot
# var = ['sigma','n2']
# f,ax=plt.subplots(len(var),1,sharex=True)
# for i,ax in enumerate(ax):
# 	if i==0:
# 		data.mld.plot(ax=ax,color='k')
# 		h = data[var[i]].plot(ax=ax,rasterized=True,
# 								cbar_kwargs={'pad':0.01},
# 								vmin=22,vmax=27,
# 								cmap=cm.dense)
# 	elif i==1:
# 		data.mld.plot(ax=ax,color='k')
# 		h = data[var[i]].pipe(np.log10).plot(ax=ax,rasterized=True,
# 								cbar_kwargs={'pad':0.01},
# 								vmin=-5,vmax=1,
# 								cmap=cm.amp)
# 	ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
# 	ax.set(ylim=[-500,0],title=var[i],xlabel=None)
# plt.show()
