import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import seaborn as sns

sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,7])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

def plot_shears(input,output):
	file =  str(input) #os.path.join(project_path,i)
	# c = count()
	id = str(output[0]).split('_')[1].split('.')[0]
	data = xr.open_dataset(str(file))

	# fix some stuff but this will be done in convert_mat_to_xr
	data = data.assign_coords(z=-data.z)
	_, index = np.unique(data.time, return_index=True)
	data = data.isel(time=index)

	var = ['eps','chi']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
	    h = data[var[i]].pipe(np.log10).plot(ax=ax,
			rasterized=True,cbar_kwargs={'pad':0.01},vmax=2,vmin=-8,cmap='viridis')
	    ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
	    ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(id,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[0]))

plot_shears(snakemake.input,snakemake.output)

##

# input = './data/xarray/xr_7701b_grid.nc'
#
# file =  str(input) #os.path.join(project_path,i)
# # c = count()
# data = xr.open_dataset(str(file))
#
# # fix some stuff but this will be done in convert_mat_to_xr
# data = data.assign_coords(z=-data.z)
# _, index = np.unique(data.time, return_index=True)
# data = data.isel(time=index)
#
# var = ['eps','chi']
# f,ax=plt.subplots(len(var),1,sharex=True)
# for i,ax in enumerate(ax):
#     h = data[var[i]].pipe(np.log10).plot(ax=ax,rasterized=True,cbar_kwargs={'pad':0.01},cmap='viridis')
#     ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
#     ax.set(ylim=[-500,0],title=var[i],xlabel=None)
# plt.suptitle(data.floatid,x=0.15,y=0.91,weight='bold')
# plt.show()
