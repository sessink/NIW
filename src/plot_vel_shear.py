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
	data = xr.open_dataset(str(file))

	# fix some stuff but this will be done in convert_mat_to_xr
	data = data.assign_coords(z=-data.z)
	_, index = np.unique(data.time, return_index=True)
	data = data.isel(time=index)

	var = ['u1','u1','v1','v1']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
	    h = data[var[i]].plot(ax=ax,rasterized=True,cbar_kwargs={'pad':0.01},vmin=-1,vmax=1,cmap='RdBu_r')
	    ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
	    ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(data.floatid,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[0]))

	var = ['du1dz','du2dz','dv1dz','dv2dz']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
	    h = data[var[i]].plot(ax=ax,rasterized=True,
	                          vmin=-1e-2,vmax=1e-2,
	                          cbar_kwargs={'pad':0.01},
	                          cmap='RdBu_r')
	    ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
	    ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(data.floatid,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[1]))

plot_shears(snakemake.input,snakemake.output)