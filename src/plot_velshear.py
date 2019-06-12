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
	file =  str(input)
	id = str(output[0]).split('_')[1].split('.')[0]
	data = xr.open_dataset(file)

	var = ['u','v']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
	    h = data[var[i]].plot(ax=ax,rasterized=True,
								cbar_kwargs={'pad':0.01},
								vmin=-1,vmax=1,
								cmap='RdBu_r')
	    ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
	    ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(id,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[0]))

	var = ['dudz','dvdz']
	f,ax=plt.subplots(len(var),1,sharex=True)
	for i,ax in enumerate(ax):
	    h = data[var[i]].plot(ax=ax,rasterized=True,
	                          vmin=-1e-2,vmax=1e-2,
	                          cbar_kwargs={'pad':0.01,'format':'%.3f'},
	                          cmap='RdBu_r')
	    ax.set_xticks(pd.date_range(data.time.min().values,data.time.max().values,freq='M',))
	    ax.set(ylim=[-500,0],title=var[i],xlabel=None)
	plt.suptitle(id,x=0.15,y=0.91,weight='bold')
	plt.savefig(str(output[1]))

plot_shears(snakemake.input,snakemake.output)
