from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# My Stuff
from src.tools import alphabet, str2date

# set up figure params
sns.set(style='ticks', context='paper', palette='colorblind')
mpl.rc('figure', dpi=100, figsize=[6, 6])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
w0 = 2
t = np.linspace(0,1)
u = np.sin(2*np.pi*t+0.2*np.pi)
v = np.sin(2*np.pi*t) 

plt.figure()
plt.plot(t,u,label='u')
plt.plot(t,v,label='v')
plt.legend()
plt.show()


phase = np.arctan2(v,u)
plt.figure()
plt.plot(t,phase)
plt.plot(t[:-1],np.unwrap( np.diff(phase)) )
