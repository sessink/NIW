import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumtrapz

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[8.5, 11])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
