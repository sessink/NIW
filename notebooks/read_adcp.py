#!/usr/bin/env python
# coding: utf-8

# In[39]:
import numpy as np
import xarray as xr
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

# In[21]:
def load_matfile(file):
    '''Read Matlab structure files and convert to numpy arrays'''
    import scipy.io as sio
    import numpy as np
    return sio.loadmat(file, struct_as_record=True, squeeze_me=True)

def datenum2datetime(datenum):
    '''Convert Matlab datenum to Python Datetime'''
    from datetime import datetime, timedelta
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) -        timedelta(days=366)

dn2dt_vec = np.vectorize(lambda x: datenum2datetime(x))


# In[4]: 
path_to_matfile = '../data/adcp/KS-16-12.mat'
dat = load_matfile(path_to_matfile)
# dat is like a matlab struct with dat['u'], dat['v'], dat['z'] and dat['Jday_gmt']


# In[40]: This is just reformating into a dataset with dimensions
ds = xr.Dataset({ 
    'u': (['z', 'time'], dat['u']),
    'v': (['z', 'time'], dat['v'])},
    coords={'z': dat['z'],
            'time': dat['Jday_gmt']}
)

# converting matlab datenum to python datetime
ds = ds.assign_coords(time=(dn2dt_vec(ds.time)))


# In[41]: Sorting out bad values and making z negative
ds['u'] = ds.u.where(ds.u>-30)
ds['v'] = ds.v.where(ds.u>-30)
if ds['z'].diff('z').mean()>0:
    ds['z'] = -ds['z']


# In[43]: Quick plot
f, ax = plt.subplots(2,1, sharex=True)
ds.u.plot(robust=True, ax=ax[0])
ds.v.plot(robust=True, ax=ax[1]);


