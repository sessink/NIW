# Scientific Computing
# Standard Library
import glob

import numpy as np
import pandas as pd
import xarray as xr
import tables

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# My Stuff
from src.tools import alphabet, str2date, load_matfile

# set up figure params
sns.set(style='ticks', context='paper', palette='colorblind')
mpl.rc('figure', dpi=100, figsize=[11, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
def convert_tmsdata(chi_dir):
    dat = load_matfile(chi_dir)


    f_cps = 0.5*(dat['flabeg']+dat['flaend'])
    time = pd.to_datetime(dat['uxt'],unit='s')

    dat = xr.Dataset({
        'sla1' : (['time','f_cps'],dat['Sla1']),
        'sla2' : (['time','f_cps'],dat['Sla1']),
        'logavgoff': dat['logavgoff'],
        'logavgsf': dat['logavgsf'],
        'nobs':dat['nobs']
    },
    coords={'time':time,'f_cps':f_cps}
    )

    return dat

def convert_ctddata(ctd_dir):
    dat = load_matfile(ctd_dir)
    dat['CTD']['jday_gmt'].flatten()[0].shape
    dat['CTD']['T'].flatten()[0].shape

    dat['CTD']['P'].flatten()[0]
    dat = xr.Dataset({
    },
    coords={'time':time,'f_cps':f_cps}
    )

    return dat

# %% read blocks
# FIXME: Blocks??
chi_dir = 'data/chi/ema-7786b-0030-tms.mat'
dat = convert_tmsdata(chi_dir)

floatid = hpid
kzmin = 20; kzmax = 400; threshold = 4;
Chi = chiprofile_fun(floatid,hpid,chi_data_dir,kzmin,kzmax,plotting,threshold);

# %% 1) convert realtime-transmitted scaled spectrum (sla)
# to digitized voltage Spectrum
jblock =0

dat['slad1'] = (dat.sla1 - dat.logavgoff)/dat.logavgsf
dat['slad2'] = (dat.sla2 - dat.logavgoff)/dat.logavgsf

# %% 2) convert to raw spectrum of temperature
beta = 25; Vref = 4; Inet = 0.8;
scale2 = (beta * Vref * 2**(-23) / Inet)**2
dat['rawTsp1'] = 10**(dat.slad1[jblock,:]/10)*scale2
dat['rawTsp2'] = 10**(dat.slad2[jblock,:]/10)*scale2

# %% 3) get T,N,P,W, and dT/dz from ctd
# TODO: which CTD data to use?
ctd_dir = 'data/NIWmatdata/7786b_grid.mat'
data = load_matfile(ctd_dir)

data


data = xr.open_dataset(ctd_dir)
data['dTdz'] = data.T.differentiate('z')
data
data['w'] = data.z.differentiate('time')

# %% 4) compute transfer functions
# %% 5) remove noise Spectrum
# %% 6) convert temperature frequency spectrum to wavenumber Spectrum
# %% 7) compute chi, kT, and eps1
