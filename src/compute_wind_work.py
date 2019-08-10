# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

from tools import compute_mld

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=120, figsize=[10, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def first_finite(arr, axis):
    mask = arr.notnull()
    return xr.where(mask.any(axis=axis), mask.argmax(axis=axis), np.nan)

def compute_wind_work(infile, outfile):

    # infile = 'data/xarray/qc_7779a.nc'
    dat = xr.open_dataset(str(infile[0]))
    met = xr.open_dataset(str(infile[1]))
    id = str(infile).split('_')[1].split('.')[0]
    metfloat = met.sel(floatid=id)

    # extract uppermost velocity measuremnt
    indu = first_finite(dat.u, 0).fillna(0).astype(int)
    indv = first_finite(dat.v, 0).fillna(0).astype(int)

    dat['u_surf'] = dat.u[indu]
    dat['v_surf'] = dat.v[indv]

    # interpolate currents onto metdata timestamp
    metfloat['u'] = ('time', dat.u_surf.interp(time=metfloat.time))
    metfloat['v'] = ('time', dat.v_surf.interp(time=metfloat.time))

    # compute tau dot u
    metfloat['tauxdotu'] = metfloat.u * metfloat.tx
    metfloat['tauydotv'] = metfloat.v * metfloat.ty
    metfloat['utau'] = metfloat.tauxdotu + metfloat.tauydotv

    dat['mettime']=metfloat.time.values
    dat['tauxdotu'] = ('mettime', metfloat['tauxdotu'])
    dat['tauydotv'] = ('mettime', metfloat['tauydotv'])
    dat['utau'] = ('mettime', metfloat['utau'])
    dat['tx'] = ('mettime', metfloat['tx'])
    dat['ty'] = ('mettime', metfloat['ty'])
    dat['lw'] = ('mettime', metfloat['lw'])
    dat['sw'] = ('mettime', metfloat['sw'])
    dat['Qnet'] = ('mettime', metfloat['Qnet'])
    dat = dat.assign_coords(mettime=dat.mettime)

    dat = compute_mld(dat)

    dat.to_netcdf(str(outfile))

# %%
compute_wind_work(snakemake.input,snakemake.output)
# metfloat.utau.plot(label=r'u$\tau_x$')
# metfloat.tauxdotu.plot(label=r'v$\tau_y$')
# metfloat.tauydotv.plot(label=r'$\mathbf{u}\cdot\mathbf{\tau}$')
# plt.ylabel('wind work $W~m^{-2}$')
# plt.legend()
