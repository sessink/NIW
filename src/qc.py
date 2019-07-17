# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[8.5, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def qc(infile, outfile, figurepath):
    # from scipy.signal import medfilt2d

    Wmin = 0.05
    RotPmax = 20
    verrmax = 0.03

    data = xr.open_dataset(str(infile))
    dataorig = data.copy()
    dataorig['u'] = 0.5 * (dataorig['u1'] + dataorig['u2'])
    dataorig['v'] = 0.5 * (dataorig['v1'] + dataorig['v2'])

    uv_mask = (np.abs(data.W) > Wmin) & (data.RotP < RotPmax)
    u1_mask = uv_mask & (data.verr1 < verrmax)
    u2_mask = uv_mask & (data.verr2 < verrmax)

    data['u1'] = data.u1.where(u1_mask)
    data['v1'] = data.u1.where(u1_mask)
    data['u2'] = data.u2.where(u2_mask)
    data['v2'] = data.v2.where(u2_mask)

    data['u'] = 0.5 * (data['u1'] + data['u2'])
    data['v'] = 0.5 * (data['v1'] + data['v2'])

    # data['u'] = (('z','time'),medfilt2d(data.u))
    # data['v'] =(('z','time'),medfilt2d(data.v))

    data['chi'] = 0.5 * (data['chi1'] + data['chi2'])
    data['eps'] = 0.5 * (data['eps1'] + data['eps2'])
    data['dudz'] = 0.5 * (data['du1dz'] + data['du2dz'])
    data['dvdz'] = 0.5 * (data['dv1dz'] + data['dv2dz'])


    data = data.drop(['W','RotP','verr1','verr2',
                      'u1','u2','v1','v2',
                      'eps1','eps2','chi1','chi2'])
    data.to_netcdf(str(outfile))

    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    data.u.plot(ax=ax[0, 0],
                ylim=(-500, 0),
                vmin=-1,
                vmax=1,
                cmap='RdBu_r',
                rasterized=True)
    ax[0, 0].set_title('QC u')
    ax[0, 0].set_xlabel(None)
    data.v.plot(ax=ax[0, 1],
                ylim=(-500, 0),
                vmin=-1,
                vmax=1,
                cmap='RdBu_r',
                rasterized=True)
    ax[0, 1].set_title('QC v')
    ax[0, 1].set_xlabel(None)
    ax[0, 1].set_ylabel(None)
    dataorig.u.plot(ax=ax[1, 0],
                    ylim=(-500, 0),
                    vmin=-1,
                    vmax=1,
                    cmap='RdBu_r',
                    rasterized=True)
    ax[1, 0].set_title('raw u')
    dataorig.v.plot(ax=ax[1, 1],
                    ylim=(-500, 0),
                    vmin=-1,
                    vmax=1,
                    cmap='RdBu_r',
                    rasterized=True)
    ax[1, 1].set_title('raw v')
    ax[1, 1].set_ylabel(None)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(str(figurepath))
    plt.close()


# %% MAIN
qc(snakemake.input, snakemake.output[0], snakemake.output[1])


# %% testing
# infile = 'data/xarray/xr_7784b.nc'
#
# Wmin = 0.05
# RotPmax = 20
# verrmax = 0.02
#
# data = xr.open_dataset(str(infile))
# dataorig = data.copy()
# dataorig['u'] = 0.5 * (dataorig['u1'] + dataorig['u2'])
# dataorig['v'] = 0.5 * (dataorig['v1'] + dataorig['v2'])
#
# uv_mask = (np.abs(data.W) > Wmin) & (data.RotP < RotPmax)
# u1_mask = uv_mask & (data.verr1 < verrmax)
# u2_mask = uv_mask & (data.verr2 < verrmax)
#
# data['u1'] = data.u1.where(u1_mask)
# data['v1'] = data.u1.where(u1_mask)
# data['u2'] = data.u2.where(u2_mask)
# data['v2'] = data.v2.where(u2_mask)
#
# data['u'] = 0.5 * (data['u1'] + data['u2'])
# data['v'] = 0.5 * (data['v1'] + data['v2'])


# # %%
# f, ax = plt.subplots(2, 2,sharex=True)
# ax[0,0].scatter(dataorig.u1,dataorig.W,rasterized=True)
# ax[0,0].scatter(data.u1,data.W,color='r',rasterized=True)
# ax[0,0].set_xlabel('u1')
# ax[0,0].set_ylabel('W [m/s]')
#
# ax[0,1].scatter(dataorig.u1,dataorig.RotP,rasterized=True)
# ax[0,1].scatter(data.u1,data.RotP,color='r',rasterized=True)
# ax[0,1].set_xlabel('u1')
# ax[0,1].set_ylabel('RotP [s]')
#
# ax[1,0].scatter(dataorig.u1,dataorig.verr1,rasterized=True)
# ax[1,0].scatter(data.u1,data.verr1,color='r',rasterized=True)
# ax[1,0].set_xlabel('u1')
# ax[1,0].set_ylabel('verr1 [m/s]')
#
# ax[1,1].scatter(dataorig.u2,dataorig.verr2,rasterized=True)
# ax[1,1].scatter(data.u2,data.verr2,color='r',rasterized=True)
# ax[1,1].set_xlabel('u2')
# ax[1,1].set_ylabel('verr2 [m/s]')
#
# plt.savefig('figures/qc/7784b_qv_variables.pdf')
# plt.show()
