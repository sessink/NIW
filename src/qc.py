# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

from tools import str2date, avg_funs

# set up figure params
sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[8.5, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

# %%
def qc_turbulence(data):
    '''
    clean chi and eps with RC's scripts
    '''
    # infile = 'data/xarray/xr_7784b.nc'
    # data = xr.open_dataset(str(infile))

    dtdzmin = 1.5e-3
    chimax = 5e-5
    kTmax = 1e-1
    zmin = 0 # disabled
    # for ratios
    lb = 0.5
    ub = 2

    floats = np.array([
        '7779a', '7781a', '7783a', '7786a', '7787a', '7788a',
        '7700b', '7701b','7780b', '7784b', '7785b', '7786b'
    ])
    fi = np.where(floats == data.floatid)[0][0]
    good_chi1, good_chi2 = np.load('../data/xarray/good_chi.npy')

    # 1) thresholds for chi
    data['dtdz1'] = np.sqrt(0.5 * data.chi1 / data.kT1)
    data['dtdz2'] = np.sqrt(0.5 * data.chi2 / data.kT2)

    bad = (data.dtdz1 <= dtdzmin) | (data.chi1 >= chimax) | (data.kT1 >= kTmax) #| (data.z > zmin)
    data['chi1'] = data['chi1'].where(~bad)
    data['kT1'] = data['kT1'].where(~bad)
    data['eps1'] = data['eps1'].where(~bad)

    bad = (data.dtdz2 <= dtdzmin) | (data.chi2 >= chimax) | (data.kT2 >= kTmax) #| (data.z > zmin)
    data['chi2'] = data['chi2'].where(~bad)
    data['kT2'] = data['kT2'].where(~bad)
    data['eps2'] = data['eps2'].where(~bad)

    # 2) periods of functioning chi sensor
    tmin, tmax = str2date(good_chi1[fi, 0]), str2date(good_chi1[fi, 1])
    bad = (data.time < tmin) | (data.time > tmax)
    data['chi1'] = data['chi1'].where(~bad)
    data['kT1'] = data['kT1'].where(~bad)
    data['eps1'] = data['eps1'].where(~bad)

    tmin, tmax = str2date(good_chi2[fi, 0]), str2date(good_chi2[fi, 1])
    bad = (data.time < tmin) | (data.time > tmax)
    data['chi2'] = data['chi2'].where(~bad)
    data['kT2'] = data['kT2'].where(~bad)
    data['eps2'] = data['eps2'].where(~bad)

    # 3) compare two sensors
    def combine_fun(array1, array2, lb=lb, ub=ub):
        ratio = array1 / array2
        bad = (ratio <= lb) | (ratio >= ub)

        chi1fin = np.isfinite(array1)
        chi2fin = np.isfinite(array2)

        a1 = np.minimum(array1.where(bad & chi1fin),
                        array2.where(bad & chi1fin))
        a2 = np.minimum(array1.where(bad & chi2fin),
                        array2.where(bad & chi2fin))
        a3 = avg_funs(array1.where(~bad), array2.where(~bad))

        concat = xr.concat([a1, a2, a3], dim='temp')
        return concat.mean(dim='temp')

    data['kT'] = combine_fun(data.kT1, data.kT2)
    data['chi'] = combine_fun(data.chi1, data.chi2)
    data['eps'] = combine_fun(data.eps1, data.eps2)

    data = data.drop(
        ['eps1', 'eps2', 'chi1', 'chi2', 'kT1', 'kT2', 'dtdz1', 'dtdz2'])
    return data

def qc_velocity(data):
    Wmin = 0.05
    RotPmax = 20
    verrmax = 0.02

    # where cond:  what to keep!
    uv_mask = (np.abs(data.W) > Wmin) & (data.RotP < RotPmax)
    u1_mask = uv_mask & (data.verr1 < verrmax)
    u2_mask = uv_mask & (data.verr2 < verrmax)

    data['u1'] = data.u1.where(u1_mask)
    data['u2'] = data.u2.where(u2_mask)
    data['v1'] = data.u1.where(u1_mask)
    data['v2'] = data.v2.where(u2_mask)

    data['u'] = avg_funs(data['u1'], data['u2'])
    data['v'] = avg_funs(data['v1'], data['v2'])
    data['dudz'] = avg_funs(data['du1dz'], data['du2dz'])
    data['dvdz'] = avg_funs(data['dv1dz'], data['dv2dz'])

    data = data.drop([
        'W', 'RotP', 'verr1', 'verr2', 'u1', 'u2', 'v1', 'v2', 'du1dz',
        'du2dz', 'dv1dz', 'dv2dz'
    ])
    return data


def plot_velocities(data, dataorig, figurepath):
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
    dataorig.u1.plot(ax=ax[1, 0],
                     ylim=(-500, 0),
                     vmin=-1,
                     vmax=1,
                     cmap='RdBu_r',
                     rasterized=True)
    ax[1, 0].set_title('raw u')
    dataorig.v1.plot(ax=ax[1, 1],
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


def plot_epsilon(data, dataorig, figurepath):
    f, ax = plt.subplots(2, 1, sharex=True)
    data.eps.pipe(np.log10).plot(ax=ax[0],
                                 ylim=(-500, 0),
                                 vmin=-10,
                                 vmax=-2,
                                 cmap='viridis',
                                 rasterized=True)
    ax[0].set_title('QC eps')
    ax[0].set_xlabel(None)

    dataorig.eps1.pipe(np.log10).plot(ax=ax[1],
                                      ylim=(-500, 0),
                                      vmin=-10,
                                      vmax=-2,
                                      cmap='viridis',
                                      rasterized=True)
    ax[1].set_title('raw eps')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(str(figurepath))
    plt.close()


# %%
def qc(infile, outfile, figurepath1, figurepath2):
    # from scipy.signal import medfilt2d

    # infile = 'data/xarray/xr_7784b.nc'

    data = xr.open_dataset(str(infile))
    dataorig = data.copy()

    data = qc_velocity(data)
    data = qc_turbulence(data)

    data.to_netcdf(str(outfile))
    plot_velocities(data, dataorig, figurepath1)
    plot_epsilon(data, dataorig, figurepath2)


# %% MAIN
# qc(snakemake.input, snakemake.output[0], snakemake.output[1],
   # snakemake.output[2])

# %% testing
# infile = 'data/xarray/xr_7784b.nc'
# #
# # Wmin = 0.05
# # RotPmax = 20
# # verrmax = 0.02
# # dtdzmax = 1.5e-5
# # chimin = 5e-5
# # kTmin = 1e-1
# # lb = 0.5  # for ratios
# # ub = 2
# #
# data = xr.open_dataset(str(infile))
#
# test = qc_turbulence(data)
#
# dataorig = data.copy()
# dataorig['u'] = 0.5 * (dataorig['u1'] + dataorig['u2'])
# dataorig['v'] = 0.5 * (dataorig['v1'] + dataorig['v2'])
#
# uv_mask = (np.abs(data.W) > Wmin) & (data.RotP < RotPmax)
# u1_mask = uv_mask & (data.verr1 < verrmax)
# u2_mask = uv_mask & (data.verr2 < verrmax)
#
# data['dtdz1'] = np.sqrt(0.5 * data.chi1 / data.kT1)
# bad = (data.dtdz1 <= dtdzmax) | (data.chi1 >= chimin) | (data.kT1 >= kTmin)
# data[['chi1', 'kT1', 'eps1']] = data[['chi1', 'kT1', 'eps1']].where(bad)
#
# data.eps1.notnull().sum()
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
