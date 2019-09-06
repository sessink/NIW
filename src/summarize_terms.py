# Scientific Computing
# Standard Library
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
from tools import alphabet, str2date

# set up figure params
sns.set(style='ticks', context='paper', palette='colorblind')
mpl.rc('figure', dpi=100, figsize=[7.5, 10])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
# dir = glob.glob('data/ml/ml_ww_*_3h_2Tf.nc')
def compute_stats(array):
    import numpy as np
    from scipy.stats import skew, kurtosis
    temp = array.pipe(np.abs).pipe(np.log10).values.flatten()
    temp = temp[np.isfinite(temp)]
    mean = np.nanmean(temp)
    sk = skew(temp)
    kur = kurtosis(temp)
    return mean, sk, kur


def print_stats(ax, dataarray):
    mean, sk, kur = compute_stats(dataarray)
    ax.annotate(f'mean = {mean:2.2f},' + '\n' + f'sk = {sk:2.2f},' + '\n' +
                f'kur = {kur:2.2f}', (0.8, 0.8),
                xycoords='axes fraction')


def plot_histograms(outfile, files, flatten=False, **kwargs):
    files = files.dropna(dim='floatid', how='all')
    f, ax = plt.subplots(3, 1, sharex=True)
    if flatten:
        ax[0].hist(
            files.taudotu.pipe(np.abs).pipe(np.log10).values.flatten(),
            **kwargs)
    else:
        ax[0].hist(files.taudotu.pipe(np.abs).pipe(np.log10), **kwargs)
        ax[0].legend(files.floatid.values,
                     loc='upper left',
                     bbox_to_anchor=(1, 1))
    ax[0].set_xlabel(r'$\overline{\mathbf{\tau}\cdot\mathbf{u}}$')
    ax[0].set_ylabel('PDF')
    print_stats(ax[0], files.taudotu)
    files.taudotu.pipe(np.abs).pipe(np.log10)

    if flatten:
        ax[1].hist(
            files.dHKEdt_resid.pipe(np.log10).values.flatten(), **kwargs)
    else:
        ax[1].hist(files.dHKEdt_resid.pipe(np.log10), **kwargs)
    ax[1].set_xlabel(r'$\overline{d\mathit{HKE}}$')
    ax[1].set_ylabel('PDF')
    print_stats(ax[1], files.dHKEdt_resid)

    if flatten:
        ax[2].hist(
            files.eps.pipe(np.abs).pipe(np.log10).values.flatten(), **kwargs)
    else:
        ax[2].hist(files.eps.pipe(np.abs).pipe(np.log10), **kwargs)
    ax[2].set_xlabel(r'$\overline{\epsilon}$')
    ax[2].set_ylabel('PDF')
    print_stats(ax[2], files.eps)

    alphabet(ax)
    plt.tight_layout()
    plt.savefig(str(outfile))


def plot_scatter(outfile, data, title):
    hke_thres = 0.05

    data['dHKEdt_abs'] = np.sqrt(data.dHKEdt_resid**2)
    data['taudotu_abs'] = np.sqrt(data.taudotu**2)
    f, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].scatter(x='eps',
                  y='dHKEdt_abs',
                  data=data,
                  label='no storm',
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[0].plot(data.eps, data.eps, lw=1, color='k')
    ax[0].scatter(x='eps',
                  y='dHKEdt_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r',
                  label='storm',
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[0].set(xlim=(1e-9, 1e-4), ylim=(1e-9, 1e-4), yscale='log', xscale='log')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel(r'$\overline{\epsilon}$')
    ax[0].set_ylabel(r'$\overline{d\mathit{HKE}}$')
    ax[0].legend()

    ax[1].scatter(x='eps',
                  y='taudotu_abs',
                  data=data,
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[1].plot(data.eps, data.eps, lw=1, color='k')
    ax[1].scatter(x='eps',
                  y='taudotu_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r',
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[1].set(xlim=(1e-9, 1e-4), ylim=(1e-9, 1e-4), yscale='log', xscale='log')
    ax[1].set_aspect('equal')
    ax[1].set_xlabel(r'$\overline{\epsilon}$')
    ax[1].set_ylabel(r'$\overline{\mathbf{\tau}\cdot\mathbf{u}}$')

    ax[2].scatter(x='dHKEdt_abs',
                  y='taudotu_abs',
                  data=data,
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[2].plot(data.dHKEdt_abs, data.dHKEdt_abs, lw=1, color='k')
    ax[2].scatter(x='dHKEdt_abs',
                  y='taudotu_abs',
                  data=data.where(data.hke > hke_thres),
                  color='r',
                  alpha=0.6,
                  rasterized=True,
                  edgecolors=None)
    ax[2].set(xlim=(1e-9, 1e-4), ylim=(1e-9, 1e-4), yscale='log', xscale='log')
    ax[2].set_aspect('equal')
    ax[2].set_xlabel(r'$\overline{d\mathit{HKE}}$')
    ax[2].set_ylabel(r'$\overline{\mathbf{\tau}\cdot\mathbf{u}}$')

    plt.suptitle(f'{title}')
    alphabet(ax)
    plt.savefig(str(outfile))


# %%
def read_files(infiles):
    # from glob import glob
    # dir = glob('data/ml/ww_????a_1h*')
    files = []
    floatids = []
    for file in snakemake.input:
        # for file in dir:
        files.append(xr.open_dataset(file))
        floatids.append(file.split('_')[1])

    files = xr.concat(files, dim='floatid')
    files['floatid'] = floatids
    files['exp'] = xr.where(files.time < str2date('2017,6,1'), 'a', 'b')
    return files


# %%
files = read_files(snakemake.input)
plot_histograms(snakemake.output[0],
                files,
                flatten=True,
                bins=50,
                range=(-10, -4),
                density=True)
plot_histograms(snakemake.output[1],
                files.where(files.exp == 'a'),
                bins=15,
                range=(-10, -4),
                density=True)
plot_histograms(snakemake.output[2],
                files.where(files.exp == 'b'),
                bins=15,
                range=(-10, -4),
                density=True)

plot_scatter(snakemake.output[3], files, 'Year A+B')
plot_scatter(snakemake.output[4], files.where(files.exp == 'a'), 'Year A')
plot_scatter(snakemake.output[5], files.where(files.exp == 'b'), 'Year B')
