# Scientific Computing
# Standard Library
import glob

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
mpl.rc('figure', dpi=100, figsize=[11, 5])
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


def plot_histograms(outfile, files, flatten=False, **kwargs):
    files = files.dropna(dim='floatid', how='all')
    f, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
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
    mean, sk, kur = compute_stats(files.taudotu)
    ax[0].annotate(f'mean = {mean:2.2f}, sk = {sk:2.2f}, kur = {kur:2.2f}',
                   (0.8, 0.9),
                   xycoords='axes fraction')

    files.taudotu.pipe(np.abs).pipe(np.log10)
    kur
    if flatten:
        ax[1].hist(
            files.dHKEdt_resid.pipe(np.log10).values.flatten(), **kwargs)
    else:
        ax[1].hist(files.dHKEdt_resid.pipe(np.log10), **kwargs)
    ax[1].set_xlabel(r'$\overline{d\mathit{HKE}}$')
    ax[1].set_ylabel('PDF')
    mean, sk, kur = compute_stats(files.dHKEdt_resid)
    ax[1].annotate(f'mean = {mean:2.2f}, sk = {sk:2.2f}, kur = {kur:2.2f}',
                   (0.8, 0.9),
                   xycoords='axes fraction')

    if flatten:
        ax[2].hist(
            files.eps.pipe(np.abs).pipe(np.log10).values.flatten(), **kwargs)
    else:
        ax[2].hist(files.eps.pipe(np.abs).pipe(np.log10), **kwargs)
    ax[2].set_xlabel(r'$\overline{\epsilon}$')
    ax[2].set_ylabel('PDF')
    mean, sk, kur = compute_stats(files.eps)
    ax[2].annotate(f'mean = {mean:2.2f}, sk = {sk:2.2f}, kur = {kur:2.2f}', (0.8, 0.9),
                   xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig(str(outfile))


# %%
def read_files(infiles):
    files = []
    floatids = []
    for file in snakemake.input:
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
