import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks',context='paper')
plt.style.use('sebstyle')
# %%

res = xr.open_dataset('data/resampled/resample_7700b_3h.nc')

res.u.dropna(dim='z',how='all').plot(ylim=(-500,0))

filt = xr.open_dataset('data/filtered/filt_7700b_3h_4Tf.nc')

filt.u_lowpass.plot(ylim=(-500,0))
