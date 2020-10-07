###
sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
f, ax = plt.subplots(3,1, sharex=True, figsize=(14,8))

raw.hkeNI.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-4, vmax=-1.5, cmap=purp, ax=ax[0], rasterized=True,
                             cbar_kwargs={'pad':0.01,'label':'log HKE'})
raw.mld.plot(color='k', lw=2, ax=ax[0])
ax[0].set_ylim(-500,0)
ax[0].set_xlabel(None)
ax[0].set_ylabel('Depth [m]')


raw.eta_ape.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-4, vmax=-1.5, cmap=purp, ax=ax[1], rasterized=True,
                           cbar_kwargs={'pad':0.01,'label':'log APE'})
raw.mld.plot(color='k', lw=2,ax=ax[1])
ax[1].set_ylim(-500,0)
ax[1].set_xlabel(None)
ax[1].set_ylabel('Depth [m]')

raw.hke_ape.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-3, vmax=3, cmap='RdBu_r', ax=ax[2], rasterized=True,
                               cbar_kwargs={'pad':0.01,'label':'log HKE/APE'})
raw.mld.plot(color='k', lw=2, ax=ax[2])
ax[2].set_ylim(-500,0)
ax[2].set_xlabel(None)
ax[2].set_ylabel('Depth [m]')

alphabet(ax)

###
raw.hkeNI.fillna(0).resample(time='2h').mean().rolling(time=18, center=True).mean().differentiate('time',datetime_unit='s').plot(robust=True)
plt.ylim(-500,)

sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
f, ax = plt.subplots(3,1, sharex=True, figsize=(14,8))

raw.hkeNI.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-4, vmax=-1.5, cmap=purp, ax=ax[0], rasterized=True,
                             cbar_kwargs={'pad':0.01,'label':'log HKE'})
raw.mld.plot(color='k', lw=2, ax=ax[0])
ax[0].set_ylim(-500,0)
ax[0].set_xlabel(None)
ax[0].set_ylabel('Depth [m]')


raw.eta_ape.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-4, vmax=-1.5, cmap=purp, ax=ax[1], rasterized=True,
                           cbar_kwargs={'pad':0.01,'label':'log APE'})
raw.mld.plot(color='k', lw=2,ax=ax[1])
ax[1].set_ylim(-500,0)
ax[1].set_xlabel(None)
ax[1].set_ylabel('Depth [m]')

raw.hke_ape.fillna(0).rolling(time=6, center=True).mean().pipe(np.log10).plot(
    vmin=-3, vmax=3, cmap='RdBu_r', ax=ax[2], rasterized=True,
                               cbar_kwargs={'pad':0.01,'label':'log HKE/APE'})
raw.mld.plot(color='k', lw=2, ax=ax[2])
ax[2].set_ylim(-500,0)
ax[2].set_xlabel(None)
ax[2].set_ylabel('Depth [m]')

alphabet(ax)

###
fig, ax = plt.subplots(2,1, sharex=True)
h = raw.hkeNI.plot(robust=True, add_colorbar=False)
raw.taudotu_ni.plot(ax=ax[0])
raw.taudotu_ni_ni.plot(ax=ax[0])
plt.colorbar(h,ax=ax[0],)

h = raw.hkeNI.plot(robust=True, ax=ax[1], add_colorbar=False)
plt.colorbar(h,ax=ax[1])