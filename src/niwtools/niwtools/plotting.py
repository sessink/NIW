import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm
from .tools import *
import pandas as pd
purp = pd.read_pickle('../data/purp_colormap.pkl')

path= '/Users/sebastianessink/Dropbox (MIT)/niw/figures/'

# set up figure params
sns.set(style='ticks', context='notebook',font_scale=1.3)
mpl.rc('figure', dpi=100, figsize=[12, 6])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

def plot_energies(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(14,8))

    raw.hkeNI.fillna(0).rolling(time=18).mean().pipe(np.log10).plot(
        vmin=-4, vmax=-1.5, cmap=purp, ax=ax[0], rasterized=True,
                                 cbar_kwargs={'pad':0.01,'label':'log HKE [$m^2 s^{-2}$]'})
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')


    raw.eta_ape.fillna(0).rolling(time=18).mean().pipe(np.log10).plot(
        vmin=-4, vmax=-1.5, cmap=purp, ax=ax[1], rasterized=True,
                               cbar_kwargs={'pad':0.01,'label':'log APE [$m^2 s^{-2}$]'})
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    raw.hke_ape.fillna(0).rolling(time=18).mean().pipe(np.log10).plot(
        vmin=0, vmax=5, cmap='RdBu_r', ax=ax[2], rasterized=True,
                                   cbar_kwargs={'pad':0.01,'label':'log HKE/APE'})
    raw.mld.plot(color='k', lw=2, ax=ax[2])
    ax[2].set_ylim(-500,0)
    ax[2].set_xlabel(None)
    ax[2].set_ylabel('Depth [m]')

    alphabet(ax)

    fig.savefig(path+f'events/{event}/hke_ape_{event}_{floatid}.pdf')

def plot_subinertial_vel(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(2,1, sharex=True, figsize=(14,8))

    (raw.u-raw.uNI).plot(cmap='RdBu_r', ax=ax[0], rasterized=True,
                                 cbar_kwargs={'pad':0.01,'label':'u [$m s^{-1}$]'})
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')


    (raw.v-raw.vNI).plot(cmap='RdBu_r', ax=ax[1], rasterized=True,
                                 cbar_kwargs={'pad':0.01,'label':'v [$m^2 s^{-1}$]'})
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/subinerial_velocity_{event}_{floatid}_nihke.pdf')

def plot_inertial_vel(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(2,1, sharex=True, figsize=(14,8))

    raw.uNI.plot(cmap='RdBu_r', ax=ax[0], rasterized=True,
                                 cbar_kwargs={'pad':0.01,'label':'u [$m s^{-1}$]'})
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')


    raw.vNI.plot(cmap='RdBu_r', ax=ax[1], rasterized=True,
                                 cbar_kwargs={'pad':0.01,'label':'v [$m^2 s^{-1}$]'})
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/inerial_velocity_{event}_{floatid}_nihke.pdf')

def plot_n2_s2(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(4,1, sharex=True, figsize=(14,8))
    raw.n2.pipe(np.log10).plot(ax=ax[0], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'N$^2$ [$s^{-2}$]'})
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.SNI.pipe(np.log10).plot( ax=ax[1], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'NI S$^2$ [s$^{-2}$]'})
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    raw.dudzNI.plot( ax=ax[2], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'NI $u_z$ [s$^{-1}$]'})
    raw.mld.plot(color='k', lw=2,ax=ax[2])
    ax[2].set_ylim(-500,0)
    ax[2].set_xlabel(None)
    ax[2].set_ylabel('Depth [m]')

    (raw.n2/raw.S2).pipe(np.log10).plot(cmap='RdBu_r', ax=ax[3], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'Ri = N$^2$ S$_{tot}^{-2}$'}, center=0.25)
    raw.mld.plot(color='k', lw=2,ax=ax[3])
    ax[3].set_ylim(-500,0)
    ax[3].set_xlabel(None)
    ax[3].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/n2andshear_{event}_{floatid}_nihke.pdf')

def plot_eps(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(2,1, sharex=True, figsize=(14,8))
    raw.eps.dropna('time','all').pipe(np.log10).plot(ax=ax[0], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':r'$\varepsilon$ [W $kg^1$]'},cmap=purp)
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.kT.dropna('time','all').pipe(np.log10).plot( ax=ax[1], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'$k_T$ [$m^2 s^{-1}$]'},cmap=purp)
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/epsilon_{event}_{floatid}_nihke.pdf')

def plot_taudotu(raw,event,floatid):
    plt.figure(figsize=(8,5))
    raw.tau.to_series().plot.area(label=r'|$\mathbf{\tau}|$')
    raw.taudotu.plot(lw=5,label=r'$\mathbf{\tau} \cdot \mathbf{u}$')
    raw.taudotu_ni.plot(lw=5,label=r'$\mathbf{\tau} \cdot \mathbf{u}_{NI}$')

    plt.legend()
    plt.ylabel('wind stress AND wind work')
    plt.xlabel(None)

    plt.savefig(path+f'events/{event}/wind_{event}_{floatid}.pdf')

def plot_power(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(2,1, sharex=True, figsize=(14,8))
    raw.dhke.dropna('time','all').rolling(time=18, center=True).mean().plot(ax=ax[0], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':r'dHKE/dt [$m^2 s^3$]'},cmap='RdBu_r')
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.dape.dropna('time','all').plot( ax=ax[1], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':'dAPE/dt [$m^2 s^3$]'},cmap='RdBu_r')
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/power_{event}_{floatid}.pdf')

def plot_efluxes(raw,event,floatid):
    sns.set(style='ticks', context='notebook', palette='colorblind', font_scale=1.3)
    f, ax = plt.subplots(3,1, sharex=True, figsize=(14,8))
    raw.up.dropna('time','all').rolling(time=18, center=True).mean().plot(ax=ax[0], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':r"<u'p'> [$m^3 s^3$]"},cmap='RdBu_r')
    raw.mld.plot(color='k', lw=2, ax=ax[0])
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.vp.dropna('time','all').rolling(time=18, center=True).mean().plot( ax=ax[1], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':r"<v'p'> [$m^3 s^3$]"},cmap='RdBu_r')
    raw.mld.plot(color='k', lw=2,ax=ax[1])
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    raw.wp.rolling(time=18, center=True).mean().plot( ax=ax[2], rasterized=True,robust=True,
                                 cbar_kwargs={'pad':0.01,'label':r"<w'p'> [$m^3 s^3$]"},cmap='RdBu_r')
    raw.mld.plot(color='k', lw=2,ax=ax[2])
    ax[2].set_ylim(-500,0)
    ax[2].set_xlabel(None)
    ax[2].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/efluxes_{event}_{floatid}.pdf')

def plot_eflux_maps(raw,event,floatid):
    sel1 = raw.where( (raw.z > -100) ).mean('z')
    sel2 = raw.where( (raw.z < -100) & (raw.z > -300) ).mean('z')
    sel3 = raw.where( (raw.z < -300) & (raw.z > -600) ).mean('z')

    fig, ax = plt.subplots(3,1, figsize=(5,12), sharex=True)

    h = ax[0].quiver(sel1.lon, sel1.lat, sel1.up, sel1.vp, sel1.dtime/86400)
    h = ax[1].quiver(sel2.lon, sel2.lat, sel2.up, sel2.vp, sel1.dtime/86400)
    h = ax[2].quiver(sel3.lon, sel3.lat, sel3.up, sel3.vp, sel1.dtime/86400)

    # fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)
    plt.colorbar(h, ax=ax[:], shrink=0.5, label='Time [days]')
    # plt.subplots_adjust()
    # plt.tight_layout()
    fig.savefig(path+f'events/{event}/efluxes_map_{event}_{floatid}.pdf')

def plot_phase(raw,event,floatid):
    f, ax = plt.subplots(2,1, figsize=(14,8), sharex=True)
    raw.ang_br0.plot(ax=ax[0], cbar_kwargs={'pad':0.01,'label':r"velocity phase"},cmap='twilight_shifted',rasterized=True)
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.angS_br0.plot(ax=ax[1], cbar_kwargs={'pad':0.01,'label':r"shear phase"},cmap='twilight_shifted',rasterized=True)
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/backrotated_phase_{event}_{floatid}.pdf')

def plot_kz(raw,event,floatid):
    f, ax = plt.subplots(2,1, figsize=(14,8), sharex=True)
    raw.hke_kz.plot(ax=ax[0], cbar_kwargs={'pad':0.01,'label':r"Energy-weighted k$_z$"},cmap='RdBu_r',rasterized=True, robust=True)
    ax[0].set_ylim(-500,0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Depth [m]')

    raw.hke_shear_kz.plot(ax=ax[1], cbar_kwargs={'pad':0.01,'label':r"Shear-weighted k$_z$"},cmap='RdBu_r',rasterized=True, robust=True)
    ax[1].set_ylim(-500,0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Depth [m]')

    alphabet(ax)

    f.savefig(path+f'events/{event}/energy_weighted_kz_{event}_{floatid}.pdf')
