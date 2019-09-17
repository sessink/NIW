# Scientific Computing
# Standard Library

import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
# %%
from scipy.optimize import curve_fit, minimize

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmocean import cm

from sklearn.linear_model import LinearRegression
from src.tools import alphabet, load_matfile, str2date

import warnings
warnings.simplefilter("ignore")

# set up figure params
sns.set(style='ticks', context='paper', palette='colorblind')
mpl.rc('figure', dpi=100, figsize=[11, 5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %%
def convert_tmsdata(chi_dir):
    dat = load_matfile(chi_dir)

    f_cps = 0.5 * (dat['flabeg'] + dat['flaend'])
    time = pd.to_datetime(dat['uxt'], unit='s')

    dat = xr.Dataset(
        {
            'sla1': (['time', 'f_cps'], dat['Sla1']),
            'sla2': (['time', 'f_cps'], dat['Sla1']),
            'logavgoff': dat['logavgoff'],
            'logavgsf': dat['logavgsf'],
            'nobs': dat['nobs']
        },
        coords={
            'time': time,
            'f_cps': f_cps
        })

    return dat


def convert_ctddata(ctd_dir):
    import gsw
    dat = load_matfile(ctd_dir)
    time = pd.to_datetime(dat['UXT'], unit='s')

    dat = xr.Dataset(
        {
            'T': ('time', dat['T']),
            'S': ('time', dat['S']),
            'p': ('time', dat['P'])
        },
        coords={
            'time': time,
        })

    # TODO: need to check units when integrating!
    dat['sigma'] = ('time', gsw.sigma0(dat['S'], dat['T']) + 1000)
    dat['z'] = -dat.p
    dat['w'] = dat.z.differentiate('time', datetime_unit='s')
    temp = dat.swap_dims({'time': 'z'})
    temp['N2'] = -9.81 * temp.sigma.differentiate('z') / 1025
    temp['dTdz'] = temp.T.differentiate('z')

    return temp.swap_dims({'z': 'time'})


def H2ADCfun(Hz):
    ''' H2 ADC transfer function
    '''

    Fc5 = 120
    Fc3 = 210  # in Hz
    sinc5 = np.sin(np.pi * Hz / Fc5) / (np.pi * Hz / Fc5)
    sinc3 = np.sin(np.pi * Hz / Fc3) / (np.pi * Hz / Fc3)
    H = (sinc5**5) * (sinc3**3)
    return H**2


def H2FP07fun(Hz, U):
    ''' H2 Fp07 transfer function

        Hz is frequency in Hz
        U is velocity in m/s
    '''
    tau = 0.006 * U**(-0.5)
    return (1 + (2 * np.pi * Hz * tau)**2)**(-1)


def H2preampfun(Hz):
    ''' H2 Preamp transfer function
    '''
    Fc1 = 339
    Fc2 = 339
    Gd = 0.965
    # in Hz
    H2_1 = (1 - (Hz**2) / Fc1 / Fc2)**2
    H2_2 = (Hz / Fc1 + Hz / Fc2 + 2 * np.pi * Hz * Gd)**2
    H2_3 = (1 + (Hz / Fc1)**2) * (1 + (Hz / Fc2)**2)
    return H2_1 + H2_2 / H2_3


def remove_noise_sp(tms, threshold):
    # TODO: Empirical, check against raw spectrum
    noisesp = noise_sp(tms.f_cps)
    tms['corrTsp1_cps'] = tms.corrTsp1_cps.where(
        tms.corrTsp1_cps / (threshold * noisesp) > 1, 0)
    tms['corrTsp2_cps'] = tms.corrTsp2_cps.where(
        tms.corrTsp2_cps / (threshold * noisesp) > 1, 0)
    return tms


def compute_batchelor_sp(tms):
    '''
    Batchelor temperature gradient spectrum

    reference: Oakey, 1982
    adapted from: RC Lien, 1992
    '''
    from scipy.special import erfc

    D = 1.4e-7
    nu = 1.2e-6
    q = 3.7

    kb = (tms.eps1 / nu / D**2)**(0.25)
    a = np.sqrt(2 * q) * tms.k_rpm / kb
    uppera = []
    for ai in a:
        uppera.append(erfc(ai / np.sqrt(2)) * np.sqrt(0.5 * np.pi))
    g = 2 * np.pi * a * (np.exp(-0.5 * a**2) - a * np.array(uppera))
    tms['batchelorsp1'] = np.sqrt(0.5 * q) * (tms.chi1 /
                                              (kb * D)) * g / (2 * np.pi)

    kb = (tms.eps2 / nu / D**2)**(0.25)
    a = np.sqrt(2 * q) * tms.k_rpm / kb
    uppera = []
    for ai in a:
        uppera.append(erfc(ai / np.sqrt(2)) * np.sqrt(0.5 * np.pi))
    g = 2 * np.pi * a * (np.exp(-0.5 * a**2) - a * np.array(uppera))
    tms['batchelorsp2'] = np.sqrt(0.5 * q) * (tms.chi2 /
                                              (kb * D)) * g / (2 * np.pi)
    return tms


def batchelor(k_rpm, chi, kb_rpm):
    from scipy.special import erfc
    D = 1.4e-7
    nu = 1.2e-6
    q = 3.7
    # kbref = (tms.eps1 / nu / D**2)**(0.25)

    a = np.sqrt(2 * q) * k_rpm / kb_rpm
    uppera = []
    for ai in a:
        uppera.append(erfc(ai / np.sqrt(2)) * np.sqrt(0.5 * np.pi))
    g = 2 * np.pi * a * (np.exp(-0.5 * a**2) - a * np.array(uppera))
    return np.sqrt(0.5 * q) * (chi / (kb_rpm * D)) * g / (2 * np.pi)


def kraichnan(k_rpm, chi, kb_rpm):
    from scipy.special import erfc
    D = 1.4e-7
    nu = 1.2e-6
    q = 3.7

    yk = np.sqrt(q_kr)* k_rpm / kb_rpm
    return

def noise_sp(f_cps):
    # noisesp = 1.0e-11 * [1+(f/130)**3]**2
    return 1e-11 * (1 + (f_cps / 20)**3)**2


def chiprofile(tms, ctd):
    ''' Procedure to calculate chi from EM-APEX float

    see: RC's write-up "EM-APEX Turbulence Measurements"
    '''
    from scipy.interpolate import interp1d

    # tms = convert_tmsdata(chi_dir)
    # tms = tms.isel(time=100)
    # tms
    # % 1) convert realtime-transmitted scaled spectrum (sla)
    # to digitized voltage Spectrum
    tms['slad1'] = (tms.sla1 - tms.logavgoff) / tms.logavgsf
    tms['slad2'] = (tms.sla2 - tms.logavgoff) / tms.logavgsf

    # % 2) convert to raw spectrum of temperature
    beta = 25
    Vref = 4  # volt
    Inet = 0.8
    scale2 = (beta * Vref / (2**23 * Inet))**2
    tms['rawTsp1'] = 10**(tms.slad1 / 10) * scale2
    tms['rawTsp2'] = 10**(tms.slad2 / 10) * scale2

    # % 3) get background T,N,P,W, and dT/dz from ctd
    tms['p'] = ctd.p.interp(time=tms.time)
    tms['N2'] = ctd.N2.interp(time=tms.time)
    tms['N'] = np.abs(np.sqrt(tms['N2']))
    tms['T'] = ctd.T.interp(time=tms.time)
    tms['dTdz'] = ctd.dTdz.interp(time=tms.time)
    tms['w'] = np.abs(ctd.w.interp(time=tms.time))

    tms['k_cpm'] = tms.f_cps / tms.w
    tms['f_rps'] = tms.f_cps * 2 * np.pi
    tms['k_rpm'] = tms.f_rps / tms.w

    # % 4) compute transfer functions and compute corrected T spectrum
    tms['H2adc'] = H2ADCfun(tms.f_cps)
    tms['H2preamp'] = H2preampfun(tms.f_cps)
    tms['H2fp07'] = H2FP07fun(tms.f_cps, tms.w)
    tms['H2total_cps'] = tms.H2adc * tms.H2preamp * tms.H2fp07
    tms['corrTsp1_cps'] = tms.rawTsp1 / tms.H2total_cps
    tms['corrTsp2_cps'] = tms.rawTsp2 / tms.H2total_cps

    # % 5) remove noise Spectrum
    threshold = 4
    tms = remove_noise_sp(tms, threshold)

    # % 6) convert temperature frequency spectrum to wavenumber Spectrum
    tms['corrTsp1_rpm'] = tms.corrTsp1_cps * tms.w / (2 * np.pi)
    tms['corrdTdzsp1_rpm'] = tms.k_rpm**2 * tms.corrTsp1_rpm

    tms['corrTsp2_rpm'] = tms.corrTsp2_cps * tms.w / (2 * np.pi)
    tms['corrdTdzsp2_rpm'] = tms.k_rpm**2 * tms.corrTsp2_rpm

    # % 7) compute chi, kT, and eps1
    kzmin = 20
    kzmax = 400
    gamma = 0.2
    D = 1.4e-7

    # QUESTION: most values are k_rpm >> 400, is that right?
    tms = tms.swap_dims({'f_cps': 'k_rpm'})
    condition = (tms.k_rpm <= kzmax) & (tms.k_rpm >= kzmin)

    if condition.sum() >= 3:
        tms['chi1'] = 6 * D * tms.corrdTdzsp1_rpm.where(condition).dropna(
            dim='k_rpm').integrate('k_rpm')
        tms['chi2'] = 6 * D * tms.corrdTdzsp2_rpm.where(condition).dropna(
            dim='k_rpm').integrate('k_rpm')
    else:
        tms['chi1'] = np.nan
        tms['chi2'] = np.nan

    tms['kt1'] = 0.5 * tms.chi1 / tms.dTdz**2
    tms['eps1'] = tms.kt1 * tms.N2 / gamma

    tms['kt2'] = 0.5 * tms.chi2 / tms.dTdz**2
    tms['eps2'] = tms.kt2 * tms.N2 / gamma

    tms = compute_batchelor_sp(tms)

    # 8) Goto method
    def cost_function(kb, f_cps, w, k_rpm, chi, corrdTdz):
        '''
        Cost function for MLE to fit spectra

        see: Ruddick et al, 1996 and Goto et al., 2016
        '''
        from scipy.stats import chi2
        kzmin = 20
        kzmax = 400
        df = 2
        noise = noise_sp(f_cps) * w / (2 * np.pi)
        a = df / (batchelor(k_rpm.values, chi.values, kb) + noise)
        b = chi2.pdf(corrdTdz * a, df)
        c = np.log(a * b)

        condition = (k_rpm <= kzmax) & (k_rpm >= kzmin)
        return -np.sum(c.where(condition)).values


    def signal_to_noise(observed,noise):
        return observed/noise

    plt.plot( signal_to_noise(tms.corrdTdzsp1_rpm,noise_sp(tms.f_cps) * tms.w / (2 * np.pi)) )


    D = 1.4e-7
    nu = 1.2e-6
    tms['kb1_gt'] = minimize(cost_function,
                             x0=340,
                             args=(tms.f_cps, tms.w, tms.k_rpm, tms.chi1,
                                   tms.corrdTdzsp1_rpm)).x[0]
    tms['eps1_gt'] = tms['kb1_gt']**4 * nu * D**2

    tms['kb2_gt'] = minimize(cost_function,
                             x0=340,
                             args=(tms.f_cps, tms.w, tms.k_rpm, tms.chi2,
                                   tms.corrdTdzsp2_rpm)).x[0]
    tms['eps2_gt'] = tms['kb2_gt']**4 * nu * D**2  #* (2 * np.pi)**4


    return tms


# %% MAIN
liste = ['7786b-0200','7786b-0300']
all_profiles=[]
for l in liste:
    l = '7786b-0200'
    chi_dir = 'data/chi/ema-'+l+'-tms.mat'
    tms = convert_tmsdata(chi_dir)
    ctd_dir = 'data/chi/ema-'+l+'-ctd.mat'
    ctd = convert_ctddata(ctd_dir)

    turb = []
    for jblock in range(tms.nobs.values):
        # tms = tms.isel(time=jblock)
        tms_block = tms.isel(time=jblock)
        tms_block = chiprofile(tms_block, ctd)
        tms_block = tms_block.swap_dims({'k_rpm': 'f_cps'})
        turb.append(tms_block)

    turb = xr.concat(turb, dim='time')
    all_profiles.append(turb)

all_profiles =  xr.concat(all_profiles, dim='time')

# %% Plotting
blocks = np.arange(0, turb.time.size, 10)

for i in blocks:
    f = plt.figure(figsize=(8, 20))

    ax0 = f.add_subplot(4, 1, 1)
    temp = turb.isel(time=i)
    ax0.plot(temp.f_cps, temp.H2adc, label='H2adc')
    ax0.plot(temp.f_cps, temp.H2fp07, label='H2fp07')
    ax0.plot(temp.f_cps, temp.H2preamp, label='H2preamp')
    ax0.plot(temp.f_cps, temp.H2total_cps, label='H2total')
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_ylabel(r'Transfer function squared H^2')
    ax0.axvline(2e1)
    ax0.axvline(4e2)
    ax0.legend()

    ax1 = f.add_subplot(4, 1, 2, sharex=ax0)
    ax1.plot(temp.f_cps, temp.rawTsp1, label='raw')
    ax1.plot(temp.f_cps, temp.corrTsp1_cps, label='corrected')
    ax1.plot(temp.f_cps, noise_sp(temp), ls='--', label='noise')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel(r'Raw and Corrected $\Phi_T$ [C$^2$ Hz$^{-1}$]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(2e1)
    ax1.axvline(4e2)
    ax1.legend()

    ax3 = f.add_subplot(4, 1, 3, sharex=ax2)
    ax3.plot(temp.k_rpm,
             temp.corrdTdzsp1_rpm,
             label=r'$\Phi_{\partial_z T}^1$',
             color='C0')
    # ax3.plot(temp.k_rpm,temp.corrdTdzsp2_rpm,marker='+',label=r'$\Phi_{\partial_z T}^2$')
    ax3.plot(temp.k_rpm,
             temp.batchelorsp2,
             ls='--',
             label=r'Batchelor',
             color='C1')
    ax3.plot(temp.k_rpm,
             temp.corrTsp1_rpm,
             label=r'Corrected $\Phi_T$(k_z)',
             color='C2')
    ax3.set_xlabel('k$_z$ [m$^{-1}$]')
    ax3.set_ylabel(r'Corrected $\Phi_{\partial_z T}$ [C$^2$ m$^{-1}$]')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.axvline(2e1)
    ax3.axvline(4e2)
    ax3.set_ylim(1e-9, 1e-2)
    ax3.legend()

    ax4 = ax3.twinx()
    ax4.plot(temp.k_rpm, temp.k_rpm * temp.corrdTdzsp1_rpm, color='C3')
    # ax4.plot(temp.k_rpm,temp.k_rpm*temp.corrdTdzsp2_rpm,marker='+')
    ax4.plot(temp.k_rpm, temp.k_rpm * temp.batchelorsp2, ls='--', color='C4')
    ax4.plot(temp.k_rpm,
             temp.k_rpm * temp.corrTsp1_rpm,
             label=r'Corrected $\Phi_T$(k_z)',
             color='C2')
    ax4.set_xlabel('k$_z$ [m$^{-1}$]')
    ax4.set_ylabel(r'Corrected $k_z\Phi_{\partial_z T}$ [C$^2$ m$^{-1}$]')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.axvline(2e1)
    ax4.axvline(4e2)
    ax4.set_ylim(1e-9, 1e-2)

    plt.savefig(f'figures/chi_calculation/0chi_t{i:03d}.pdf')
    plt.close()

# %%
blocks = np.arange(0, turb.time.size, 10)
plt.figure(figsize=(6, 5))
for i in blocks:
    temp = turb.isel(time=i)
    plt.plot(temp.k_rpm,
             temp.corrdTdzsp1_rpm,
             label=r'$\Phi_{\partial_z T}^1$',
             color='C0')
    plt.plot(temp.k_rpm,
             batchelor(temp.k_rpm, temp.chi1, temp.kb1_gt),
             '--',
             color='r',
             label='Batchelor Goto')
    plt.plot(temp.k_rpm,
             temp.batchelorsp1,
             '--',
             color='g',
             label='Batchelor Old')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-9, 1e-2)
    plt.legend()
    plt.savefig(f'figures/chi_calculation/compare_eps/compare_{i:03}.pdf')
    plt.close()

# %%
blocks = np.arange(0, turb.time.size, 10)
plt.figure(figsize=(6, 5))
for i in blocks:
    temp = turb.isel(time=i)
    # plt.plot(temp.k_rpm,
    #              temp.corrdTdzsp1_rpm,
    #              label=r'$\Phi_{\partial_z T}^1$',
    #              color='C0')
    plt.plot(temp.k_rpm,
             np.log10(
                 batchelor(temp.k_rpm, temp.chi1, temp.kb1_gt) /
                 temp.batchelorsp1),
             '--',
             color='r',
             label='log( Goto/Old )')
    # plt.plot(temp.k_rpm,temp.batchelorsp1,'--',color='g',label='Batchelor Old')
    plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(1e-9, 1e-2)
    plt.legend()
    plt.savefig(f'figures/chi_calculation/compare_eps/ratio_{i:03}.pdf')
    plt.close()

# %%

def func(x, m, n):
    return m * x + n

x = all_profiles.eps1.dropna('time')
y = all_profiles.eps1_gt.dropna('time')

x = np.log( x.dropna('time'))
y = np.log( y.dropna('time'))
okay = ~np.isnan(x) & ~np.isnan(y)
coeff, pcov = curve_fit(func, x.where(okay), y.where(okay))
perr = np.sqrt(np.diag(pcov))

residuals = y - func(x, *coeff)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

plt.figure(figsize=(6, 6))
plt.plot(np.exp(x.where(okay)), np.exp(y.where(okay)), '.')
plt.plot(np.exp(x), np.exp(func(x, coeff[0], coeff[1])))
# plt.ylim(1e-12, 1e-7)
# plt.xlim(1e-12, 1e-7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ren-Chieh's epsilon")
plt.ylabel("Goto's epsilon")
plt.title('7786b-0200')
plt.annotate(fr'R$^2$ {r_squared.values:2.2f}', (0.2, 0.85),
             xycoords='figure fraction')
plt.annotate(f'm = {coeff[0]:2.2f},' + '\n' + f'n = {coeff[1]:2.2f}',
             (0.7, 0.85),
             xycoords='figure fraction')
plt.savefig('figures/chi_calculation/compare_eps/regress_eps.pdf')
plt.show()

# %%
blocks = np.arange(0, turb.time.size, 10)

f, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
turb.eps1.pipe(np.log10).plot.hist(ax=ax[0])
turb.eps1_gt.pipe(np.log10).plot.hist(ax=ax[1])
plt.xlim(-12, -5)
plt.savefig(f'figures/chi_calculation/compare_eps/eps_hist.pdf')
plt.close()
