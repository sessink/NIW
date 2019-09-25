def convert_tmsdata(chi_dir):
    from tools import load_matfile
    import pandas as pd
    import xarray as xr
    
    dat = load_matfile(chi_dir)

    f_cps = 0.5 * (dat['flabeg'] + dat['flaend'])
    time = pd.to_datetime(dat['uxt'], unit='s')

    dat = xr.Dataset(
        {
            'sla1': (['time', 'f_cps'], dat['Sla1']),
            'sla2': (['time', 'f_cps'], dat['Sla1']),
            'logavgoff': dat['logavgoff'],
            'logavgsf': dat['logavgsf'],
            'nobs': dat['nobs'],
            'floatid':chi_dir.split('-')[1]
        },
        coords={
            'time': time,
            'f_cps': f_cps
        })

    return dat


def convert_ctddata(ctd_dir):
    import gsw
    from tools import load_matfile
    import pandas as pd
    import xarray as xr
    
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
    import numpy as np
    Fc5 = 120
    Fc3 = 210  # in Hz
    sinc5 = np.sin(np.pi * Hz / Fc5) / (np.pi * Hz / Fc5)
    sinc3 = np.sin(np.pi * Hz / Fc3) / (np.pi * Hz / Fc3)
    H = (sinc5**5) * (sinc3**3)
    return H**2


def H2FP07fun(Hz, w):
    ''' H2 Fp07 transfer function

        Hz is frequency in Hz
        U is velocity in m/s
    '''
    import numpy as np
#     gamma = -0.5 # Hill, 1987
    gamma = -0.32 # Gregg&Meager, 1980
    tau0 = 0.005 # [ms] Gregg&Meager, 1980
    tau = tau0 * w**gamma
    return (1 + (2 * np.pi * Hz * tau)**2)**(-2)


def H2preampfun(Hz):
    ''' H2 Preamp transfer function
    '''
    import numpy as np
    Fc1 = 339
    Fc2 = 339
    Gd = 0.965
    # in Hz
    H2_1 = (1 - (Hz**2) / Fc1 / Fc2)**2
    H2_2 = (Hz / Fc1 + Hz / Fc2 + 2 * np.pi * Hz * Gd)**2
    H2_3 = (1 + (Hz / Fc1)**2) * (1 + (Hz / Fc2)**2)
    return H2_1 + H2_2 / H2_3




def kraichnan(k_rpm, chi, kb_rpm):
    '''
    Kraichnan temperature gradient spectrum

    adapted from: Goto et al., 2016
    '''
    import numpy as np
    import math
    D = 1.4e-7
    nu = 1.2e-6
    qk = 5.27
    
    yk = math.sqrt(qk)* k_rpm / kb_rpm
    nom = chi*math.sqrt(qk)*yk**2*np.exp(-math.sqrt(6)*yk)
    denom = (D*kb_rpm*yk)
    return nom/denom


def noise_sp(f_cps):
    # noisesp = 1.0e-11 * [1+(f/130)**3]**2
    return 1e-11 * (1 + (f_cps / 20)**3)**2

def remove_noise_sp(tms, threshold):
    '''
    Remove values that are less than the noise spectrum
    '''
    # TODO: Empirical, check against raw spectrum
    noisesp = noise_sp(tms.f_cps)
    tms['corrTsp1_cps'] = tms.corrTsp1_cps.where(
        tms.corrTsp1_cps / (threshold * noisesp) > 1, 0)
    tms['corrTsp2_cps'] = tms.corrTsp2_cps.where(
        tms.corrTsp2_cps / (threshold * noisesp) > 1, 0)
    return tms

def batchelor(k_rpm, chi, kb_rpm):
    '''
    Batchelor temperature gradient spectrum

    reference: Oakey, 1982
    adapted from: RC Lien, 1992
    '''
    import numpy as np
    import math
    from scipy.special import erfc
    D = 1.4e-7
    nu = 1.2e-6
    q = 3.7
    
    a = np.sqrt(2 * q) * k_rpm / kb_rpm
    uppera = []
    for ai in a:
        uppera.append(erfc(ai / math.sqrt(2)) * math.sqrt(0.5 * math.pi))
    g = 2 * math.pi * a * (np.exp(-0.5 * a**2) - a * np.array(uppera))
    return math.sqrt(0.5 * q) * (chi / (kb_rpm * D)) * g / (2 * math.pi)