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
            'nobs': dat['nobs']
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


def H2FP07fun(Hz, U):
    ''' H2 Fp07 transfer function

        Hz is frequency in Hz
        U is velocity in m/s
    '''
    import numpy as np
    tau = 0.006 * U**(-0.5)
    return (1 + (2 * np.pi * Hz * tau)**2)**(-1)


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

