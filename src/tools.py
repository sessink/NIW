def load_matfile(file):
    '''Read Matlab structure files and convert to numpy arrays'''
    import scipy.io as sio
    return sio.loadmat(file, struct_as_record=True, squeeze_me=True)


def datenum2datetime(datenum):
    '''Convert Matlab datenum to Python Datetime'''
    from datetime import datetime, timedelta
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) -\
        timedelta(days=366)


def compute_mld(data):
    '''
    mld criterion: depth of z|_(rho_10m +0.03kg/m3)

    last changed: july 2,19
    '''
    import numpy as np
    import xarray as xr
    from scipy.interpolate import interp1d

    mld = np.zeros(data.time.size)
    for t in range(data.time.size):
        test = data.isel(time=t)
        if test.rho0.size > 0:
            f = interp1d(test.rho0, test.z)
            a = test.rho0.interp(z=-10) + 0.03
            if (a > test.rho0.min()) & (a < test.rho0.max()):
                mld[t] = f(a)
            else:
                mld[t] = np.nan
        else:
            mld[t] = np.nan
    data['mld'] = ('time', mld)
    return data


def exp_moving_avg(data, tau):
    '''
        Python implementation after Eckner (2019, unpublished)
    '''
    import numpy as np
    out = np.zeros_like(data.u)
    out[0] = data.u[0]
    for j in range(2, len(data.time)):
        deltat = data.time[j] - data.time[j - 1]
        w = np.exp(-deltat / tau)
        w2 = (1 - w) / tau
        out[j] = out[j - 1] * w + data.u[j] * \
            (1 - w2) + data.u[j - 1] * (w2 - w)
    return out

def alphabet(ax):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVQXYZ'
    for j, axx in enumerate(ax):
        axx.annotate(alphabet[j], (0, 1.02),
                     xycoords='axes fraction',
                     weight='bold')
