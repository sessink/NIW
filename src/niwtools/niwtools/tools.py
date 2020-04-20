import numpy as np
import xarray as xr

def load_matfile(file):
    '''Read Matlab structure files and convert to numpy arrays'''
    import scipy.io as sio
    return sio.loadmat(file, struct_as_record=True, squeeze_me=True)

def str2date(string, format='%Y,%m,%d'):
    '''convert any date string into something comparable to
       xarray's date index (np.datetime64)
    '''
    from datetime import datetime
    import numpy as np
    if type(string) is not str:
        string = string.flatten()[0]
    return np.datetime64(datetime.strptime(string, format))

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

def interp_in_space(twod, oned):
    collect = []
    for t in oned.time:
        temp1 = oned.sel(time=t)
        temp2 = twod.sel(time=t)

        if ~temp2.lon.isnull() & ~temp2.lat.isnull():
            collect.append(temp1.interp(lon=temp2.lon,lat=temp2.lat))
    collect = xr.concat(collect,dim='time')
    return collect

def integral(data,z1,z2):
    if z2>z1:
        data = data.where( (data.z >= z1) & (data.z <= z2) ) # this is a problem because sometimes it will be the opposite order
        sign= -1
    else:
        data = data.where( (data.z >= z2) & (data.z <= z1) )
        sign= 1
    data['z'] = data.z * (-1)*sign
    array = []
    return data.dropna('z').integrate('z')

def compute_ape(ds):
    liste=[]
    for i,z in enumerate(ds.z):
        liste.append( 9.81*integral(ds.rho_prime,z,ds.zref.isel(z=i)) )
    return xr.concat( liste, dim='z')

def integrate_columns(data,lower,upper):
    '''
        Integrate each profile over depth range, e.g., MLD to 0.
    '''
    # mld=data.mld
    data = data.where( (data.z >= lower) & (data.z < upper) )
    # data['z'] = data.z * (-1)
    array = []
    for t in range(data.time.size):
        # TODO: could do better here with simpson's rule
        if data.isel(time=t).dropna('z').size > 3:
            array.append(data.isel(time=t).dropna('z').integrate('z'))
        else:
            array.append(data.isel(time=t).dropna('z').integrate('z')*np.nan)

    # find zmin observed depth
    if np.mean(upper)==0:
        zmin = data[first_finite(data,0)].z
    else:
        zmin = upper

    return xr.concat(array/(lower-zmin), dim='time')

def bandpass_velocity(raw, low_f, high_f):
    import xrscipy.signal as dsp
    import gsw

    # TODO: fix naming of variables

    # convert datetime to seconds since t=0
    raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values*1e-9, dtype=float))
    # raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values, dtype=float))

    # make dtime a dimension
    raw = raw.swap_dims({'time':'dtime'})

    # filtering proceduce
    # determine sampling timestep and Nyquist frequency
    T = ( dsp.get_sampling_step(raw, dim='dtime') )
    # T = 5000
    fs = 1/T
    ny = 0.5*fs

    # limits for bandpass
    # low_f = gsw.f(lat_mean)*low # in 1/s
    # high_f = gsw.f(lat_mean)*high # in 1/s
    eps=0 # how to fill nans
    # # pick an order?
    print(low_f/ny)
    print(high_f/ny)
    ulow = dsp.bandpass(raw.u.fillna(eps), low_f/ny, high_f/ny, dim='dtime', in_nyq=True, order=4)
    vlow = dsp.bandpass(raw.v.fillna(eps), low_f/ny, high_f/ny, dim='dtime', in_nyq=True, order=4)
    # ulow = dsp.bandpass(raw.u.fillna(eps), low_f, high_f, dim='dtime', in_nyq=False, order=4)
    # vlow = dsp.bandpass(raw.v.fillna(eps), low_f, high_f, dim='dtime', in_nyq=False, order=4)

    # swap dims back
    ulow = ulow.swap_dims({'dtime':'time'})
    vlow = vlow.swap_dims({'dtime':'time'})
    raw = raw.swap_dims({'dtime':'time'})

    # remove time and space means?
    ulow = ulow #- ulow.mean(dim='z') - ulow.mean(dim='time')
    vlow = vlow #- vlow.mean(dim='z') - ulow.mean(dim='time')

    mask = ~np.isnan(raw.u) & ~np.isnan(raw.v)
    raw['uNI'] = ulow.where(mask)
    raw['vNI'] = vlow.where(mask)

    return raw

def bandpass_variable(raw,array, low_f, high_f):
    import xrscipy.signal as dsp
    import gsw

    # TODO: fix naming of variables

    # convert datetime to seconds since t=0
    raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values*1e-9, dtype=float))
    # raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values, dtype=float))

    # make dtime a dimension
    raw = raw.swap_dims({'time':'dtime'})

    # filtering proceduce
    # determine sampling timestep and Nyquist frequency
    T = ( dsp.get_sampling_step(raw, dim='dtime') )
    fs = 1/T
    ny = 0.5*fs

    # limits for bandpass
    eps=0 # how to fill nans
    # # pick an order?
    print(low_f/ny)
    print(high_f/ny)
    ulow = dsp.bandpass(raw[array].fillna(eps), low_f/ny, high_f/ny, dim='dtime', in_nyq=True, order=4)

    # swap dims back
    ulow = ulow.swap_dims({'dtime':'time'})
    raw = raw.swap_dims({'dtime':'time'})

    # remove time and space means?
    ulow = ulow #- ulow.mean(dim='z') - ulow.mean(dim='time')

    mask = ~np.isnan(raw[array])
    varname = array+'NI'
    raw[varname] = ulow.where(mask)

    return raw

def bandstop_variable(raw,array, low_f, high_f):
    import xrscipy.signal as dsp
    import gsw

    # TODO: fix naming of variables

    # convert datetime to seconds since t=0
    raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values*1e-9, dtype=float))
    # raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values, dtype=float))

    # make dtime a dimension
    raw = raw.swap_dims({'time':'dtime'})

    # filtering proceduce
    # determine sampling timestep and Nyquist frequency
    T = ( dsp.get_sampling_step(raw, dim='dtime') )
    fs = 1/T
    ny = 0.5*fs

    # limits for bandpass
    eps=0 # how to fill nans
    # # pick an order?
    print(low_f/ny)
    print(high_f/ny)
    ulow = dsp.bandstop(raw[array].fillna(eps), low_f/ny, high_f/ny, dim='dtime', in_nyq=True, order=4)

    # swap dims back
    ulow = ulow.swap_dims({'dtime':'time'})
    raw = raw.swap_dims({'dtime':'time'})

    # remove time and space means?
    ulow = ulow #- ulow.mean(dim='z') - ulow.mean(dim='time')

    mask = ~np.isnan(raw[array])
    varname = array+'NI'
    raw[varname] = ulow.where(mask)

    return raw

def lowpass_variable(raw,array, low_f, high_f):
    import xrscipy.signal as dsp
    import gsw

    # TODO: fix naming of variables

    # convert datetime to seconds since t=0
    raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values*1e-9, dtype=float))
    # raw['dtime'] = ('time', np.array( (raw.time - raw.time.isel(time=0)).values, dtype=float))

    # make dtime a dimension
    raw = raw.swap_dims({'time':'dtime'})

    # filtering proceduce
    # determine sampling timestep and Nyquist frequency
    T = ( dsp.get_sampling_step(raw, dim='dtime') )
    fs = 1/T
    ny = 0.5*fs

    # limits for bandpass
    eps=0 # how to fill nans
    # # pick an order?
    print(low_f/ny)
    print(high_f/ny)
    ulow = dsp.lowpass(raw[array].dropna('z'), low_f/ny, dim='dtime', in_nyq=True, order=4)

    # swap dims back
    ulow = ulow.swap_dims({'dtime':'time'})
    raw = raw.swap_dims({'dtime':'time'})

    # remove time and space means?
    ulow = ulow #- ulow.mean(dim='z') - ulow.mean(dim='time')

    mask = ~np.isnan(raw[array])
    varname = array+'LOW'
    raw[varname] = ulow.where(mask)

    return raw

def first_finite(arr, axis):
    '''spits out the indices'''
    mask = arr.notnull() & (arr > 0)
    return xr.where(mask.any(axis=axis), mask.argmax(axis=axis), np.nan).fillna(0).astype(int)

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

def avg_funs(array1, array2):
    '''take average taking into account nans'''
    import xarray as xr
    concat = xr.concat([array1, array2], dim='temp')
    return concat.mean(dim='temp')
