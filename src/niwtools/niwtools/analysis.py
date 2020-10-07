import numpy as np
import pandas as pd
import xarray as xr
import gsw
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import xrscipy.signal as dsp
from xarrayutils.vertical_coordinates import linear_interpolation_regrid, linear_interpolation_remap
from .tools import *

def compute_shear(raw):
    raw['dudz'] = raw.u.differentiate('z',)
    raw['dvdz'] = raw.v.differentiate('z')
    raw['S'] = 0.5*(raw.dudz**2 + raw.dvdz**2)

    raw['dudzNI'] = raw.uNI.differentiate('z')
    raw['dvdzNI'] = raw.vNI.differentiate('z')
    raw['SNI'] = 0.5*(raw.dudzNI**2 + raw.dvdzNI**2)
    return raw

def get_new_coodinates(data, variable, new_min, new_max):
    ''' regrid data onto sigma
    '''
    if data.z.min()<0:
        data.coords['z'] = -data.z
    new_vals = np.arange(new_min,new_max, 0.01)
    new_values = xr.DataArray(new_vals, coords=[('sigma', new_vals)]) # define the new temperature grid

    return linear_interpolation_regrid(data.z, data[variable], new_values, target_value_dim='sigma')

def remap_variable(data,variable,new_coords):
    if data.z.min()<0:
        data.coords['z'] = -data.z
    _,data['zmap'] = xr.broadcast(data[variable],data.z)
    return linear_interpolation_remap(data.z, data.zmap, new_coords)

def compute_iso_disp(raw):
    f = gsw.f(40.7)/(2*np.pi)
    raw = lowpass_variable(raw,'rho0', 0.6*f, 1.25*f)
    raw['rho_ref'] = raw.rho0LOW

    raw['sigma'] = raw.rho0 -1000
    raw['sigma_ref'] = raw.rho_ref -1000

    raw['sigma'] = xr.where(raw['sigma']<20, np.nan, raw['sigma'])
    raw['sigma_ref'] = xr.where(raw['sigma_ref']<20, np.nan, raw['sigma_ref'])
    new_min = raw.sigma.min()
    new_max = raw.sigma.max()
    #return raw

    sigma_coords = get_new_coodinates(raw,'sigma', new_min, new_max)
    sigma_ref_coords = get_new_coodinates(raw,'sigma_ref', new_min, new_max)

    ds_z = remap_variable(raw,'sigma',sigma_coords).rename({'remapped':'sigma'})
    ds_z_ref = remap_variable(raw,'sigma_ref',sigma_ref_coords).rename({'remapped':'sigma'})
    eta_sigma = (ds_z_ref-ds_z).transpose() # this is in sigma space

    z_values = xr.DataArray(raw.z.values, coords=[('z', raw.z.values)]) # define the new z grid
    z_coord = linear_interpolation_regrid(ds_z.sigma, ds_z.transpose(), z_values, target_value_dim='z')
    eta = linear_interpolation_remap(eta_sigma.sigma, eta_sigma, z_coord).transpose()
    raw['eta'] = eta.rename({'remapped':'z'}) #.transpose()

    if raw.z.max()>0:
        raw.coords['z'] = -raw.z
    return raw

def compute_tendency(raw):
    raw['dape'] = (raw.eta_ape.resample(time='1h')
        .interpolate('linear').rolling(time=18, center=True)
        .mean()
        .differentiate('time',datetime_unit='s')).interp_like(raw.eta_ape)
    raw['dhke'] = (raw.hkeNI.resample(time='1h')
        .interpolate('linear').rolling(time=18, center=True)
        .mean()
        .differentiate('time',datetime_unit='s')).interp_like(raw.hkeNI)
    return raw

def phase_cos(t, a, b, c):
    fi = gsw.f(40.7)
    return a*np.cos(fi*t + b) + c

def cosine_fit(tt,yy):
    # those are the window time vectors
    yy = yy[~np.isnan(yy)]
    if len(yy)>4:
        try:
            tt = tt[~np.isnan(yy)]
            popt, _ = curve_fit(phase_cos, tt, yy) # fit over window
        except:
            popt = np.array([np.nan,np.nan,np.nan])
    else:
        popt = np.array([np.nan,np.nan,np.nan])
    return popt

def xr_cosine_fit(tt,yy):
    return xr.apply_ufunc(cosine_fit, tt, yy,
                          input_core_dims=[['new'],['new']],
                          output_core_dims=[['popt']],
                          vectorize=True,
                          output_dtypes=['float64'],
                          output_sizes={'popt': 3})

def fit_cosine_to_eta(raw):
    raw['ddtime'] = raw.dtime
    test = raw[['ddtime','eta']].dropna('time', how='all').dropna('z', how='all')
#     test = test.resample(time='2h').interpolate('linear')
    test = test.swap_dims({'time':'dtime'})
    a = test.rolling(dtime=18, center=True).construct('new')
    results = xr_cosine_fit(a.ddtime, a.eta)
    test['amp'] = results.isel(popt=0,drop=True)
    test['phase'] = results.isel(popt=1,drop=True)
    test['resid'] = results.isel(popt=2,drop=True)
    test['fit'] = phase_cos(test.dtime, test.amp, test.phase, test.resid)
#     raw['eta_fit'] = phase_cos(test.ddtime, test.amp, test.phase, test.resid)
    return test

def integral_1d2(zz,yy):
#     print(data.dims)
    result = []
    zlist = []

    for zi,z in enumerate(zz[:-100]):
        ztest = zz[zi:]
        ytest= yy[zi:]
#         print( ztest[~np.isnan(ytest)].size)
        res = trapz(ztest[~np.isnan(ytest)],ytest[~np.isnan(ytest)])
        if res.size >0 :
            result.append( res )
            zlist.append(z)
        else:
            result.append( np.nan )
            zlist.append(np.nan)
#     print(len(zlist))
#     print(len(result))
    return np.array(result), -np.array(zlist)

def wrap_integral(zz,yy):
    return xr.apply_ufunc(integral_1d2,zz,yy,
                          input_core_dims=[['z'],['z']],
                          output_core_dims=[['newz'],['newz']],
                          vectorize=True)

def find_pprime(raw):
    raw['bprime_fit'] = -raw.n2*raw.eta_fit
    raw['bprime_fit_anom'] = raw.bprime_fit - raw.bprime_fit.mean('time') - raw.bprime_fit.mean('z')

    testwrap, testwrapz = wrap_integral(raw.bprime_fit_anom.z,raw.bprime_fit_anom)
    testwrap['newz'] = ('newz', -testwrapz.mean('time'))
    raw['pprime'] = testwrap.interp(newz=raw.z).transpose()

    raw['pprime'] = raw.pprime - raw.pprime.mean('z')
    return raw

def compute_efluxes(raw):
    raw['up'] = (raw.uNI-raw.uNI.mean('time')-raw.uNI.mean('z'))*raw.pprime
    raw['vp'] = (raw.vNI-raw.vNI.mean('time')-raw.vNI.mean('z'))*raw.pprime
    raw['wp'] = (raw.w_fit*raw.pprime)
    return raw

def backrotate_phase(ang):
    ref_time = pd.to_datetime('28/12/1988')
    timestamp = pd.to_datetime(ang.time.values)
    Tf =2 * np.pi / gsw.f(40.7)
    dt = (timestamp - ref_time) / pd.to_timedelta(1, unit='s') % Tf
    phase_add = (dt.values * gsw.f(40.7)).astype('float')
    ang_br = ang + phase_add*180/np.pi

    return wrap(ang_br)

def backrotate_rc(phase0):
    f= gsw.f(40.7)
    ref_time = pd.to_datetime('28/12/1988')
    timestamp = pd.to_datetime(phase0.time.values)
    dt = (timestamp - ref_time) / pd.to_timedelta(1, unit='s')
    backrotate = np.arctan2(np.sin(f*dt.values),np.cos(f*dt.values))*180/np.pi

    return wrap(phase0 + backrotate)

def wrap(test, deg=True):
    if deg:
        phases = (( -test + 180) % (2.0 * 180 ) - 180) * -1.0
    else:
        phases = (( -test + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0
    return phases

def compute_phase(raw):
    raw['ang'] =  ( ('z','time'),  np.angle(raw.uNI + 1j*raw.vNI, deg=True))
    raw['angS'] =  ( ('z','time'), np.angle(raw.dudzNI + 1j*raw.dvdzNI, deg=True) )
    raw['angS_br0'] = backrotate_phase(raw.angS)
    raw['ang_br0'] = backrotate_phase(raw.ang)
    raw['rcs_phase'] = backrotate_rc(raw.ang)
    return raw

def compute_kz(raw):
    raw['kz_vel'] = raw.ang_br0.pipe(np.radians).differentiate('z').pipe(lambda x: x/(2*np.pi))
    raw['kz_vel1'] = (raw.ang_br0.pipe(np.radians).pipe(np.cos).differentiate('z')/ raw.ang_br0.pipe(np.radians).pipe(np.sin)).pipe(lambda x: -x/(2*np.pi))
    raw['kz_shear'] = raw.angS_br0.pipe(np.radians).differentiate('z').pipe(lambda x: x/(2*np.pi))
    raw['kz_shear1'] = (raw.angS_br0.pipe(np.radians).pipe(np.cos).differentiate('z')/ raw.angS_br0.pipe(np.radians).pipe(np.sin)).pipe(lambda x: -x/(2*np.pi))
    raw['hke_kz'] = raw.kz_vel1*raw.hkeNI/raw.hkeNI.mean('time')
    raw['hke_shear_kz'] = raw.kz_shear1*raw.S2/raw.S2.mean('time')
    return raw
