def compute_mld(data):
	'''
	mld criterion: depth of z|_(rho_10m +0.03kg/m3)

	'''
	import numpy as np
	import xarray as xr
	from scipy.interpolate import interp1d

	mld = np.zeros(data.time.size)
	for t in range(data.time.size):
		test = data.isel(time=t)
		# test = test.dropna(dim='z')
		if test.sigma.size>0:
			f = interp1d(test.sigma,test.z)
			a = test.sigma.interp(z=-10)+0.03
			a
			if (a>test.sigma.min()) & (a<test.sigma.max()):
				mld[t] = f(a)
			else:
				mld[t]=np.nan
		else:
			mld[t]=np.nan
	data['mld'] = ('time',mld)
	return data


def exp_moving_avg(data,tau):
    '''
        Python implementation after Eckner (2019, unpublished)
    '''
    out = np.zeros_like(data.u)
    out[0] = data.u[0]
    for j in range(2,len(data.time)):
        deltat = data.time[j]-data.time[j-1]
        w = np.exp(-deltat/tau)
        w2 = (1-w)/tau
        out[j] = out[j-1]*w + data.u[j]*(1-w2) + data.u[j-1]*(w2-w)
    return out
