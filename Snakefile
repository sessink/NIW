import pandas as pd

configfile: 'config.yaml'
workdir: config['path']
FLOATS = config['floats']
filter_period = str(config['filter_period'])
resample_period = config['resample_period']
dates = pd.date_range(start='2017-11-01', end='2017-12-01', freq='6H').strftime("%d%m%y_%Hh")

include: 'utils.smk'

rule all:
    input:
        expand('data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/summary/summary_{float}_{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/ni_currents/compare_ni_{float}_{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/ni_currents/compare_ni_hke_{float}_{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/ml_timeseries/ml_timeseries_{float}__{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        # expand('data/ml/mlall_{resample_period}_{filter_period}Tf.nc',
        #        filter_period=filter_period, resample_period=resample_period),
        # expand('figures/nio_maps_{float}_{filter_period}Tf.pdf',float=FLOATS,filter_period=filter_period),
        expand('figures/weather_maps/weather_{date}.pdf',date=dates),
        'figures/float_traj.pdf',
        expand("viz/{graph}.{fmt}", graph=["rulegraph", "dag"], fmt=["pdf", "png"]),

rule convert_mat_files:
	input:
		'data/NIWmatdata/{float}.mat'
	output:
		'data/xarray/xr_{float}.nc'
	script:
		'src/convert_mat_to_xr.py'

rule convert_metdata:
    input:
        'data/metdata/float_cfs_hourly.mat'
    output:
        'data/metdata/float_cfs_hourly_2016.nc',
        'data/metdata/float_cfs_hourly_2017.nc',
        'data/metdata/float_cfs_hourly.nc'
    script:
        'src/convert_metdata.py'

rule convert_model_fields:
    input:
        'data/CFS/CFSv2_wind_rh_t_p_2016_2018.mat',
        'data/CFS/land.nc'
    output:
        'data/CFS/CFSv2_wind_rh_t_p_2016_2018.nc',
    script:
        'src/convert_model_fields.py'

rule resample:
	input:
		'data/xarray/xr_{float}.nc'
	output:
		'data/resampled/resample_{float}_{resample_period}.nc'
	script:
		'src/resample.py'

rule filter:
	input:
		'data/resampled/resample_{float}_{resample_period}.nc'
	output:
		'data/filtered/filt_{float}_{resample_period}_{filter_period}Tf.nc'
	script:
		'src/filter.py'

rule summary_plts:
    input:
        'data/filtered/filt_{float}_{resample_period}_{filter_period}Tf.nc'
    output:
        'figures/summary/summary_{float}_{resample_period}_{filter_period}Tf.pdf'
    script:
        'src/summary_plots.py'

rule compare_ni:
    input:
        'data/filtered/filt_{float}_{resample_period}_{filter_period}Tf.nc'
    output:
        'figures/ni_currents/compare_ni_{float}_{resample_period}_{filter_period}Tf.pdf',
        'figures/ni_currents/compare_ni_hke_{float}_{resample_period}_{filter_period}Tf.pdf'
    script:
        'src/compare_ni.py'

rule compute_ml_avg:
	input:
		'data/filtered/filt_{float}_{resample_period}_{filter_period}Tf.nc'
	output:
		'data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc'
	script:
		'src/compute_ml_avg.py'

# rule combine_ml_avg:
#     input:
#         expand('data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc',
#                float=FLOATS,resample_period=resample_period,
#                filter_period=filter_period)
#     output:
#         'data/ml/mlall_{resample_period}_{filter_period}Tf.nc'
#     benchmark:
#         repeat('benchmark/bench_{resample_period}_{filter_period}Tf.txt',3)
#     script:
#         'src/combine_ml_avg.py'

rule plt_ml_timeseries:
    input:
        'data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc',
        'data/metdata/float_cfs_hourly.nc'
    output:
        'figures/ml_timeseries/ml_timeseries_{float}__{resample_period}_{filter_period}Tf.pdf'
    script:
        'src/ml_timeseries.py'

rule make_weather_maps:
    input:
        'data/CFS/CFSv2_wind_rh_t_p_2016_2018.nc'
    output:
        'figures/weather_maps/weather_{date}.pdf'
    script:
        'src/plot_weather.py'

# rule nio_maps:
#     input:
#         'data/ml/ml_{float}_{filter_period}Tf.nc'
#     output:
#         'figures/nio_maps_{float}_{filter_period}Tf.pdf',
#         'figures/nio_map.pdf'
#     script:
#         'src/nio_maps.py'

rule make_map_all:
    input:
        expand('data/xarray/xr_{float}_grid.nc',float=FLOATS)
    output:
        'figures/float_traj.pdf'
    script:
        'src/make_maps.py'
