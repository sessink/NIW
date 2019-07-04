configfile: 'config.yaml'
workdir: config['path']
FLOATS = config['floats']
filter_period = str(config['filter_period'])
resample_period = config['resample_period']

rule all:
    input:
        expand('data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/summary/summary_{float}_{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period),
        expand('figures/summary/compare_ni_{float}_{resample_period}_{filter_period}Tf.pdf',
               float=FLOATS, filter_period=filter_period,
               resample_period=resample_period)
        # expand('figures/nio_maps_{float}_{filter_period}Tf.pdf',float=FLOATS,filter_period=filter_period),
        # 'figures/float_traj.pdf',

rule convert_mat_files:
	input:
		'data/NIWmatdata/{float}_grid.mat'
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

rule summary_plots:
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
        'figures/summary/compare_ni_{float}_{resample_period}_{filter_period}Tf.pdf'
    script:
        'src/compare_ni.py'

rule compute_ml_averages:
	input:
		'data/filtered/filt_{float}_{resample_period}_{filter_period}Tf.nc'
	output:
		'data/ml/ml_{float}_{resample_period}_{filter_period}Tf.nc'
	script:
		'src/compute_ml_averages.py'

rule ml_timeseries:
    input:
        'data/ml/ml_{float}_{filter_period}Tf.nc',
        'data/metdata/float_cfs_hourly.nc'
    output:
        'figures/ml_timeseries_{float}_{filter_period}Tf.pdf'
    script:
        'src/ml_timeseries.py'

# rule nio_maps:
#     input:
#         'data/ml/ml_{float}_{filter_period}Tf.nc'
#     output:
#         'figures/nio_maps_{float}_{filter_period}Tf.pdf',
#         'figures/nio_map.pdf'
#     script:
#         'src/nio_maps.py'
#
# rule make_map_all:
#     input:
#         expand('data/xarray/xr_{float}_grid.nc',float=FLOATS)
#     output:
#         'figures/float_traj.pdf'
#     script:
#         'src/make_maps.py'
