configfile: 'config.yaml'
workdir: config['path']

FLOATS = config['floats']

rule all:
    input:
        expand('data/ml/ml_{float}_filt_5Tf.nc', float=FLOATS),
        # expand('figures/vel_{float}_filt.pdf',float=FLOATS),
        # expand('figures/dens_{float}.pdf',float=FLOATS),
        # expand('figures/shear_{float}.pdf',float=FLOATS),
        # expand('figures/vel_{float}_filt.pdf',float=FLOATS),
        # expand('figures/shear_{float}_filt.pdf',float=FLOATS),
        # expand('figures/epschi_{float}.pdf',float=FLOATS),
        # 'figures/float_traj.pdf',

rule convert_mat_files:
	input:
		'data/NIWmatdata/{float}_grid.mat'
	output:
		'data/xarray/xr_{float}_grid.nc'
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

rule resample_and_filter:
	input:
		'data/xarray/xr_{float}_grid.nc'
	output:
		'data/filtered/xr_{float}_grid_filt_5Tf.nc'
	script:
		'src/resample_filter.py'

rule ml_averages:
	input:
		'data/filtered/xr_{float}_grid_filt_5Tf.nc'
	output:
		'data/ml/ml_{float}_filt_5Tf.nc'
	script:
		'src/compute_ml_averages.py'

rule ml_timeseries:
    input:
        'data/ml/ml_{float}_filt_5Tf.nc',
        'data/metdata/float_cfs_hourly.nc'
    output:
        'figures/ml_timeseries_{float}.pdf'
    script:
        'src/ml_timeseries.py'

rule nio_maps:
    input:
        'data/filtered/ml_{float}_filt_5Tf.nc'
    output:
        'figures/nio_maps_{float}.pdf'
        'figures/nio_map.pdf'
    script:
        'src/nio_maps.py'

# rule make_map_all:
#     input:
#         expand('data/xarray/xr_{float}_grid.nc',float=FLOATS)
#     output:
#         'figures/float_traj.pdf'
#     script:
#         'src/make_maps.py'
# #
# rule plot_vel:
#     input:
#         'data/xarray/xr_{float}_grid_filt_5Tf.nc'
#     output:
#         'figures/vel_{float}_filt.pdf',
#     script:
#         'src/plot_vel.py'
#
# rule plot_dens:
#     input:
#         'data/xarray/xr_{float}_grid.nc'
#     output:
#         'figures/dens_{float}.pdf',
#     script:
#         'src/plot_dens_n2.py'
#
# rule vel_shear:
#     input:
#         'data/xarray/xr_{float}_grid.nc'
#     output:
#         'figures/vel_{float}.pdf',
#         'figures/shear_{float}.pdf'
#     script:
#         'src/plot_velshear.py'

# rule turb:
#     input:
#         'data/xarray/xr_{float}_grid.nc'
#     output:
#         'figures/epschi_{float}.pdf',
#     script:
#         'src/plot_turb.py'
