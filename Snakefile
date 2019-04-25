configfile: 'config.yaml'
workdir: config['path']

FLOATS = config['floats']

rule all:
    input:
        expand('figures/vel_{float}.pdf',float=FLOATS),
        expand('figures/shear_{float}.pdf',float=FLOATS),
        'figures/float_traj.pdf',
        # 'figures/shear_{float}.pdf'
        #expand('figures/float_traj_{group}.pdf',group=config['groups'])
        
rule convert_mat_files:
	input:
		'data/NIWmatdata/{float}_grid.mat'
	output:
		'data/xarray/xr_{float}_grid.nc'
	script:
		'src/convert_mat_to_xr.py'

# rule make_map_all:
#     input:
#         expand('data/xarray/xr_{float}_grid.nc',float=FLOATS)
#     output:
#         'figures/float_traj.pdf'
#         #expand('figures/float_traj_{group}.pdf',group=config['groups'])
#     script:
#         'src/make_maps.py'

rule vel_shear:
    input:
        'data/xarray/xr_{float}_grid.nc'
    output:
        'figures/vel_{float}.pdf',
        'figures/shear_{float}.pdf'
        #expand('figures/float_traj_{group}.pdf',group=config['groups'])
    script:
        'src/plot_vel_shear.py'