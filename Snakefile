configfile: 'config.yaml'
workdir: config['path']

rule all:
    input:
        expand('data/xarray/xr_{float}_grid.nc',float=config['floats']),
        'figures/float_traj.pdf',
        #expand('figures/float_traj_{group}.pdf',group=config['groups'])
        
rule convert_mat_files:
	input:
		'data/NIWmatdata/{float}_grid.mat'
	output:
		'data/xarray/xr_{float}_grid.nc'
	script:
		'src/convert_mat_to_xr.py'

rule make_map_all:
    input:
        expand('data/xarray/xr_{float}_grid.nc',float=config['floats'])
    output:
        'figures/float_traj.pdf'
        #expand('figures/float_traj_{group}.pdf',group=config['groups'])
    script:
        'src/make_maps.py'