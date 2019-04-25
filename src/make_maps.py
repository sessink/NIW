import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import seaborn as sns
from itertools import count

# # mapping
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# #mpl.rcParams.keys()
#project_path = '/Users/sebastian/Dropbox (MIT)/postdoc/'
sns.set(style='ticks',context='paper')
mpl.rc('figure', dpi=120, figsize=[10,5])
mpl.rc('savefig',dpi=500,bbox='tight')
mpl.rc('legend',frameon=False)

f,ax=plt.subplots(1,1)
ax = plt.axes(projection=ccrs.PlateCarree())
img_extent = [140, 162, 34, 45]
ax.set_extent(img_extent, crs=ccrs.PlateCarree())

for i in snakemake.input:
    file =  str(i) #os.path.join(project_path,i)
    c = count()
    dat = xr.open_dataset(str(file))
    ax.plot(dat.lon,dat.lat,lw=2,label=dat.floatid)
ax.set(ylabel='Latitude',
       xlabel='Longitude')
# ax.set_title(f'Deployment {letter.upper()}')
plt.legend(bbox_to_anchor=(1.08, 1),title='Floats')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.1, alpha=1, linestyle='-')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cmap = plt.get_cmap('Blues')
norm = plt.Normalize(0, 10000)
for letter, level in [
                      ('L', 0),
                      ('K', 200),
                      ('J', 1000),
                      ('I', 2000),
                      ('H', 3000),
                      ('G', 4000),
                      ('F', 5000),
                      ('E', 6000),
                      ('D', 7000),
                      ('C', 8000),
                      ('B', 9000),
                      ('A', 10000)]:
    bathym = cfeature.NaturalEarthFeature(name='bathymetry_{}_{}'.format(letter, level),
                                 scale='10m', category='physical')
    ax.add_feature(bathym, facecolor=cmap(norm(level)), edgecolor='face')

ax.add_feature(cfeature.LAND)
gshhs = cfeature.GSHHSFeature(scale='i', levels=None)
ax.add_feature(gshhs)
ax.text(142.2, 43.5, 'Hokkaido', transform=ccrs.PlateCarree())
ax.text(140.4, 39.7, 'Honshu', transform=ccrs.PlateCarree())
# ax.gridlines()
output_path = str(snakemake.output) #os.path.join(project_path,str(snakemake.output))
plt.savefig(output_path)
plt.close()