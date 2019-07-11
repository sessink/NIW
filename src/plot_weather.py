# Standard Library
from datetime import datetime

# Scientific Computing
import numpy as np
import pandas as pd
import xarray as xr

# Plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[10, 8])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)


# %% FUNCTIONS
def plotMap():
    #Set the projection information
    proj = ccrs.PlateCarree()
    #Create a figure with an axes object on which we will plot. Pass the projection to that axes.
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj))

    #Zoom in
    img_extent = [140, 162, 34, 50]
    ax.set_extent(img_extent, crs=proj)

    #Add map features
    # ax.add_feature(cfeature.LAND, facecolor='0.9'
    # )  #Grayscale colors can be set using 0 (black) to 1 (white)

    land_50m = cfeature.NaturalEarthFeature('physical',
                                            'land',
                                            '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
    ax.add_feature(land_50m)
    ax.add_feature(
        cfeature.LAKES,
        alpha=0.9)  #Alpha sets transparency (0 is transparent, 1 is solid)
    ax.add_feature(cfeature.BORDERS, zorder=10)
    # ax.add_feature(cfeature.COASTLINE, zorder=10)
    gshhs = cfeature.GSHHSFeature(scale='i', levels=None)
    ax.add_feature(gshhs)

    #We can use additional features from Natural Earth (http://www.naturalearthdata.com/features/)
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray', zorder=10)

    #Add lat/lon gridlines every 20° to the map
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=0.1,
                      alpha=1,
                      linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.text(142.2, 43.5, 'Hokkaido', transform=proj)
    ax.text(140.4, 39.7, 'Honshu', transform=proj)

    return fig, ax


def plot_weather(date, dss):
    fig, ax = plotMap()

    preslevels = np.linspace(950, 1400)
    prescontour = ax.contour(dss.lon,
                             dss.lat,
                             dss.pres.T,
                             colors='k',
                             levels=preslevels,
                             linewidths=1,
                             zorder=11,
                             transform=ccrs.PlateCarree())
    plt.clabel(prescontour, preslevels, inline=True, fmt='%1i', fontsize=12)

    speedlevels = np.linspace(10, 36,num=10)
    sp = ax.contourf(dss.lon,
                     dss.lat,
                     dss.wspeed.T,
                     levels=speedlevels,
                     cmap=plt.cm.YlOrRd,
                     zorder=2,
                     transform=ccrs.PlateCarree())

    x, y = np.meshgrid(dss.lon, dss.lat, indexing='ij')
    wslice = slice(0, -1, 10)
    ax.barbs(x[wslice, wslice],
             y[wslice, wslice],
             dss.u10.values[wslice, wslice],
             dss.v10.values[wslice, wslice], 
             transform=ccrs.PlateCarree(),
             zorder=20)

    plt.colorbar(sp,
                 ax=ax,
                 ticks=np.arange(20, 35, 2),
                 label=r'Wind speed m$s^{-1}$',
                 orientation='horizontal',
                 aspect=40,
                 shrink=0.87,
                 pad=0.04)
    plt.title(date)
    plt.savefig('figures/weather_maps/weather_%s.pdf' %
                (date))
    plt.close()


def weather_wrapper(infile, outfile):
    ds = xr.open_dataset(str(infile))
    # dates = pd.date_range(start='1/1/2018', end='1/08/2018', freq='6H')

    date= str(outfile).split(sep='weather_')[-1].split('.')[0]
    dss = ds.sel(time=pd.to_datetime(date,format="%d%m%y_%Hh"))
    plot_weather(date, dss)


# %% MAIN

weather_wrapper(snakemake.input, snakemake.output)

# %% testing
# infile = 'data/CFS/CFSv2_wind_rh_t_p_2016_2018.nc'
# outfile = 'figures/weather_maps/weather_061117_18h.pdf'
# ds = xr.open_dataset( str(infile) )
# date= str(outfile).split(sep='weather_')[-1].split('.')[0]
# dss = ds.sel(time=pd.to_datetime(date,format="%d%m%y_%Hh"))
#
# x, y = np.meshgrid(dss.lon, dss.lat, indexing='ij')
#
# speedlevels = np.linspace(10, 36)
#
# plt.contourf(dss.lon,
#                  dss.lat,
#                  dss.wspeed.T,
#                  levels=speedlevels,
#                  cmap=plt.cm.YlOrRd,
#                  zorder=2)
# plt.colorbar()
                 # transform=ccrs.PlateCarree())

# plot_weather(date, dss)

# # %% rotation
#
# sns.set(style='ticks', context='paper')
# mpl.rc('figure', dpi=100, figsize=[10, 8])
# mpl.rc('savefig', dpi=500, bbox='tight')
# mpl.rc('legend', frameon=False)
#
# ds['ang'] = np.degrees(np.arctan2(ds.v10, ds.u10))
#
#
# def compare_ang(a1, a2):
#     return 180 - np.abs(np.abs(a1 - a2) - 180)
#
#
# date = datetime(2017, 10, 29, 18, 0, 0)
# dss = ds.sel(time=date)
# dss2 = ds.sel(time=datetime(2017, 10, 29, 12, 0, 0))
# dss['dang'] = compare_ang(dss.ang, dss2.ang)
#
# fig, ax = plotMap()
#
# preslevels = np.linspace(950, 1400)
# prescontour = ax.contour(dss.lon,
#                          dss.lat,
#                          dss.pres.T,
#                          colors='k',
#                          levels=preslevels,
#                          linewidths=1,
#                          zorder=11,
#                          transform=ccrs.PlateCarree())
# plt.clabel(prescontour, preslevels, inline=True, fmt='%1i', fontsize=12)
#
# rotlevels = np.linspace(0, 180)
# sp = ax.contourf(dss.lon,
#                  dss.lat,
#                  dss.dang.T,
#                  levels=rotlevels,
#                  cmap='RdBu_r',
#                  zorder=2,
#                  transform=ccrs.PlateCarree())
#
# x, y = np.meshgrid(dss.lon, dss.lat, indexing='ij')
# wslice = slice(0, -1, 10)
# ax.barbs(x[wslice, wslice],
#          y[wslice, wslice],
#          dss.u10.values[wslice, wslice],
#          dss.v10.values[wslice, wslice],
#          transform=ccrs.PlateCarree(),
#          zorder=20)
#
# plt.colorbar(sp,
#              ax=ax,
#              ticks=np.arange(0, 180, 10),
#              label=r'Wind speed m$s^{-1}$',
#              orientation='horizontal',
#              aspect=40,
#              shrink=0.87,
#              pad=0.04)
# plt.title(date.strftime("%b %d %Y, %H:%M:%S"))
# plt.savefig('figures/weather_maps/windrot_%s.pdf' %
#             (date.strftime("%b%d%Y%H:%M")))
# plt.show()
