# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:16:20 2026

@author: alleh
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta 

import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from colormaps import cmap_abs, cmap_windspeed, cmap_wvmr, cmap_purplebrown40, cmap_adv_div_brown_green #cmap_bluered16, cmap_adv_seq_mhue_inferno20
from basic_plot_funcions import savefig, grid_edges

def plot_ppl_wvmr_10s(start, end, hmax, vmax, fig_size, ticks):

    vmin=0
    #vmax=15
    fontsize = 24

    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\pplMR_10s.nc')
    data = data.rename({'range': 'height'})
    mask = ((data.time >= start) & (data.time <= end))
    data = data.sel(time=mask)
    
    end=end-np.timedelta64(1, "s")
    # Extend x (time array) and y (height array) to include first and last edges
    param = data['MR'].to_numpy().transpose()
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    # par_cmap = cmap_ppls_wvmr()#cmap_wvmr()# 'Blues'
    # par_cmap.set_under("black")
    # par_cmap.set_over("white")
    par_cmap = cmap_wvmr() #'viridis_r' #cmap_purplebrown40()#'Blues'
    param_label = r'wvmr (g kg$^{-1}$)'# 'water vapor mixing ratio (g/kg)'
    param = np.ma.masked_invalid(param)  # Ensures NaNs are masked

    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='auto', vmin=vmin, vmax=vmax, cmap=par_cmap) 
    # pcm = ax.pcolormesh(t, h, param, shading='auto', cmap=par_cmap, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    cbar.set_label(param_label, size=fontsize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
    
    ax.set_xlim([start, end])
    if ticks == 'fullday':
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    else:
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 20, 40], interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.tick_params(direction='out', labelsize=fontsize)
    ax.set_ylim([0,hmax])

    # Add Title and Labels
    ax.set_title(f"PPLS-10s: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    
    # Background color for NaN values
    ax.set_facecolor([0.9, 0.9, 0.9])
    fig.patch.set_facecolor([1, 1, 1])
    fig.patch.set_alpha(1.0)
    ax.set_facecolor([0.8, 0.8, 0.8])
    fig.tight_layout()
    plt.show()

    return '10s', fig

def plot_ppl_wvmr_20m(start, end, hmax,vmax, fig_size, ticks):

    vmin=0
    # vmax=13
    fontsize=24
    start2 = start - np.timedelta64(1, 'm')

    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\pplMR_1200s.nc')
    data = data.rename({'range': 'height'})
    mask = ((data.time >= start2) & (data.time <= end))
    data = data.sel(time=mask)
    end=end-np.timedelta64(1, "s")
    # Extend x (time array) and y (height array) to include first and last edges
    param = data['MR'].to_numpy().transpose()
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    par_cmap = cmap_wvmr()#cmap_wvmr()# 'Blues'
    # par_cmap.set_under("black")
    # par_cmap.set_over("white")

    param_label = r'wvmr (g kg$^{-1}$)'# 'water vapor mixing ratio (g/kg)'
    param = np.ma.masked_invalid(param)  # Ensures NaNs are masked

    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='flat', vmin=vmin, vmax=vmax, cmap=par_cmap) 
    
    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    cbar.set_label(param_label, size=fontsize)
    
    ax.set_xlim([start, end])
    if ticks == 'fullday':
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    else:
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 20, 40], interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(direction='out', labelsize=fontsize)
    ax.set_ylim([0,hmax])
    
    # Add Title and Labels
    ax.set_title(f"PPLS-20min: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    ax.set_facecolor([0.8, 0.8, 0.8])
    
    plt.tight_layout()
    plt.show()

    return '20m', fig

version = 0

# fig_size = [18, 6] #[15, 7] #figsize
hmax = 5
vmax = 15
date  = datetime(2024, 8, 24)

# ticks = "fullday"
# fig_size = [18, 6] #[21,7] # [18, 6]
# start = np.datetime64(date)
# end   = np.datetime64(date) + np.timedelta64(1, 'D')
# end  = np.datetime64(end) - np.timedelta64(1,'ns')

ticks = "timewindow"
fig_size = [15, 7] # [15, 5]
# start = np.datetime64(date) + np.timedelta64((60*10+00), 'm')
# end   = np.datetime64(date) + np.timedelta64((60*15+00), 'm')
start = np.datetime64(date) + np.timedelta64((60*20+40), 'm') #for 24th
end   = np.datetime64(date) + np.timedelta64((60*21+49), 'm') 
start = np.datetime64(date) + np.timedelta64((60*18+35), 'm') #for 23th
end   = np.datetime64(date) + np.timedelta64((60*20+11), 'm')


time, fig = plot_ppl_wvmr_10s(start, end, hmax, vmax, fig_size, ticks)
# time, fig = plot_ppl_wvmr_20m(start, end, hmax, vmax, fig_size, ticks)

# filename = f"PPL{time}_wvmr_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
# # folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "shorttime")
# folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", f"PPL_{time}_wvmr", "Raw")
# savefig(fig, folderpath, filename)

################
# ticks = "fullday"
# fig_size = [21,7] # [18, 6]
# date_beg, date_end = datetime(2024, 8, 15), datetime(2024, 9, 8)
# dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]

# for date in dates: 
#     start =  np.datetime64(date) 
#     end   =  np.datetime64(date) + np.timedelta64(1, 'D')
    
#     time, fig = plot_ppl_wvmr_10s(start, end, hmax, vmax, fig_size, ticks)
#     # time, fig = plot_ppl_wvmr_20m(start, end, hmax, vmax, fig_size, ticks)

#     filename = f"PPLS{time}_wvmr_{hmax}km_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
#     folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", f"PPL_{time}_wvmr", "newcolor_raw")
#     savefig(fig, folderpath, filename)