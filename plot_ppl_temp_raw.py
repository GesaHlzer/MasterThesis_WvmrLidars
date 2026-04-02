# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:19:29 2026

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

from colormaps import cmap_bluered40
from basic_plot_funcions import savefig, grid_edges

def plot_ppl_temp_10s(start, end, hmax, fig_size, ticks):

    fontsize = 24

    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\pplT_10s.nc')
    data = data.rename({'range': 'height'})
    mask = ((data.time >= start) & (data.time <= end))
    data = data.sel(time=mask)
    
    param = data['T'].to_numpy().transpose()
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    param = data.T.to_numpy().transpose()
    par_cmap = cmap_bluered40() #  mlp.colormaps['YlGn']
    param_label = 'temperature (K)'
    p_min, p_max = 250, 310 # data_ppl.T.min(), 300

    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='flat', vmin=p_min, vmax=p_max, cmap=par_cmap) 
    
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
    ax.set_title(f"PPL (10s): {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    
    fig.tight_layout()
    plt.show()

    return '10s', fig

def plot_ppl_temp_20m(start, end, hmax, fig_size, ticks):

    vmin=2
    vmax=13
    fontsize=24

    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\pplT_1200s.nc')
    data = data.rename({'range': 'height'})
    mask = ((data.time >= start) & (data.time <= end))
    data = data.sel(time=mask)
    
    # Extend x (time array) and y (height array) to include first and last edges
    param = data['T'].to_numpy().transpose()
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    param = data.T.to_numpy().transpose()
    par_cmap = cmap_bluered40() #  mlp.colormaps['YlGn']
    param_label = 'temperature (K)'
    p_min, p_max = 250, 310 # data_ppl.T.min(), 300

    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='flat', vmin=p_min, vmax=p_max, cmap=par_cmap) 
        
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
    ax.set_title(f"PPL (20m): {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    
    fig.tight_layout()
    plt.show()

    return '20m', fig

version = 1

fig_size = [18, 6] #[15, 7] #figsize
hmax = 6


# # date_beg, date_end = datetime(2024, 8, 7), datetime(2024, 8, 7)
# # dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]
# # for date in dates:
# date  = datetime(2024, 8, 28)

# # ticks = "fullday"
# # fig_size = [21,7] # [18, 6]
# # start =  np.datetime64(date)
# # end   =  np.datetime64(date) + np.timedelta64(1, 'D')

# ticks = "timewindow"
# fig_size = [15, 7] # [15, 5]
# start = np.datetime64(date) + np.timedelta64((60*18+30), 'm')
# end   = np.datetime64(date) + np.timedelta64((60*20+30), 'm')
# # start = np.datetime64(date) + np.timedelta64((60*20+50), 'm') #for 28th
# # end   = np.datetime64(date) + np.timedelta64((60*22+30), 'm') #for 28th
# # start = np.datetime64(date) + np.timedelta64((60*12+45), 'm') #for 28th
# # end   = np.datetime64(date) + np.timedelta64((60*14+30), 'm') #for 28th

# time, fig = plot_ppl_temp_10s(start, end, hmax, fig_size, ticks)
# # time, fig = plot_ppl_temp_20m(start, end, hmax, fig_size, ticks)

# filename = f"PPL{time}_temp_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
# folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "shorttime")
# #savefig(fig, folderpath, filename)

################
ticks = "fullday"
fig_size = [21,7] # [18, 6]
date_beg, date_end = datetime(2024, 8, 22), datetime(2024, 9, 8)
dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]

for date in dates: 
    start =  np.datetime64(date)
    end   =  np.datetime64(date) + np.timedelta64(1, 'D')
    
    time, fig = plot_ppl_temp_10s(start, end, hmax, fig_size, ticks)
    # time, fig = plot_ppl_temp_20m(start, end, hmax, fig_size, ticks)
    
    filename = f"PPL{time}_temp_{hmax}km_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", f"PPL_{time}_temp", "Raw")
    savefig(fig, folderpath, filename)
        