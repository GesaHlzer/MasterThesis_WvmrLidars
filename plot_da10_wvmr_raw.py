# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:12:20 2026

@author: alleh
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from colormaps import cmap_wvmr
from basic_plot_funcions import savefig, grid_edges


def read_DA10(date): #unused if completed dataset used
    ncdir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'DA10', 'wvmr') 
    date2 = date + timedelta(days=1)

    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    file_path = os.path.join(ncdir, year, month, day)
    
    filenames = [f for f in os.listdir(file_path) if f.endswith('.nc')]
    ds =  [xr.open_dataset(f'{file_path}\{f}') for f in filenames]
    
    try:
        year2 = date2.strftime('%Y')
        month2 = date2.strftime('%m')
        day2 = date2.strftime('%d')
        file_path2 = os.path.join(ncdir, year2, month2, day2)
        
        filenames2 = [f for f in os.listdir(file_path2) if f.endswith('.nc')]
        ds.append(xr.open_dataset(f'{file_path2}\{filenames2[0]}'))
        files_combined = xr.concat(ds, dim='time')
    except:
        files_combined = xr.concat(ds, dim='time')
    
    files_combined.sel(time=slice(str(date), str(date + timedelta(days=1))))
    
    return files_combined  

def plot_da10_wvmr(start, end, vmax, fig_size, ticks):
    
    vmin=0
    #vmax=18
    fontsize = 24
    hmax = 4
    
    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc')
    #data['wvmr'] = data['wvmr'].where(data['height'] <= data['water_vapor_max_range'])
    mask = ((data.time >= start) & (data.time <= end))
    data = data.sel(time=mask)
    
    # Extend x (time array) and y (height array) to include first and last edges
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    param = data['water_vapor'].to_numpy().transpose()
    par_cmap = cmap_wvmr() #'Blues'
    param_label = r'wvmr (g kg$^{-1}$)'# 'water vapor mixing ratio (g/kg)'
    param = np.ma.masked_invalid(param)  # Ensures NaNs are masked
    
    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='flat', vmin=vmin, vmax=vmax, cmap=par_cmap) 
    
    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    cbar.set_label(param_label, size=fontsize)
     
    mr_valid = data.water_vapor_max_range.values / 1000
    mr_valid = h[np.abs(h[:, None] - mr_valid).argmin(axis=0)]
    mr_valid = np.concatenate(([mr_valid[0],], mr_valid))
    ax.plot(t, mr_valid, color='red')
        
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
    ax.set_title(f"DA10: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    ax.set_facecolor([0.8, 0.8, 0.8])
    fig.tight_layout()
    plt.show()
    
    return fig

version = 0
vmax = 15 
date  = datetime(2024, 8, 23)

# ticks = "fullday"
# fig_size = [21,7] # [18, 6]
# start =  np.datetime64(date)
# end   =  np.datetime64(date) + np.timedelta64(1, 'D')
# end   = np.datetime64(end) - np.timedelta64(1,'ns')

ticks = "timewindow"
fig_size = [15, 7] # [15, 5]
# start = np.datetime64(date) + np.timedelta64((60*18+30), 'm')
# end   = np.datetime64(date) + np.timedelta64((60*22+00), 'm')
# # start = np.datetime64(date) + np.timedelta64((60*20+50), 'm') #for 28th
# # end   = np.datetime64(date) + np.timedelta64((60*22+30), 'm') #for 28th
start = np.datetime64(date) + np.timedelta64((60*8+5), 'm') #for 28th
end   = np.datetime64(date) + np.timedelta64((60*16+30), 'm') #for 28th

fig = plot_da10_wvmr(start, end, vmax, fig_size, ticks)
# filename = f"DA10_wvmr_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
# # folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "shorttime")
# folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "DA10_wvmr", "NewColor_raw")
# savefig(fig, folderpath, filename)

################
# ticks = "fullday"
# fig_size = [21,7] # [18, 6]
# date_beg, date_end = datetime(2024, 8, 23), datetime(2024, 9, 8)
# dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]

# for date in dates: 
#     start =  np.datetime64(date)
#     end   =  np.datetime64(date) + np.timedelta64(1, 'D') - np.timedelta64(1,'ns')
    
#     fig = plot_da10_wvmr(start, end, vmax, fig_size, ticks)
#     filename = f"DA10_wvmr_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}_v{version}.png"
#     folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "DA10_wvmr", "NewColor_raw")
#     savefig(fig, folderpath, filename)


# print("done")