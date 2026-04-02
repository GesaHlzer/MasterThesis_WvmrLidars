# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:20:27 2026

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

from colormaps import cmap_abs #cmap_bluered16, cmap_adv_seq_mhue_inferno20
from basic_plot_funcions import savefig, grid_edges

def read_DA10(date): #unused if completed dataset used
    ncdir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'DA10', 'abs') 
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

def plot_da10_abs(date, fig_size, hmax): # 'wvmr' 'abs'
    
    fontsize = 22
    # Read data 
    data = xr.open_dataset(r'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_abs.nc')
    data = data.rename({'range': 'height'})
    #data_dial = read_DA10(date_beg, date_end, par)
    
    start =  np.datetime64(date)
    end   =  np.datetime64(date) + np.timedelta64(1, 'D')
    mask = ((data.time >= start) & (data.time <= end))
    data = data.sel(time=mask)
    
    # Extend x (time array) and y (height array) to include first and last edges    
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    param = data.beta_att.values.transpose()
    par_cmap = cmap_abs() #cmap_abs() #'Grays' # 'inferno_r' #cmap_backscatter() # 'twilight_shifted'
    norm = mcolors.LogNorm(vmin=np.nanmin(data["beta_att"].where(data["beta_att"] > 0)), vmax=np.nanmax(data["beta_att"]))
    # norm = mcolors.LogNorm(vmin=np.nanmin(param[param > 0]),vmax=np.nanmax(param))
    
    param_label ='att. vol. backscatter coef. (m$^{-1}$sr$^{-1}$)'
    ylim =[0, 18]
    
    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, param, shading='flat', cmap=par_cmap, norm=norm)
    #, norm=cbar_norm  # vmax=(max(40, param.max())),vmin=0, vmax=0.0002, Adjust to match dimensions
    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither', norm='log')
    cbar.ax.tick_params(direction='out', labelsize=fontsize-1, size=10)
    cbar.set_label(param_label, size=fontsize)
    #cbar.ax.set_yscale("log")
    ax.set_xlim([mdates.date2num(date - timedelta(minutes=3)), mdates.date2num(date + timedelta(days=1))])
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.tick_params(direction='out', labelsize=fontsize)
    
    # Add Title and Labels
    title_text = f"DA10 backscatter: {date.strftime('%Y-%m-%d')}" # .strftime('%H UTC %d-%m-%Y')
    ax.set_title(title_text, fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    
    ax.set_ylim([0,hmax])
                
    fig.tight_layout()
    plt.show()
    
    return fig

fig_size = [18,6] #[18,6] # [18, 6]

date  = datetime(2024, 8, 28)
hmax = 18
# Create date list

date_beg, date_end = datetime(2024, 8, 22), datetime(2024, 9, 8)
dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]
for date in dates:

    fig = plot_da10_abs(date, fig_size, hmax)
    filename = f"DA10_abs_{date.strftime('%Y-%m-%d')}.png"
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "DA10_abs")
    savefig(fig, folderpath, filename)

print("done")
