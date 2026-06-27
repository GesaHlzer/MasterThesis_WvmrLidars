# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:22:29 2026

@author: alleh
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from basic_plot_funcions import haversine
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from colormaps import cmap_wvmr, cmap_windspeed

# time period
start = np.datetime64("2024-08-23") #- np.timedelta64(1, 'm')
end   = np.datetime64("2024-08-25") #- np.timedelta64(1, 's')
begt  = start #+ np.timedelta64(1, 'm')
endt  = end - np.timedelta64(1, 'ns')

Fontsize = 26
Ticksize = 10

data_aws =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\stationsdata.nc")


def plot_aws_temp(ax, data_aws, clim=[8, 38], band_thickness=0.12):
    """
    Plot station temperature as horizontal colored bands, 
    sorted by altitude, each band ~band_thickness meters thick.
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws['height']  = ds_aws['altitude'] - 577
    
    # Sort stations by altitude
    alt_order = ds_aws['height'].argsort().values
    ds_sorted = ds_aws.isel(station=alt_order)

    heights   = ds_sorted['height'].values/1000
    h0 = heights[0]
    heights[0] = heights[0] -0.045
    time      = ds_sorted['time'].values
    temp      = ds_sorted['temp'].values  # shape: (station, time)
    shortcuts = ds_sorted['shortcut'].values
    stations = ds_sorted['station'].values

    cmap = plt.get_cmap('RdYlBu') #turbo #jet
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    time_num = mdates.date2num(time)
    # grid edges for time
    dt = (time_num[1] - time_num[0]) / 2
    t_edges = np.concatenate([[time_num[0] - dt], time_num + dt])
    
    for i, (h, shortcut) in enumerate(zip(heights, shortcuts)):
        
            y0 = h - band_thickness / 2
            y1 = h + band_thickness / 2
    
            for j in range(len(time_num)):
                color = cmap(norm(temp[i, j]))
                ax.fill_betweenx([y0, y1], t_edges[j], t_edges[j+1], color=color)
            
            if i==0: station_label = f' {shortcut} ({int(h0*1000)} m AGL)'
            else: station_label = f' {shortcut} ({int(h*1000)} m AGL)'
            # Station label on the right
            ax.text(time_num[1] + dt*2, h, station_label,
                    va='center', fontsize=15)

    #pcm = ax.pcolormesh(t_edges, y_edges, temp, cmap=cmap, norm=norm, shading='flat')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('temp. (°C)', fontsize=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                    
    # # Station labels on right side
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(heights)
    # ax2.set_yticklabels([f'{s} ({int(h)}m)' for s, h in zip(shortcuts, heights)],
    #                     fontsize=Fontsize - 4)
    # ax2.tick_params(direction='out', length=0)  # no tick marks, just labels

    ax.set_title('AWS temperature', fontsize=Fontsize)

    return ax


def plot_aws_pres(ax, data_aws, clim=[700, 1100], band_thickness=0.12):
    """
    Plot station temperature as horizontal colored bands, 
    sorted by altitude, each band ~band_thickness meters thick.
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws['height']  = ds_aws['altitude'] - 577
    
    # Sort stations by altitude
    alt_order = ds_aws['height'].argsort().values
    ds_sorted = ds_aws.isel(station=alt_order)

    heights   = ds_sorted['height'].values/1000
    h0 = heights[0]
    heights[0] = heights[0] -0.045
    time      = ds_sorted['time'].values
    pres      = ds_sorted['p_estimated'].values  # shape: (station, time)
    shortcuts = ds_sorted['shortcut'].values
    stations = ds_sorted['station'].values

    cmap = plt.get_cmap('RdYlGn') #turbo #jet')
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    time_num = mdates.date2num(time)
    # grid edges for time
    dt = (time_num[1] - time_num[0]) / 2
    t_edges = np.concatenate([[time_num[0] - dt], time_num + dt])
    
    for i, (h, shortcut) in enumerate(zip(heights, shortcuts)):
        
            y0 = h - band_thickness / 2
            y1 = h + band_thickness / 2
    
            for j in range(len(time_num)):
                color = cmap(norm(pres[i, j]))
                ax.fill_betweenx([y0, y1], t_edges[j], t_edges[j+1], color=color)
            
            if i==0: station_label = f' {shortcut} ({int(h0*1000)} m AGL)'
            else: station_label = f' {shortcut} ({int(h*1000)} m AGL)'
            # Station label on the right
            ax.text(time_num[1] + dt*2, h, station_label,
                    va='center', fontsize=15)

    #pcm = ax.pcolormesh(t_edges, y_edges, temp, cmap=cmap, norm=norm, shading='flat')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('pres. (hp)', fontsize=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
                                    
    # # Station labels on right side
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(heights)
    # ax2.set_yticklabels([f'{s} ({int(h)}m)' for s, h in zip(shortcuts, heights)],
    #                     fontsize=Fontsize - 4)
    # ax2.tick_params(direction='out', length=0)  # no tick marks, just labels

    ax.set_title('AWS temperature', fontsize=Fontsize)

    return ax

def plot_aws_wvmr(ax, data_aws, clim=[10, 15], band_thickness=0.12):
    """
    Plot station temperature as horizontal colored bands, 
    sorted by altitude, each band ~band_thickness meters thick.
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws['height']  = ds_aws['altitude'] - 577
    
    # Sort stations by altitude
    alt_order = ds_aws['height'].argsort().values
    ds_sorted = ds_aws.isel(station=alt_order)

    heights   = ds_sorted['height'].values/1000
    h0 = heights[0]
    heights[0] = heights[0] -0.045
    time      = ds_sorted['time'].values
    wvmr      = ds_sorted['wvmr'].values  # shape: (station, time)
    shortcuts = ds_sorted['shortcut'].values
    stations = ds_sorted['station'].values

    cmap = cmap_wvmr()
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    time_num = mdates.date2num(time)
    # grid edges for time
    dt = (time_num[1] - time_num[0]) / 2
    t_edges = np.concatenate([[time_num[0] - dt], time_num + dt])
    
    for i, (h, shortcut) in enumerate(zip(heights, shortcuts)):
        
            y0 = h - band_thickness / 2
            y1 = h + band_thickness / 2
    
            for j in range(len(time_num)):
                color = cmap(norm(wvmr[i, j]))
                ax.fill_betweenx([y0, y1], t_edges[j], t_edges[j+1], color=color)
            
            if i==0: station_label = f' {shortcut} ({int(h0*1000)} m AGL)'
            else: station_label = f' {shortcut} ({int(h*1000)} m AGL)'
            # Station label on the right
            ax.text(time_num[1] + dt*2, h, station_label,
                    va='center', fontsize=15)

    #pcm = ax.pcolormesh(t_edges, y_edges, temp, cmap=cmap, norm=norm, shading='flat')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
                                    
    # # Station labels on right side
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(heights)
    # ax2.set_yticklabels([f'{s} ({int(h)}m)' for s, h in zip(shortcuts, heights)],
    #                     fontsize=Fontsize - 4)
    # ax2.tick_params(direction='out', length=0)  # no tick marks, just labels

    ax.set_title('AWS temperature', fontsize=Fontsize)

    return ax

def plot_aws_ws(ax, data_aws, clim=[0, 20], band_thickness=0.12):
    """
    Plot station temperature as horizontal colored bands, 
    sorted by altitude, each band ~band_thickness meters thick.
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws['height']  = ds_aws['altitude'] - 577
    
    # Sort stations by altitude
    alt_order = ds_aws['height'].argsort().values
    ds_sorted = ds_aws.isel(station=alt_order)

    heights   = ds_sorted['height'].values/1000
    h0 = heights[0]
    heights[0] = heights[0] -0.045
    time      = ds_sorted['time'].values
    ff      = ds_sorted['ff'].values  # shape: (station, time)
    shortcuts = ds_sorted['shortcut'].values
    stations = ds_sorted['station'].values

    cmap = cmap_windspeed()
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    time_num = mdates.date2num(time)
    # grid edges for time
    dt = (time_num[1] - time_num[0]) / 2
    t_edges = np.concatenate([[time_num[0] - dt], time_num + dt])
    
    for i, (h, shortcut) in enumerate(zip(heights, shortcuts)):
        
            y0 = h - band_thickness / 2
            y1 = h + band_thickness / 2
    
            for j in range(len(time_num)):
                color = cmap(norm(ff[i, j]))
                ax.fill_betweenx([y0, y1], t_edges[j], t_edges[j+1], color=color)
            
            if i==0: station_label = f' {shortcut} ({int(h0*1000)} m AGL)'
            else: station_label = f' {shortcut} ({int(h*1000)} m AGL)'
            # Station label on the right
            ax.text(time_num[1] + dt*2, h, station_label,
                    va='center', fontsize=15)

    #pcm = ax.pcolormesh(t_edges, y_edges, temp, cmap=cmap, norm=norm, shading='flat')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
                                    
    # # Station labels on right side
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(heights)
    # ax2.set_yticklabels([f'{s} ({int(h)}m)' for s, h in zip(shortcuts, heights)],
    #                     fontsize=Fontsize - 4)
    # ax2.tick_params(direction='out', length=0)  # no tick marks, just labels

    ax.set_title('AWS temperature', fontsize=Fontsize)

    return ax



fig, ax = plt.subplots(figsize=(20, 4))

# ax = plot_aws_temp(ax, data_aws)
# ax = plot_aws_pres(ax, data_aws)
# ax = plot_aws_wvmr(ax, data_aws)
ax = plot_aws_ws(ax, data_aws)



ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
ax.set_xlim([start, end])
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(direction='out', labelsize=Fontsize)
ax.set_facecolor([0.8, 0.8, 0.8])