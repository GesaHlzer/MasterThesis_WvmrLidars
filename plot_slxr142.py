# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:13:27 2026

@author: alleh
"""
fig_size = [15, 5] #(20, 6)

import os
import numpy as np
import netCDF4
import xarray as xr
from datetime import datetime, timezone, timedelta 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
from basic_plot_funcions import savefig, grid_edges, cmap_windspeed

def read_slxr142(date):
    
    ncdir = os.path.join(os.path.dirname(os.getcwd()), 
                                         'data', 
                                         'SLXR142', 
                                         date.strftime('%Y%m'))
    ncfile = os.path.join(ncdir, date.strftime('%Y%m%d') + '.nc')
    nc =  netCDF4.Dataset(ncfile, 'r')
    
    # GEt all & remove last profile (assumes 2D arrays)
    height = nc.variables['height'][:]
    time = nc.variables['time'][:-1]
    ff = nc.variables['ff'][:, :-1]
    dd = nc.variables['dd'][:, :-1]
    intensity = nc.variables['intensity'][:, :-1]
    
    datetimes = [datetime.fromtimestamp(float(t), tz=timezone.utc) 
                 if not np.ma.is_masked(t) and not np.isnan(t) 
                 else None for t in time]
       
    # Convert lists to numpy arrays
    datetimes = np.array(datetimes)
    ff = np.array(ff)
    dd = np.array(dd)
    intensity = np.array(intensity)
    
    # Convert wind direction (dd) and speed (ff) into u and v components
    u_wind = -ff * np.sin(np.radians(dd))  # u-component (East-West)
    v_wind = -ff * np.cos(np.radians(dd))  # v-component (North-South)
    
    #  Apply intensity threshold
    threshold_indices = intensity < 1.0045
    ff[threshold_indices] = np.nan
    dd[threshold_indices] = np.nan
    
    # Create xarray Dataset
    data_slxr142 = xr.Dataset(
                            {'ff': (['height', 'time'], ff),
                             'dd': (['height', 'time'], dd),
                             'u_wind':(['height', 'time'], u_wind),
                             'v_wind':(['height', 'time'], v_wind),
                             'intensity': (['height', 'time'], intensity)},
                    coords={'time': datetimes,
                            'height': height} )
    return data_slxr142

def wind_barb(ax, data_sl, time, height):
    #(ax, ff, dd, time, height_km, ssize=0.01, bsize=0.006, bspace=0.18, tsize=2, lwidth=1, msize=2, angle=10):
    """
    Custom function to draw wind barbs

    Function for plotting wind barbs.
    Inputs:
    - ff: wind speed in knots
    - dd: wind direction in degrees
    - x, y: coordinates of origin
    - ssize: size of the stem
    - bsize: size of barbs
    - bspace: space between barbs
    - tsize: size of triangles
    - lwidth: width of barb lines
    - msize: size of marker at origin
    - angle: tilt angle of the barbs
    """
    # ssize=0.009, bsize=0.0045, bspace=0.18, tsize=2, lwidth=1, msize=5, angle=20
    
    par = data_sl['ff'].to_numpy() 
    u_wind_kn = data_sl['u_wind'].to_numpy() * 1.94384  # Convert m/s to knots
    v_wind_kn = data_sl['v_wind'].to_numpy() * 1.94384  # Convert m/s to knots
    
    # Calculations for Wind Barbs & grid Selection
    time_num = mdates.date2num(time)
    height_km = height.to_numpy()/1000
    
    # Create a mask for valid data where par is not NaN
    valid_mask = ~np.isnan(par)
    time_grid, height_grid = np.meshgrid(time_num, height_km)
    time_grid_masked = np.where(valid_mask, time_grid, np.nan)
    height_grid_masked = np.where(valid_mask, height_grid, np.nan)
    u_wind_kn = np.where(valid_mask, u_wind_kn, np.nan)
    v_wind_kn = np.where(valid_mask, v_wind_kn, np.nan)
    skip = (slice(None, None, 4), slice(None, None, 3))  # Use every 4th in time and 3rd in height
    
    ax.barbs(time_grid_masked[skip], height_grid_masked[skip], u_wind_kn[skip], v_wind_kn[skip], 
             length=4.2, #)
             linewidth=1.2, pivot='tip', sizes=dict(spacing=0.25, emptybarb=0.1)
             ) #  height = 0.4,
    
    ax.plot(time_grid_masked[skip], height_grid_masked[skip], 'ko', markersize=2)

    return ax

def wind_barb_legend(ax):
    
    # ---- Horizontal Legend Above the Main Plot
    ax_inset = inset_axes(ax, width="25%", height="23%", loc='lower left',
                          bbox_to_anchor=(0, 1.03, 1, 0.15),
                          bbox_transform=ax.transAxes, borderpad=0)

    # Define a custom coordinate system for the inset axes (4 samples horizontally)
    ax_inset.set_xlim(0, 5)
    ax_inset.set_ylim(0, 1)
    ax_inset.axis('off')  # Hide borders, ticks, labels
    
    # Define sample wind barb configurations. For a wind from the east, u is negative.
    samples = [
        {"label": "Calm",   "u": 0,   "v": 0},
        #{"label": "<5 kn",  "u": -3,  "v": 0},
        {"label": "5 kn",   "u": -5,  "v": 0},
        {"label": "10 kn",  "u": -10, "v": 0},
        {"label": "25 kn",  "u": -25, "v": 0},
        {"label": "50 kn",  "u": -50, "v": 0},
        ]
    
    # Place each sample evenly across the inset axes horizontally.
    for i, sample in enumerate(samples):
       x = i + 0.5   # Center of each sample (if xlim is [0,6])
       y = 0.8       # Vertical center within the inset
       
       # Draw the wind barb symbol matching the main plot style.
       ax_inset.barbs([x], [y], [sample["u"]], [sample["v"]],
                      length=5, linewidth=1.2, pivot='tip',
                      sizes=dict(spacing=0.2, emptybarb=0.15)
                      )
        #  spacing    # Distance between barbs
        #  height = 0.5     # Height of a barb relative to length
        #  emptybarb  # Radius of circle (if wind is calm)

       # Overlay the origin dot.
       ax_inset.plot([x], [y], 'ko', markersize=2.4)
       
       # Place the label below the symbol.
       ax_inset.text(x, y - 0.5, sample["label"], ha='center', va='top', fontsize=10)
                                               
    return ax_inset
    
def plot_slxr142(date, plot_horizontal_lines='no'):
    
    # Read data
    data_slxr142 = read_slxr142(date)
    
    par = data_slxr142['ff'].to_numpy() 
    par_cmap = cmap_windspeed()
    
    time = data_slxr142['time'].values
    height = data_slxr142['height']
    
    # t (time numeric) & h (height in km)
    t, h = grid_edges(time, height)   

    date_beg = mdates.date2num(date)
    date_end = mdates.date2num(date + timedelta(days=1))

    # ---- Plot Data
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, par, cmap=par_cmap, shading='flat')  # Adjust to match dimensions
    cbar = plt.colorbar(pcm, label='ff values')
    cbar.set_label('horizontal wind speed (m/s)', size=18)
    cbar.ax.tick_params(direction='out', labelsize=17, size=10)
    
    ax.set_xlim([date_beg, date_end])
    ax.set_ylim([0, h.max()])
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.tick_params(direction='out', labelsize=18)

    # ---- Add Plotting Lines if desired
    if plot_horizontal_lines == 'yes':
        ax.plot([date_beg, date_end], [0.2, 0.2], linestyle=':', linewidth=2, color=[0.5, 0.5, 0.5])
        ax.plot([date_beg, date_end], [1, 1], linestyle=':', linewidth=2, color=[0.5, 0.5, 0.5])
    
    # Add Title and Labels
    title_text = f"SLXR142: {date.strftime('%Y-%m-%d')}"
    ax.set_title(title_text, fontsize=18)
    ax.set_xlabel('time (UTC)', fontsize=18)
    ax.set_ylabel('height (km AGL)', fontsize=18)
    
    # ---- Add Wind Barbs & a legend
    ax = wind_barb(ax, data_slxr142, time, height)
    wind_barb_legend(ax)
    
    # Background color for NaN values
    ax.set_facecolor([0.9, 0.9, 0.9])
    fig.patch.set_facecolor([1, 1, 1])
    fig.patch.set_alpha(1.0)
    
    fig.tight_layout()
    plt.show()

    return fig

date = datetime(2024, 8, 7)    

date_beg, date_end = datetime(2024, 8, 22), datetime(2024, 9, 8)
dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]

for date in dates: 
    filename = f"SLXR142_{date.strftime('%Y-%m-%d')}.png"
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "SLXR142")
    
    fig = plot_slxr142(date, plot_horizontal_lines='no')
    savefig(fig, folderpath, filename)

print("done")