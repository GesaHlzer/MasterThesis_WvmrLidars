# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:42:47 2026

@author: alleh
"""

fig_size = [15, 5] #(20, 6)

import os
import pandas as pd
import numpy as np

import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

from colormaps import cmap_bluered16, cmap_adv_seq_mhue_inferno20
from basic_plot_funcions import savefig, grid_edges
  

def read_sl88_stare(date, start, end):
    
    # ---- SETTINGS
    
    # specify window size for moving average and moving variance 
    window_size = 5160  # number of datapoints that are approx 1 h # 3601  # seconds = 1 hour
    # window_size = 2580 # 1801  # seconds = 0.5 hour
    
    # specify data cleaning options 
    apply_deltavrad_threshold = 'yes'  
    # apply_deltavrad_threshold = 'no'
    
    apply_vrad_threshold = 'yes' 
    # apply_vrad_threshold = 'no' 
    
    apply_int_threshold = 'yes' 
    # apply_int_threshold = 'no'
    
    apply_nan_threshold = 'yes' 
    # apply_nan_threshold = 'no'
    
    apply_vradmean_threshold = 'yes' 
    # apply_vradmean_threshold = 'no'
    
    apply_vradvar_threshold = 'yes'
    # apply_vradvar_threshold = 'no'
    
    
    # ---- read all SL88 files within given date/time range
    ncdir = os.path.join(os.path.dirname(os.getcwd()), 
                         'data', 
                         'SL88', 
                         'SL88_stare', 
                         date.strftime('%Y%m%d')
                         )
    
    files = sorted([os.path.join(ncdir, f) for f in os.listdir(ncdir) if f.endswith('.nc')])
    
    # Open and concatenate datasets along time
    datasets = [xr.open_dataset(f) for f in files]
    count_duplicates = 0
    
    for i, ds in enumerate(datasets):
        # Get decimal hours
        decimal_hours = ds["decimal_time"].values
        datetime_coords = pd.Timestamp(date) + pd.to_timedelta(decimal_hours, unit="h")
        
        # Replace coords and swap dims        
        ds = ds.rename({'gate_centers': 'height'})
        ds = ds.swap_dims({"NUMBER_OF_GATES": "height"})
        
        ds = ds.assign_coords(time=("NUMBER_OF_RAYS", datetime_coords))
        ds = ds.swap_dims({"NUMBER_OF_RAYS": "time"})
        
        # Drop time duplicates if there
        time_index = ds.get_index("time")
        duplicated_mask = time_index.duplicated(keep="first")
        ds = ds.isel(time=~duplicated_mask)
        # count how many duplicates
        num_duplicates = duplicated_mask.sum()
        count_duplicates = count_duplicates + num_duplicates
        
        # Update the dataset in the list
        datasets[i] = ds
    
    if count_duplicates > 0:
        print(f"Removed {count_duplicates} duplicate time entries in SL88 stare data.")
        
    ds = xr.concat(datasets, dim = 'time')
        
    mask = ((ds.time >= start) & (ds.time <= end))
    ds = ds.sel(time=mask)
    
    vrad = ds['radial_velocity'].values.copy()
    intensity = ds['intensity'].values.copy()
    
    
    # ---- Apply data cleaning options
    
    # Set the first two gates (height levels) to NaN.
    n = 2
    vrad[:n,:] = np.nan
    
    # Delta-vrad threshold filter along height
    if apply_deltavrad_threshold == 'yes':
        # Flag values where the change between successive height measurements 
        # is larger than 2 m/s (set to NaN)
        deltavrad_threshold = 2  # m/s
        deltavrad = np.abs(np.diff(vrad, axis=0))  # result shape: (height-1, time)
        rows, cols = np.where(deltavrad > deltavrad_threshold)
        vrad[rows + 1, cols] = np.nan
        
    # Vrad threshold filter
    if apply_vrad_threshold == 'yes':   
        # Flags any value that exceeds an absolute threshold as NaN
        vrad_threshold = 5  # m/s
        mask = np.abs(vrad) > vrad_threshold
        vrad[mask] = np.nan
        
    # Intensity threshold
    if apply_int_threshold == 'yes':  
        # Where intensity is below the threshold (1.003) and the corresponding vrad
        # is valid, mark vrad as NaN.
        int_threshold = 1.003
        mask_int = (intensity < int_threshold) & (~np.isnan(vrad))
        vrad[mask_int] = np.nan
        
    # NaN neighbor check along height
    if apply_nan_threshold == 'yes':        
        # For each time sample, check (excluding the first and las height)
        # if a non-NaN vrad value is isolated 
        mask_center_valid = ~np.isnan(vrad[1:-1, :])
        mask_neighbors_nan = np.isnan(vrad[:-2, :]) & np.isnan(vrad[2:, :])
        mask_isolated = mask_center_valid & mask_neighbors_nan
        isolated_rows, isolated_cols = np.where(mask_isolated)
        vrad[isolated_rows + 1, isolated_cols] = np.nan
        
        # if a non-NaN value is isolated in time (i.e., surrounded by NaNs)
        mask_center_valid = ~np.isnan(vrad[:, 1:-1])
        mask_neighbors_nan = np.isnan(vrad[:, :-2]) & np.isnan(vrad[:, 2:])
        mask_isolated = mask_center_valid & mask_neighbors_nan
        isolated_rows, isolated_cols = np.where(mask_isolated)
        # Mark these isolated time points as NaN
        vrad[isolated_rows, isolated_cols + 1] = np.nan
        
            
    # ---- Calculate moving average and moving variance:
       
    # need to transpose since time is axis=1 and need it as 0 for rolling
    vrad_T = vrad.transpose()  # Now shape is (time, height)
    vradmean = (pd.DataFrame(vrad_T).rolling(window=window_size, min_periods=1)
                                    .mean()
                                    .to_numpy()
                                    .transpose()
                )
    vradvar = (pd.DataFrame(vrad_T).rolling(window=window_size, min_periods=1)
                                  .var()
                                  .to_numpy()
                                  .transpose()
                )
    
    # apply thresholds to moving variance
    nan_mask = np.isnan(vrad)
    vradvar[nan_mask] = np.nan
    
    removed_percentage = round(100 * np.sum(nan_mask) / vradvar.size, 2)
    print("Applying threshold to remove bad data ...")
    print(f"... {removed_percentage}% removed!")
    
        
    # --- Update Dataset 
    # Update the original xarray.Dataset
    ds["vrad"] = (("height", "time"), vrad)
    ds["vrad"].attrs["long_name"] = "Doppler velocity along line of sight" 
    ds["vrad"].attrs["units"] = "m s-1"
    
    ds["vradmean"] = (("height", "time"), vradmean)
    ds["vradmean"].attrs["long_name"] = "Moving average of Doppler velocity along line of sight" 
    ds["vradmean"].attrs["units"] = "m s-1"
    
    ds["vradvar"] = (("height", "time"), vradvar)
    ds["vradvar"].attrs["long_name"] = "Moving variance of Doppler velocity along line of sight" 
    ds["vradvar"].attrs["units"] = "m2 s-2"  
    
    ds = ds[["vrad", "vradmean", "vradvar", "intensity"]]
    
    return ds

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
    
def plot_sl88_stare(date, start, end):
    
    print("\n Making DIAL SL88 vertical wind veloc. plot... ")
    
    fontsize = 24
    # ---- specify parameter type 
    para_type = 'vrad'           # instantaneous (1 s) radial velocity (vrad)
    # para_type = 'vrad_movmean'  # moving average of vrad
    # para_type = 'vrad_movvar'   # moving variance of vrad
    
    # ---- specify contour option for velocity variance
    plot_var_contour = 'yes'
    # plot_var_contour = 'no';
    
    contour_threshold = 'intensity';
    # contour_threshold = 'height'
    contour_threshold = 'none'
    
    cont_vals = [0.05, 0.2, 1.0] # [0.05, 0.2, 1.0]
    cont_col = [(0.2, 0.2, 0.2)] * 3  # Creates three identical RGB tuples
    cont_style = ['dotted', 'dashed', 'solid'] # [':', '--', '-']
    
    # --- Obtain Data to Plot
    
    data_sl88 = read_sl88_stare(date, start, end)
    
    time = data_sl88['time']
    heights = data_sl88['height']
    t, h = grid_edges(time, heights)


    # Determine the plotting parameters based on "par"
    
    if para_type == 'vrad':              # vertical velocity (m/s)
        par = data_sl88['vrad']
        cmap = cmap_bluered16()
        clim = [-2,2]
        N = 16
        cbar_label = 'vertical velocity (m/s)'
        
        
    elif para_type == 'vrad_movmean':    # mean vertical velocity (m/s)
        par = data_sl88['vradmean']
        cmap = cmap_bluered16()
        clim = [-0.8, 0.8]
        N = 16
        cbar_label = 'mean vertical velocity (m/s)'
        
    elif para_type == 'vrad_movvar':     # vertical velocity variance (m^2/s^2)
        par = data_sl88['vradvar']
        cmap = cmap_adv_seq_mhue_inferno20()
        clim = [0, 4]
        N = 20
        cbar_label ='vertical velocity variance (m²/s²)'
    
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(clim[0], clim[1], N+1), ncolors=256)
    par = np.ma.masked_invalid(par)
    #cmap.set_bad(color='gray') 

    # Contour Grid and Data if needed
    if plot_var_contour == 'yes':
        
        h2 = heights.values/1000
        t2 = mdates.date2num(time)
        
        vradvar = data_sl88['vradvar']
        intensity = data_sl88['intensity']
        
        if contour_threshold == 'intensity':
            print('Applying intensity threshold for variance contouring ...')
            vradvar[intensity < 1.015] = np.nan
            
        elif contour_threshold == 'height':
            print('Applying height threshold for variance contouring ...')
            vradvar[h2 > 1.3] = np.nan
        
        vradvar = np.ma.masked_invalid(vradvar)
        
    # ---- Plot Data 
    
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    pcm = ax.pcolormesh(t, h, par, cmap=cmap, norm=norm, shading='flat')
    
    # Axis settings
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0], interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(direction='out', labelsize=fontsize)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax.set_xlim([start, end])
    ax.set_ylim([0, 2.4])
    
    # Colorbar settings
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02) #, ticks=np.linspace(clim[0], clim[1], N)
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=9)
    cbar.ax.set_ylabel(cbar_label, fontsize=fontsize)
    pcm.set_clim(clim)
   
    # Handle contour plotting if needed
    if plot_var_contour == 'yes':
        
        contour = ax.contour(t2, h2, vradvar,
                         levels=cont_vals,
                         linewidths=2.0,
                         linestyles=cont_style,
                         colors=cont_col)
        
        handles, labels = contour.legend_elements()
        ax.legend(handles=handles, 
                  labels=[fr'$σ^2_w$ = {val} m²/s²' for val in cont_vals], 
                  loc='upper right', fontsize=fontsize-4)
        

    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_title(f"SL88: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}", fontsize=fontsize)
    
    # Background color for NaN values
    ax.set_facecolor([0.9, 0.9, 0.9])
    # fig.patch.set_facecolor([1, 1, 1])
    # fig.patch.set_alpha(1.0)
    
    fig.tight_layout()
    plt.show()

    return fig

 
date  = datetime(2024, 8, 24)
ticks = "timewindow"
fig_size = [15, 7] # [15, 5]
# start = np.datetime64(date) + np.timedelta64((60*18+35), 'm')
# end   = np.datetime64(date) + np.timedelta64((60*20+10), 'm')
start = np.datetime64(date) + np.timedelta64((60*10+10), 'm') #for 28th
end   = np.datetime64(date) + np.timedelta64((60*16+0), 'm') #for 28th
# start = np.datetime64(date) + np.timedelta64((60*7+50), 'm') #for 28th
# end   = np.datetime64(date) + np.timedelta64((60*9+15), 'm') #for 28th

filename = f"SL88_stare_{start.astype('datetime64[h]').astype(str)}_{end.astype('datetime64[h]').astype(str)}.png"
folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "SL88_stare", "shorttime")

fig = plot_sl88_stare(date, start, end)
savefig(fig, folderpath, filename)


print("done")