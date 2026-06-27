# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:44:58 2026

@author: alleh
"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
from basic_plot_funcions import savefig, grid_edges
from colormaps import cmap_wvmr, cmap_windspeed, cmap_abs, cmap_bluered16, cmap_adv_seq_mhue_inferno20
# from plot_slxr142 import wind_barb, wind_barb_legend



data_dial_wv = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc") 
data_dial_bs = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_abs.nc")
data_ppl1    = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_50.0%valid.nc") 
data_ppl2    = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_75.0%valid.nc")
# data_sl88    = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\SL88\sl88_stare\SL88_stare_202408070000_202409090000.nc")
data_slxr142 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\SLXR142\SLXR142_202407180000_202410230000.nc")
data_aws =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\stationsdata.nc")

data_ppl1['height'] = data_ppl1['height'] - 3 
data_ppl2['height'] = data_ppl2['height'] - 3 

# time period
start = np.datetime64("2024-08-23") #- np.timedelta64(1, 'm')
end   = np.datetime64("2024-08-25") #- np.timedelta64(1, 's')
begt  = start #+ np.timedelta64(1, 'm')
endt  = end - np.timedelta64(1, 'ns')

Fontsize = 28
Ticksize = 10

def plot_aws_temp(ax, data_aws, clim=[8, 38], band_thickness=0.12):
    """
    Plot station temperature as horizontal colored bands, 
    sorted by altitude, each band ~band_thickness meters thick.
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T
    #ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
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

    cmap = plt.get_cmap('RdYlBu_r') #turbo #jet
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

def plot_aws_temp_lines(ax, data_aws, var='temp'):
    """
    Plot AWS station temperature or potential temperature as time series lines.
    Designed to fit into the multi-panel figure layout.
    var: 'temp' for temperature (°C), 'theta' for potential temperature (K)
    """
    ds_aws = data_aws.sel(station=~data_aws['station'].isin(['Hauptbahn','Rastlbode', 'Olympisch']))
    
    ds_aws['time']  = ds_aws['time'] - np.timedelta64(5,  'm')  # (10min avg)
    ds_aws = ds_aws.sel(time=slice(start, end))#.values#.T

    # Sort stations by altitude
    alt_order = ds_aws['altitude'].argsort().values
    ds = ds_aws.isel(station=alt_order)
    
    if var == 'theta':
        T = ds['temp'] + 273.15
        p0 = 1000.0
        data_var = T * (p0 / ds['p_estimated']) ** 0.2854
        ylabel = 'θ (K)'
        title  = 'AWS pot. Temperatur'
    else:
        data_var = ds['temp']
        ylabel = 'T (°C)'
        title  = 'AWS Temperatur'

    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.station)))
    linestyles = ['-', '--', '-.', ':', (0,(5,2)), (0,(3,1,1,1)), '-', '--', '-.']
    
    # ── Dummy colorbar to match width of other panels ──
    sm = plt.cm.ScalarMappable(cmap='rainbow')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.set_visible(False)  # invisible but reserves the space

    legend_handles = []
    for i, station in enumerate(ds['station'].values):
        shortcut = str(ds.sel(station=station)['shortcut'].values)
        alt_asl  = int(ds.sel(station=station)['altitude'].values)
        alt_agl  = (alt_asl - 577 ) /1000 # Innsbruck valley floor ≈ 577 m ASL — adjust if needed
        y = data_var.sel(station=station).values
        line, = ax.plot(ds['time'].values, y,
                        color=colors[i],
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=1.5,
                        label=f'{alt_agl:.1f} km')#{shortcut}')#' ({alt_agl:.1f} km AGL)')
        legend_handles.append(line)
    # for i, station in enumerate(ds_aws['station'].values):
    #     shortcut = ds_aws.sel(station=station)['shortcut'].values
    #     alt = int(ds_aws.sel(station=station)['altitude'].values)
    #     h = int(ds_aws.sel(station=station)['height'].values)/1000
    #     y = data_var.sel(station=station).values
    #     ax.plot(ds_aws['time'].values, y,
    #             color=colors[i],
    #             linestyle=linestyles[i % len(linestyles)],
    #             linewidth=1.2,
    #             label=f'{shortcut} ({alt} m ASL, {h} m AGL)')

    ax.set_ylabel(ylabel, fontsize=Fontsize)
    ax.set_title(title, fontsize=Fontsize)
    # ax.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    
    ax.legend(handles=legend_handles[::-1],
              loc='upper left',
              bbox_to_anchor=(1.02, 1.0),  # rechts vom Plot, oben bündig
              borderaxespad=0,
              fontsize=Fontsize - 5,
              ncol=1,
              framealpha=0.9,
              handlelength=2.5,
              edgecolor='gray')

    ax._aws_ylabel = ylabel  # store for post-loop correction

    return ax

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
    skip = (slice(None, None, 6), slice(None, None, 7))  # Use every 4th in time and 3rd in height
    # skip = (slice(None, None, 4), slice(None, None, 3))  # Use every 4th in time and 3rd in height

    ax.barbs(time_grid_masked[skip], height_grid_masked[skip], u_wind_kn[skip], v_wind_kn[skip], 
             length=5,linewidth=1.6, pivot='tip', sizes=dict(spacing=0.3, emptybarb=0.2)
             # length=4.2,linewidth=1.2, pivot='tip', sizes=dict(spacing=0.25, emptybarb=0.1)#  height = 0.4,
             ) 
    
    ax.plot(time_grid_masked[skip], height_grid_masked[skip], 'ko', markersize=2)

    return ax

def wind_barb_legend(ax):
    
    # ---- Horizontal Legend Above the Main Plot
    ax_inset = inset_axes(ax, width="30%", height="13%", loc='upper left',
                          bbox_to_anchor=(0.01,-0.03, 1, 1), #(0, 1.03, 1, 0.15)
                          bbox_transform=ax.transAxes, borderpad=0)


    # Define a custom coordinate system for the inset axes (5 samples horizontally)
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
        y = 0.7       # Vertical center within the inset
        
        # Draw the wind barb symbol matching the main plot style.
        ax_inset.barbs([x], [y], [sample["u"]], [sample["v"]],
                       length=5.5, linewidth=1.6, pivot='tip',
                       sizes=dict(spacing=0.25, emptybarb=0.2)
                       )
         #  spacing    # Distance between barbs
         #  height = 0.5     # Height of a barb relative to length
         #  emptybarb  # Radius of circle (if wind is calm)
         
        # Overlay the origin dot.
        ax_inset.plot([x], [y], 'ko', markersize=1.5)
        
        # Place the label below the symbol.
        ax_inset.text(x, y - 0.25, sample["label"], ha='center', va='top', 
                      fontsize=15)
        
        # ---- Box hinzufügen
        ax_inset.set_axis_off()
        fancy_box = FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=1",
                                    transform=ax_inset.transAxes,
                                    facecolor='whitesmoke', edgecolor='none',
                                    alpha=0.85, zorder=0)
        ax_inset.add_patch(fancy_box)
           
    return ax_inset
    
def plot_sl88_stare(ax, hmax):#, data_sl88=data_sl88):
   
    # # begt  = start #+ np.timedelta64(1, 'm')
    # # endt  = end - np.timedelta64(1, 'ns') 
    # data_sl88 = data_sl88.sortby('time')
    # # ds_sl88 = data_sl88.copy()
    # ds_sl88 = data_sl88.sel(time=slice(start, end))#.values#.T
    import os
    from datetime import datetime
    def read_sl88_stare(date):
        
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
            
        # mask = ((ds.time >= start) & (ds.time <= end))
        # ds = ds.sel(time=mask)
        
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
    
    data_sl88_23 = read_sl88_stare(datetime(2024, 8, 23))
    data_sl88_24 = read_sl88_stare(datetime(2024, 8, 24))
    data_sl88 = xr.concat([data_sl88_23,data_sl88_24], dim='time')
    data_sl88 = data_sl88.sortby('time')
    ds_sl88 = data_sl88.sel(time=slice(start, end))#.values#.T
    
    t, h = grid_edges(ds_sl88['time'], ds_sl88['height'])
    
    # ---- specify parameter type 
    para_type = 'vrad'              # instantaneous (1 s) radial velocity (vrad)
    # para_type = 'vrad_movmean'    # moving average of vrad
    # para_type = 'vrad_movvar'     # moving variance of vrad
    
    plot_var_contour = 'yes'# 'no'  # contour for velocity variance
    contour_threshold = 'none'    #'intensity''none' height
    cont_vals = [0.05, 0.2, 1.0] # [0.05, 0.2, 1.0]
    cont_col = [(0.2, 0.2, 0.2)] * 3  # Creates three identical RGB tuples
    cont_style = ['dotted', 'dashed', 'solid'] # [':', '--', '-']
    
    
    if para_type == 'vrad':              # vertical velocity (m/s)
        par = ds_sl88['vrad']
        cmap = cmap_bluered16()
        clim = [-2,2]
        N = 16
        cbar_label = r'vert. vel. (m s$^{-1}$)'#'vertical velocity (m/s)'
          
    elif para_type == 'vrad_movmean':    # mean vertical velocity (m/s)
        par = ds_sl88['vradmean']
        cmap = cmap_bluered16()
        clim = [-0.8, 0.8]
        N = 16
        cbar_label = r'mean vertical velocity (m s$^{-1}$)'
        
    elif para_type == 'vrad_movvar':     # vertical velocity variance (m^2/s^2)
        par = ds_sl88['vradvar']
        cmap = cmap_adv_seq_mhue_inferno20()
        clim = [0, 4]
        N = 20
        cbar_label =r'vertical velocity variance (m$^{2}$ s$^{-2}$)'
    
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(clim[0], clim[1], N+1), ncolors=256)
    par = np.ma.masked_invalid(par)

    # Contour Grid and Data if needed
    if plot_var_contour == 'yes':
        
        h2 = ds_sl88['height'].values/1000
        t2 = mdates.date2num(ds_sl88['time'])
        
        vradvar = ds_sl88['vradvar']
        intensity = ds_sl88['intensity']
        
        if contour_threshold == 'intensity':
            print('Applying intensity threshold for variance contouring ...')
            vradvar[intensity < 1.015] = np.nan
            
        elif contour_threshold == 'height':
            print('Applying height threshold for variance contouring ...')
            vradvar[h2 > 1.3] = np.nan
        
        vradvar = np.ma.masked_invalid(vradvar)
        
    # ---- axis plot
    pcm = ax.pcolormesh(t, h, par, cmap=cmap, norm=norm, shading='flat')
    
    ax.set_title(f"SL88 vertical velocity",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
        fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    
    # Colorbar settings
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02) #, ticks=np.linspace(clim[0], clim[1], N)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
    cbar.ax.set_ylabel(cbar_label, fontsize=Fontsize)
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
                  loc='upper left', fontsize=15)
        
    return ax

def plot_slxr142_vad(ax, hmax, data_slxr142=data_slxr142, clim=[0,20]):
    
    # begt  = start #+ np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    data_slxr142 = data_slxr142.sortby('time')
    
    ds_slxr142 = data_slxr142.copy()
    ds_slxr142['time'] = ds_slxr142['time'] - np.timedelta64(5, 'm')
    
    ds_slxr142 = ds_slxr142.sel(time=slice(start, end))#.values#.T
    t, h = grid_edges(ds_slxr142['time'].values, ds_slxr142['height'])

    par = ds_slxr142['ff'].values
    par_cmap = cmap_windspeed()
    
    # ---- Plot ax
    title_text = ax.set_title(f"SLXR142 horizontal wind speed",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
                 fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    
    pcm = ax.pcolormesh(t, h, par, cmap=par_cmap, shading='flat', vmin=clim[0], vmax=clim[1])  # Adjust to match dimensions
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'ff (m s$^{-1}$)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    
    # ---- Add Wind Barbs & a legend
    ax = wind_barb(ax, ds_slxr142, ds_slxr142['time'].values, ds_slxr142['height'])
    wind_barb_legend(ax)
    
    # ax.set_xlim([beg, date_end])

    # # ---- Add Plotting Lines if desired
    # if plot_horizontal_lines == 'yes':
    #     ax.plot([date_beg, date_end], [0.2, 0.2], linestyle=':', linewidth=2, color=[0.5, 0.5, 0.5])
    #     ax.plot([date_beg, date_end], [1, 1], linestyle=':', linewidth=2, color=[0.5, 0.5, 0.5])
    
    return ax

def plot_dial_wv(ax, hmax, data_dial_wv=data_dial_wv, clim=[0, 15]):
       
    # begt  = start #+ np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    data_dial_wv = data_dial_wv.sortby('time')
    
    ds_dial_wv = data_dial_wv.copy()
    ds_dial_wv['time'] = ds_dial_wv['time'] - np.timedelta64(10, 'm')
    
    ds_dial_wv = data_dial_wv.sel(time=slice(start, end))#.values#.T
    
    da10_maxrange = ds_dial_wv['water_vapor_max_range']/1000
    ds_dial_wv = ds_dial_wv['water_vapor']#.where(ds_dial['height'] < ds_dial['water_vapor_max_range'])#.values.T
    ds_da10_wv_f = ds_dial_wv.values.T
    

    t, h = grid_edges(ds_dial_wv['time'], ds_dial_wv['height'])
    
    par_cmap_wv  = cmap_wvmr()#'Blues'

    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    ax.set_title(f"DA10 water vapor mixing ratio",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
                 fontsize=Fontsize)#, fontweight='bold')
    pcm  = ax.pcolormesh(t, h, ds_da10_wv_f, shading='flat', cmap=par_cmap_wv, vmin=clim[0], vmax=clim[1])
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, norm='log')
    cbar.set_label(r'wvmr (g kg$^{-1}$)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.plot(ds_dial_wv['time'], da10_maxrange, 'r')

    return ax

def plot_rl1_wv(ax, hmax, data_ppl1=data_ppl1, clim=[0, 15]):
       
    # begt  = start + np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    ds_ppl1 = data_ppl1.copy()
    ds_ppl1['time'] = ds_ppl1['time'] - np.timedelta64(5, 's')
    
    ds_ppl1 = ds_ppl1.sel(time=slice(start, end))#.values#.T
    ds_ppl1_wv = ds_ppl1['wvmr']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T
    # ds_ppl1 = ds_ppl1['wvmr_filtered']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T
    
    ds_ppl1_wv_f = ds_ppl1_wv.values.T
    
    ppl1_maxrange = ds_ppl1['wvmr_max_range']/1000
    
    t, h = grid_edges(ds_ppl1['time'], ds_ppl1['height'])
    
    par_cmap_wv  = cmap_wvmr()#'Blues'

    ax.set_title(f"PPLS-10s water vapor mixing ratio",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
                 fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    pcm  = ax.pcolormesh(t, h, ds_ppl1_wv_f, shading='flat', cmap=par_cmap_wv, vmin=clim[0], vmax=clim[1])                
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, norm='log')
    cbar.set_label(r'wvmr (g kg$^{-1}$)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    ax.plot(ds_ppl1['time'], ppl1_maxrange, 'r')

    return ax

def plot_rl2_wv(ax, hmax, data_ppl2=data_ppl2, clim=[0, 15]):
       
    # begtn = begt - np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    ds_ppl2 = data_ppl2.copy()
    ds_ppl2['time'] = ds_ppl2['time'] - np.timedelta64(10, 'm')
    
    ds_ppl2 = ds_ppl2.sel(time=slice(start, end))#.values#.T
    ds_ppl2_wv = ds_ppl2['wvmr']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T
    # ds_ppl2 = ds_ppl2['wvmr_filtered']#.where(ds_ppl2['height'] < ds_ppl2['wvmr_max_range'])#.values.T
    ds_ppl2_wv_f = ds_ppl2_wv.values.T
    
    ppl2_maxrange = ds_ppl2['wvmr_max_range']/1000
    
    t, h = grid_edges(ds_ppl2['time'], ds_ppl2['height'])
    
    par_cmap_wv = cmap_wvmr()#'Blues'

    ax.set_title(f"PPLS-20min water vapor mixing ratio",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
                 fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    pcm  = ax.pcolormesh(t, h, ds_ppl2_wv_f, shading='flat', cmap=par_cmap_wv, vmin=clim[0], vmax=clim[1])                
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, norm='log')
    cbar.set_label(r'wvmr (g kg$^{-1}$)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    ax.plot(ds_ppl2['time'], ppl2_maxrange, 'r')

    return ax

def plot_dial_bs(ax, hmax, data_dial_bs=data_dial_bs):
    
    # begt  = start + np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    data_dial_bs = data_dial_bs.sortby('time')
    ds_dial_bs = data_dial_bs.sel(time=slice(start, end))#.values#.Tpl1 = data_ppl1.sel(time=slice(start, end))#.values#.T
    ds_dial_bs_f = ds_dial_bs.beta_att.values.T

    t, h = grid_edges(ds_dial_bs['time'], ds_dial_bs['height'])
    
    par_cmap = 'Grays' #cmap_abs()
    norm = mcolors.LogNorm(vmin=np.nanmin(data_dial_bs["beta_att"].where(data_dial_bs["beta_att"] > 0)), 
                           vmax=np.nanmax(data_dial_bs["beta_att"]))
    # norm = mcolors.LogNorm(vmin=np.nanmin(param[param > 0]),vmax=np.nanmax(param))
    norm = mcolors.LogNorm(vmin=1e-8, vmax=5e-6)
    # ----  Plot Ax 
    pcm = ax.pcolormesh(t, h, ds_dial_bs_f, shading='flat', cmap=par_cmap, norm=norm)
    #, norm=cbar_norm  # vmax=(max(40, param.max())),vmin=0, vmax=0.0002, Adjust to match dimensions
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='neither', norm='log')
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    cbar.set_label('att. bsc. coef. (m$^{-1}$sr$^{-1})$', size=Fontsize)
    #cbar.ax.set_yscale("log")
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    ax.set_title(f"DA10 attenuated volume backscatter coefficient",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}",
                 fontsize=Fontsize)#, fontweight='bold')
    
    return ax

def plot_rl1_br(ax, hmax, data_ppl1=data_ppl1, clim=[0, 2.2]):
       
    # begt  = start + np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    
    ds_ppl1 = data_ppl1.sel(time=slice(start, end))#.values#.T
    ds_ppl1_br = ds_ppl1['br']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T

    ds_ppl1_br_f = ds_ppl1_br.values.T
    
    t, h = grid_edges(ds_ppl1['time'], ds_ppl1['height'])
    
    par_cmap_bsr = cmap_abs()

    ax.set_title(f"PPLS-10s backscatter ratio",#: {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}", 
                 fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    pcm  = ax.pcolormesh(t, h, ds_ppl1_br_f, shading='flat', cmap=par_cmap_bsr, vmin=clim[0], vmax=clim[1])                
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='neither')

    cbar.set_label( 'bsc. ratio (unitless)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) 
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    return ax

def plot_rl2_br(ax, hmax, data_ppl2=data_ppl2, clim=[0, 2]):
       
    # begt  = start + np.timedelta64(1, 'm')
    # endt  = end - np.timedelta64(1, 'ns') 
    ds_ppl2 = data_ppl2.copy()
    ds_ppl2['time'] = ds_ppl2['time'] - np.timedelta64(10, 'm')
    
    ds_ppl2 = ds_ppl2.sel(time=slice(start, end))#.values#.T
    ds_ppl2_br = ds_ppl2['br']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T

    ds_ppl2_br_f = ds_ppl2_br.values.T
    
    t, h = grid_edges(ds_ppl2['time'], ds_ppl2['height'])
    
    par_cmap_bsr = cmap_abs()

    ax.set_title(f"PPLS-20min backscatter ratio",#": {np.datetime_as_string(begt, unit='m')} - {np.datetime_as_string(endt, unit='m')}",
                 fontsize=Fontsize)#, fontweight='bold')
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    pcm  = ax.pcolormesh(t, h, ds_ppl2_br_f, shading='flat', cmap=par_cmap_bsr, vmin=clim[0], vmax=clim[1])                
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='neither')

    cbar.set_label('bsc. ratio (unitless)', size=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) 
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    return ax



# # Verwendung:
# fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# plot_aws_temp_lines(axes[0], data_aws, start, end, var='temp')
# plot_aws_temp_lines(axes[1], data_aws, start, end, var='theta')
# plt.tight_layout()
# plt.show()
ws=None

# rows = 8
# hmax_height_ratios = [4, 4, 4, 4, 6, 6, 6, 6] #km AGL
# vvad, vst, dw, db, r1w, r1b, r2w, r2b = 0, 1, 2, 3, 4, 5, 6, 7
# Figsize = (20, 30)

# rows = 4
# hmax_height_ratios = [5, 5, 3, 2.5] #km AGL
# Figsize = (18, 26)
# vvad, vst, r1b, r2w = 2, 3, 0, 1

# rows = 2
# Figsize = (20, 24)
# vvad,  dw = 0, 1
# hmax_height_ratios = [3, 3]

# rows = 5
# hmax_height_ratios = [ 5, 5, 3, 2.8, 2.2] #km AGL
# Figsize = (28, 35) #(26, 35)
# vvad, vst, r1b, r2w, ws = 2, 3, 0, 1, 4 #3,4,1,2,0  #

####
rows = 4
hmax_height_ratios = [5, 5, 5, 3] #km AGL
Figsize = (28, 33)#(18, 26)
r2b, db, r1w, dw =  0, 1, 2, 3


fig, (axes) = plt.subplots(nrows=rows, figsize=Figsize, 
                           gridspec_kw={'height_ratios': hmax_height_ratios})


# axes[ws] = plot_aws_temp(axes[ws], data_aws)
# axes[ws] = plot_aws_temp_lines(axes[ws], data_aws, var='theta')

axes[dw] = plot_dial_wv(axes[dw],       hmax=3)
axes[db] = plot_dial_bs(axes[db],       hmax=5)

axes[r1w] = plot_rl1_wv(axes[r1w],      hmax=5)
# axes[r1b] = plot_rl1_br(axes[r1b],      hmax=5)

# axes[r2w] = plot_rl2_wv(axes[r2w],      hmax=5)
axes[r2b] = plot_rl2_br(axes[r2b],      hmax=5)

# axes[vvad] = plot_slxr142_vad(axes[vvad], hmax=3)
# axes[vst] = plot_sl88_stare(axes[vst],  hmax=2.8)


for ax, hmax in zip(axes, hmax_height_ratios):

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(direction='out', labelsize=Fontsize, size=Ticksize)

    ax.set_xlim([start, end])
    ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
    ax.set_facecolor([0.8, 0.8, 0.8])
    
# ── Korrekturen für den AWS-Linienplot ──────────────────────────────────
if ws is not None:
    if hasattr(axes[ws], '_aws_ylabel'):
        axes[ws].set_ylabel(axes[ws]._aws_ylabel, fontsize=Fontsize)
    axes[ws].yaxis.set_major_locator(ticker.MultipleLocator(5))   # statt MultipleLocator(1)
    axes[ws].set_facecolor('white')

axes[-1].set_xlabel('time (UTC)', fontsize=Fontsize)
fig.align_xlabels()  
fig.align_ylabels()  
# fig.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.show()

# folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\LidarComparison"
# savefig(fig, folderpath, filename, show=True)
