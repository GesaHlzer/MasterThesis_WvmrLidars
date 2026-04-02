# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:04:23 2026

@author: alleh
"""

import numpy as np
import xarray as xr
import os
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

from basic_plot_funcions import grid_edges, savefig

date  = datetime(2024, 8, 28)

cut = False

if cut:
    hmax=3000
    tawesbandmeter = 120 # m
else: 
    hmax = 4000 # meter
    tawesbandmeter = 140 # m
    
dh   = 10   # meter
dt   = 60   # sec

fig_size = [15, 5]

rsbandminutes = 18 # min 

# --- Load data
da10     = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','dial_wvmr.nc'))
rsondes  = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','radiosondes.nc'))
stations = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','stationdata.nc'))

def combine_data_on_date(date): #, rsondes, stations, da10):
    
    # aws     # dt 10 min                 # hmax 2300  m
    # raso    # dt 1        # dh var      # hmax ...
    # da      # dt 1 min    # dh  9.6 m   # hmax 4000  m
    # pl      # dt 60 sec   # dh 3.75 m   # hmax 12000 m
    
    # Select timeperiod 
    start = np.datetime64(date)
    end   = np.datetime64(date + timedelta(days=1))
    
    da = da10.sel(time=slice(start, end))
    ws = stations.sel(time=slice(start, end))
    rs = rsondes.sel(launch=rsondes.date == np.datetime64(date))
 
    
    # Unify 'height' coordinates with 577 m ASL as ground reference
    # da['height'] in AGL wit ground level 571 
    ws['height']  = ws['altitude'] - 577         # 0 -> 571 m ASL
    rs['height']  = rs['altitude'] - 577         # 0 -> 571 m ASL
    
    # --- Drop unnecassary data 
    
    # Mask da10 water-vapor above water_vapor_max_range
    if cut: 
        da["water_vapor"]  = da['water_vapor'].where(da['height'] <= da['water_vapor_max_range'])
    da  =  da[['time', 'height', 'water_vapor', 'water_vapor_max_range']] #da[['mr', 'mr_unc', 'longitude', 'latitude'] ] 
    
    rs = rs[['time', 'height', 'mr']] #rs[['time', 'height', 'mr', 't', 'p', 'lon', 'lat', 'launch', 'date', 'day_night']]
    # rs = rs.where(rs.height <= hmax+10, drop=True)
    #rs = rs.dropna(dim="index", subset=["mr", 'time', 'height']) #, "t"])
    
    ws = ws.sel(station='Tawes', drop=True)
    ws = ws[['time', 'height', 'mr']] # 't'
    
    # --- define a fixed height & time grid
    
    gridheight = np.arange(0, hmax+1 , dh, dtype='int64')
    gridtime = np.arange(start, end, np.timedelta64(dt, 's'), dtype='datetime64[ns]')
    
    # --- Interpolate Lidar data on gridheight and -time
    
    # make sure all times are unique and handle NaT
    vals_da, idx_da = np.unique(da['time'].values, return_index=True)
    da = da.isel(time=np.sort(idx_da))
    da = da.interp({'height': gridheight, 'time': gridtime}, method='linear')
    

    # --- Interpolate Raso on gridheight
    
    def interpolate_raso(rs, gridheight):
        
        launches = rs.launch.values
        all_wvmr = []
        all_time = []
        all_launchtime = []
        all_endtime = []
        all_flight_s = [] 
        all_flight_min = []
        
        for L in launches:
            
            rs_L = rs.sel(launch=L)
            rs_L = rs_L.dropna(dim="index", subset=["mr", 'time', 'height'])
            
            # Extract 1D arrays for this launch
            h = rs_L.height.values
            t = rs_L.time.values.astype("datetime64[ns]").astype(float)
            wvmr = rs_L.mr.values
    
            # Mask invalid points
            mask = np.isfinite(h) & np.isfinite(t) & np.isfinite(wvmr)
        
            # Build interpolators
            f_time = interp1d(h[mask], t[mask], kind="linear", bounds_error=False, fill_value=np.nan)
            f_wvmr = interp1d(h[mask], wvmr[mask], kind="linear", bounds_error=False, fill_value=np.nan)
            
            # Interpolate
            time_ip = f_time(gridheight)
            wvmr_ip = f_wvmr(gridheight)
        
            # Convert numeric → datetime64, 'NaT' if interpolation is invalid
            time_ip_dt = np.array([np.datetime64(int(ti), "ns") if not np.isnan(ti) 
                                   else np.datetime64("NaT")
                                   for ti in time_ip])
            
            # Apply minimum measured height cutoff
            min_h_rs = np.nanmin(h)
            wvmr_ip = np.where(gridheight >= min_h_rs, wvmr_ip, np.nan)
            time_ip_dt = np.array([
                np.datetime64(int(ti), "ns") if (not np.isnan(ti) and hi >= min_h_rs)
                else np.datetime64("NaT")
                for ti, hi in zip(time_ip, gridheight)
            ])
        
            # Launch metadata
            launchtime = np.datetime64(rs_L.time.min().values, 'ns')
            endtime    = np.datetime64(rs_L.time.max().values, 'ns')
            flighttime = endtime - launchtime # Compute the elapsed time in seconds
            flight_s = flighttime / np.timedelta64(1, "s")
            flight_min = flighttime / np.timedelta64(1, "m") 
        
            # Store results
            all_wvmr.append(wvmr_ip)
            all_time.append(time_ip_dt)
            all_launchtime.append(launchtime)
            all_endtime.append(endtime)
            all_flight_s.append(flight_s)
            all_flight_min.append(flight_min)
            
        # Convert lists → arrays
        all_wvmr = np.array(all_wvmr)
        all_time = np.array(all_time)
            
        # Build final dataset
        ds = xr.Dataset(
            {
                "wvmr": (["launch", "height"], all_wvmr),
                "time": (["launch", "height"], all_time),
                "flighttime_s": (["launch"], np.array(all_flight_s)),
                "flighttime_min": (["launch"], np.array(all_flight_min)),
                "endtime": (["launch"], np.array(all_endtime)),
            },
            coords={
                "launch": launches,
                "height": gridheight,
                "launchtime": (["launch"], np.array(all_launchtime)),
            },
        )
        
        return ds

    rs = interpolate_raso(rs, gridheight)
    
    # --- Interpolate Stations on gridime
    ws_min = 0.0
    ws_max = 30.0

    ws = ws.where((ws.mr >= ws_min) & (ws.mr <= ws_max), drop=True)
    ws = ws.interp(time=gridtime, method="nearest")

    # --- Combine data to one Data Array
    
    # Lidar as baseline:
    data = da['water_vapor'].copy() #.to_numpy()  # shape (time, height)
    
    # - Overwrite with TAWES data
    band_mask = (gridheight >= 0) & (gridheight <= tawesbandmeter)
    data.loc[dict(height=gridheight[band_mask])] = (ws.mr.values[:, None])

    # - Overwrite with Radiosonde data 
    for L in rs.launch.values:
        print(L)
        rs_i = rs.sel(launch=L)
        
        rs_launch = rs_launch = rs_i.launchtime.values.astype('datetime64[m]')
        rs_end = rs_launch + np.timedelta64(rsbandminutes, 'm')
        time_mask = (gridtime >= rs_launch) & (gridtime <= rs_end)
        
        valid_h = ~np.isnan(rs_i.wvmr.values)
        data.loc[dict(time=gridtime[time_mask], height=gridheight[valid_h])] = rs_i.wvmr.values[valid_h]

        #data.loc[dict(time=gridtime[time_mask])] = (rs_i.wvmr.values[None, :])
        
    return data, rs, da

def curtain_plot(date):
    
    data, rs, da = combine_data_on_date(date)
    
    start = np.datetime64(date)
    end = np.datetime64(date + timedelta(days=1))
    
    vmin=2
    vmax=13
    fontsize=17
    
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    wvmr = data.values.transpose()
    wvmr = np.ma.masked_invalid(wvmr)
    
    
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    # Note: pcolormesh expects the arrays of cell-edge coordinates.
    pcm = ax.pcolormesh(t, h, wvmr, vmin=vmin, vmax=vmax, shading="auto", cmap='Blues') #colormaps.cmap_adv_div_brown_green())

    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.set_label(r'wvmr (kg$^{-1}$)', size=fontsize)
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    
    # Plot valid range line
    if not cut: 
        ax.plot(time.values, da.water_vapor_max_range.values/1000, linestyle="-", color="gray", linewidth="2")
        
    for L in rs.launchtime.values:

        rs_end = L+ np.timedelta64(rsbandminutes, 'm')
        ax.axvline(x=L, color='black', linewidth=1.3)
        ax.axvline(x=rs_end, color='black', linewidth=1.3)
        
        # Add "AWS" text next to the vertical radiosonde line.
        ax.annotate("rs",
                xy=(mdates.date2num(rs_end) + 0.003, 0.295),
                xytext=(mdates.date2num(rs_end) + 0.025, 0.200),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=18, color='black', ha='left', va='bottom')

    # Add "AWS" text next to the horizontal line.
    ax.axhline(y=tawesbandmeter/1000, color='black', linewidth=1.3)
    ax.annotate("AWS",
                xy=(t[-40], 0.131),
                xytext=(t[-40], 0.131 + 0.250),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=15, color='black', ha='center', va='bottom')
    
    ax.set_xlim([start, end])
    ax.set_ylim([0, h.max()])
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.tick_params(direction='out', labelsize=fontsize)

    ax.set_xlabel("time (UTC)", fontsize=fontsize)
    ax.set_ylabel("height (km AGL)", fontsize=fontsize)
    ax.set_title(f"DA10: {date.date()}", fontsize=fontsize)
    
    # Background color for NaN values
    ax.set_facecolor([0.9, 0.9, 0.9])
    fig.patch.set_facecolor([1, 1, 1])
    fig.patch.set_alpha(1.0)
    
    fig.tight_layout()
    plt.show()

    return fig

date_beg, date_end = datetime(2024, 8, 22), datetime(2024, 9, 8)
dates = [date_beg + timedelta(days=x) for x in range((date_end - date_beg).days + 1)]
for date in dates: 
    fig = curtain_plot(date)
    
    if cut: 
        filename = f"DA10_cut_{date.strftime('%Y-%m-%d')}_{hmax}.png"
    else:
        filename = f"DA10_{date.strftime('%Y-%m-%d')}_{hmax}.png"
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "DA10_wvmr", "CurtainFusion")
    savefig(fig, folderpath, filename)
    #plt.clf()
