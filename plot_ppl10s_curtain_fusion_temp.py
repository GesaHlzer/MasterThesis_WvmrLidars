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

from colormaps import cmap_bluered40
from basic_plot_funcions import grid_edges, savefig

date  = datetime(2024, 8, 28)

hmax=10000
tawesbandmeter = 300 # m
# hmax = 4000 # meter
# tawesbandmeter = 140 # m
# hmax=3000
# tawesbandmeter = 120 # m

dh   = 10   # meter
dt   = 60   # sec

fig_size = [15, 5]
rsbandminutes = 18 # min 

# --- Load data
rsondes  = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','radiosondes.nc'))
stations = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','stationdata.nc'))
ppl10s   = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()),'data','ramanlidar_10s_filtered.nc'))

def combine_data_on_date(date): #, rsondes, stations, da10):
    
    # Select timeperiod 
    start = np.datetime64(date)
    end   = np.datetime64(date + timedelta(days=1) + timedelta(seconds=1))
    
    pl = ppl10s.sel(time=~ppl10s.time.to_index().duplicated())
    pl = pl.sortby("time")
    pl = pl.sel(time=slice(start, end))
    ws = stations.sel(time=slice(start, end))
    rs = rsondes.sel(launch=rsondes.date == np.datetime64(date))
 
    # Unify 'height' coordinates with 577 m ASL as ground reference
    # da['height'] in AGL wit ground level 571 
    ws['height']  = ws['altitude'] - 577         # 0 -> 571 m ASL
    rs['height']  = rs['altitude'] - 577         # 0 -> 571 m ASL
    pl['height']  = pl['height']   - 3           # 574 -> 571 m ASL

    # --- Drop unnecassary data 
    
    # Mask da10 water-vapor above water_vapor_max_range
    pl  =  pl[['temp']]

    rs = rs[['time', 'height', 't']] #rs[['time', 'height', 'mr', 't', 'p', 'lon', 'lat', 'launch', 'date', 'day_night']]
    rs = rs.where(rs.height <= hmax+10, drop=True)
    
    ws = ws.sel(station='Tawes', drop=True)
    ws = ws[['time', 'height', 't']] 
    
    
    # --- define a fixed height & time grid
    
    gridheight = np.arange(0, hmax+1 , dh, dtype='int64')
    gridtime = np.arange(start, end, np.timedelta64(dt, 's'), dtype='datetime64[ns]')
    
    
    # --- Interpolate Lidar data on gridheight and -time
    
    # make sure all times are unique and handle NaT
    vals_pl, idx_pl = np.unique(pl['time'].values, return_index=True)
    pl = pl.isel(time=np.sort(idx_pl))
    pl =  pl.interp({'height': gridheight, 'time': gridtime}, method='linear')
    
    
    # --- Interpolate Raso on gridheight
         
    def interpolate_raso(rs, gridheight):
        
        launches = rs.launch.values
        all_temp = []
        all_time = []
        all_launchtime = []
        all_endtime = []
        all_flight_s = [] 
        all_flight_min = []
        
        for L in launches:
            
            rs_L = rs.sel(launch=L)
            rs_L = rs_L.dropna(dim="index", subset=["t", 'time', 'height'])
            
            # Extract 1D arrays for this launch
            h = rs_L.height.values
            t = rs_L.time.values.astype("datetime64[ns]").astype(float)
            temp = rs_L.t.values
    
            # Mask invalid points
            mask = np.isfinite(h) & np.isfinite(t) & np.isfinite(temp)
        
            # Build interpolators
            f_time = interp1d(h[mask], t[mask], kind="linear", bounds_error=False, fill_value=np.nan)
            f_temp = interp1d(h[mask], temp[mask], kind="linear", bounds_error=False, fill_value=np.nan)
            
            # Interpolate
            time_ip = f_time(gridheight)
            temp_ip = f_temp(gridheight)
        
            # Convert numeric → datetime64, 'NaT' if interpolation is invalid
            time_ip_dt = np.array([np.datetime64(int(ti), "ns") if not np.isnan(ti) 
                                   else np.datetime64("NaT")
                                   for ti in time_ip])
            
            # Apply minimum measured height cutoff
            min_h_rs = np.nanmin(h)
            temp_ip = np.where(gridheight >= min_h_rs, temp_ip, np.nan)
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
            all_temp.append(temp_ip)
            all_time.append(time_ip_dt)
            all_launchtime.append(launchtime)
            all_endtime.append(endtime)
            all_flight_s.append(flight_s)
            all_flight_min.append(flight_min)
            
        # Convert lists → arrays
        all_temp = np.array(all_temp)
        all_time = np.array(all_time)
            
        # Build final dataset
        ds = xr.Dataset(
            {
                "temp": (["launch", "height"], all_temp),
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
    rs['temp'] = rs['temp'] + 273.15 # °C -> K
    
    # --- Interpolate Stations on gridime
    ws['t'] = ws['t'] + 273.15 # °C -> K
    ws_min = 200
    ws_max = 350

    ws = ws.where((ws.t >= ws_min) & (ws.t <= ws_max), drop=True)
    ws = ws.interp(time=gridtime, method="nearest")


    # --- Combine data to one Data Array
    
    # Lidar as baseline:
    data = pl['temp'].copy() #.to_numpy()  # shape (time, height)
    
    # - Overwrite with TAWES data
    band_mask = (gridheight >= 0) & (gridheight <= tawesbandmeter)
    data.loc[dict(height=gridheight[band_mask])] = (ws.t.values[:, None])

    # - Overwrite with Radiosonde data 
    for L in rs.launch.values:
        #print(i)
        rs_i = rs.sel(launch=L)
        
        rs_launch = pd.to_datetime(rs_i.launchtime.values.item())
        rs_end = rs_launch + timedelta(minutes=rsbandminutes)
        time_mask = (gridtime >= rs_launch) & (gridtime <= rs_end)
        
        data.loc[dict(time=gridtime[time_mask])] = (rs_i.temp.values[None, :])
        
    return data, rs, pl

def curtain_plot(date):
    
    data, rs, pl = combine_data_on_date(date)
    
    start = np.datetime64(date)
    end = np.datetime64(date + timedelta(days=1))
    
    vmin=250
    vmax=310
    fontsize=17
    
    heights = data['height']
    time = data['time']
    t, h = grid_edges(time, heights)
    
    temp = data.values.transpose()
    temp = np.ma.masked_invalid(temp)
    
    
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    # Note: pcolormesh expects the arrays of cell-edge coordinates.
    pcm = ax.pcolormesh(t, h, temp, vmin=vmin, vmax=vmax, shading="auto", cmap=cmap_bluered40()) #colormaps.cmap_adv_div_brown_green())

    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.set_label(r'temperature (K)', size=fontsize)
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    
    # Plot valid range line
    # ax.plot(time.values, da.water_vapor_max_range.values/1000, linestyle="-", color="gray", linewidth="2")
        
    for L in rs.launchtime.values:

        rs_end = pd.to_datetime(L).to_pydatetime() + timedelta(minutes=rsbandminutes)
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
    ax.set_title(f"PPL (10s): {date.date()}", fontsize=fontsize)
    
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
    filename = f"PPL10s_temp_{hmax}m_{date.strftime('%Y-%m-%d')}.png"
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "PPL_10s_temp", "CurtainFusion")
    savefig(fig, folderpath, filename)

# fig = curtain_plot(date)
# filename = f"PPL10s_{date.strftime('%Y-%m-%d')}_{hmax}m.png"
# folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Timeseries", "PPL_10s_temp", "CurtainFusion")
# savefig(fig, folderpath, filename)
