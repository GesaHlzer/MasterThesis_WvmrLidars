# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 15:59:46 2026

Usful basic functions for plotting that 

@author: alleh
"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

def savefig(fig, folderpath, filename, dpi=300, show=False):
    
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        
    filepath = os.path.join(folderpath, filename)
    
    # Save the figure automatically.
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.clf()
    if show is False:
        plt.close(fig)  # Close the figure after saving to free memory.
     
    return print(f"Saved Figure as {filename}.")

def grid_edges(time, heights): #version of statistic with wvmr lidars
    
    # - For time: 
        
    # OLD VERSION FROM PLOT_TIMESERIES.py
    # t = pd.to_datetime(time)      # convert dial_sel times to datetime.
    # t_num = mdates.date2num(t)
    # if len(t_num) > 1: 
    #     dt = np.diff(t_num).mean() # Estimate dt using the mean difference in time (in days)
    # else: 
    #     dt = 1 / (24 * 60)   # default to one minute  
    
    # t_edges = np.concatenate(([t_num[0] - dt/2], t_num + dt/2))
    # # t_edges =  np.concatenate([t_num[0] - dt, t_num])
    
    t = np.array(time, dtype='datetime64[ns]')  # ensure datetime64 array
    t_num = mdates.date2num(t.astype('M8[ms]').astype(datetime))  # convert to matplotlib float dates
    
    if len(t_num) > 1:
        dt = np.mean(np.diff(t_num))  # average delta in days
    else:
        dt = 1 / (24 * 60)  # default to 1 minute in days
    
    t_edges = np.concatenate(([t_num[0] - dt / 2], t_num + dt / 2))  # center edges around original times

    # - for heights
    
    dh = 2  # assumed resolution in meters
    h_edges = np.concatenate(([heights[0] - dh / 2], heights + dh / 2))
    h_edges = h_edges / 1000  # convert m to km

    return t_edges, h_edges

def load_sun_times(filepath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\SSD_IMGI_SSDundDAEMMERUNG.txt"):
    
    """
    Inport and format the sun file ('SSD_IMGI_SSDundDAEMMERUNG.txt')
    for Innsbruck from     
    https://acinn-ertel.uibk.ac.at/de/toolsindex/sonnenscheindauer/
    (Sonnenscheindauer IMGI UNI Dach (von Jo Vergeiner))
    
    Returns
    -------
    df : adjustet dataframe to work with 

    """
                                                 
    # Night  (Sonne weit unter dem Horizont)               sun elevation < -18°
    # Atronomical Dämmerung (Erste minimale Helligkeit)           -18° to -12° 
    # Nautische Dämmerung   (Horizont wird erkennbar)             −12° to −6° 
    # Bürgerliche Dämmerung (Es wird hell genug zum Sehen)         −6° to  0°
    # Tag    Sonne über dem Horizont)
    
    # effektiv - der astronomische Sonnenaufgang ist oft zu dunkel ist, 
    #         um als „Tag“ zu gelten (Berge, Bewölkung, Streulicht).
    
    
    df = pd.read_csv(filepath, sep='\t', skiprows=4, header=None,
                     names=['day', 'month', 'BeginnDaemmerung', 'astronomischerSonnenaufgang',
                            'effektiverSonnenaufgang', 'effektiverSonnenuntergang', 'astronomischerSonnenuntergang',
                            'EndeDaemmung', 'HelligkeitsDauer', 'astronomischeSonnenScheinDauer', 
                            'effektiveSonnenScheinDauer'])
    
    # df['day']   = df['day'].str.replace('.', '', regex=False).astype(int)
    # df['month'] = df['month'].str.replace('.', '', regex=False).astype(int)
    df['year']  = 2024
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    time_cols = ['BeginnDaemmerung', 'astronomischerSonnenaufgang', 
                 'effektiverSonnenaufgang', 'effektiverSonnenuntergang', 
                 'astronomischerSonnenuntergang', 'EndeDaemmung']
    
    # Convert MEZ → UTC (-1h).
    for col in time_cols:
        df[col] = pd.to_datetime(df['date'].astype(str) + ' ' + df[col]) - pd.Timedelta(hours=1)
        
    df = df.set_index('date')[time_cols]
    #df.set_index('date')
    #df.drop(labels=["day", "month", "year"], axis=1, inplace=True)
    
    #, "month", "year")
    return df
    
def classify_daytime(ds):
    """
    distuingish between different daytimes (sunlight background)
    
    Parameters
    ----------
    ds : dataset that contains the coordinate "time"
    
    Returns
    -------
    daytime_array : 
        xarray of str-flags ('day'/'night'/'twilight') for each time
        
    Example:
        day_class = classify_daytime(ds)
        ds_day = ds.sel(time=day_class == 'day')
        ds_night = ds.sel(time=day_class == 'night')
        ds_twilight = ds.sel(time=day_class == 'twilight')
    """
    sun_times = load_sun_times()
    t = ds.time.values
    times = pd.DatetimeIndex(t)
    dates = times.normalize()  # Mitternacht jedes Zeitschritts
    
    #  Direkt indexieren — wiederholt Tageseinträge für jeden Zeitschritt
    sun_times_adapted = sun_times.loc[dates] # [365, 6] -> [len(t), 6]
    
    begin_daemm   = sun_times_adapted['BeginnDaemmerung'].values
    effekt_aufg   = sun_times_adapted['effektiverSonnenaufgang'].values
    effekt_unterg = sun_times_adapted['effektiverSonnenuntergang'].values
    ende_daemm    = sun_times_adapted['EndeDaemmung'].values
    
    
    is_day = (t >= effekt_aufg) & (t <= effekt_unterg)
    
    is_morning  = (t >= begin_daemm) & (t < effekt_aufg) 
    is_evening  = (t > effekt_unterg) & (t <= ende_daemm)
    is_twilight = is_morning | is_evening
    
    # Vektorisierte Masken
    is_day       = (t >= effekt_aufg) & (t <= effekt_unterg)
    is_twilight  = ((t >= begin_daemm) & (t < effekt_aufg)) | \
                   ((t > effekt_unterg) & (t <= ende_daemm))
    # night = alles andere (default) siehe Code 3 Zeilen weiter
    
    # Labels zusammenbauen
    labels = np.full(len(t), 'night', dtype=object)
    labels[is_twilight] = 'twilight'
    labels[is_day] = 'day'
    
    daytime_array = xr.DataArray(labels, coords={'time': ds.time}, dims=['time'])
    
    return daytime_array

def haversine(lat1, lon1, lat2, lon2):
    # Haversine-Funktion zur Berechnung der Distanz in Metern
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    earth_radius = 6371.0   # in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # haversine formula     
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return earth_radius * c  # Rückgabe in Kilometern 

def classify_daytime_old(data, sel='all'): # Select Daytime Range
    ds = data.copy()
    
    # for the 9th september 2024 (min daylight) in the time period
        # Daylight:  06:45 - 19:35 (CET),                 Total:12:50
    # for the 23th august 2024 (min nighttime, sun at -18° below horizon) in the time period
        # Night: 22:06 - 00:00, 00:00 - 04:25 (CET), Total:06:19
    
    # -> Chose 6 h Period for both, Day: 10-16 CET, Night: 22:15 - 4:15 CET 
    # -> 8-14 UTC & 20:15-2:15 UTC
    
    # extract hour and minute as DataArray
    hour   = ds.time.dt.hour
    minute = ds.time.dt.minute
    
    mask_night = ( ((hour > 20) | ((hour == 20) & (minute >= 15))) |
                   ((hour <  2) | ((hour ==  2) & (minute <= 15)))   
                  )
    
    mask_day   = ( ((hour > 8)  | ((hour == 8)  & (minute >= 0))) & 
                   ((hour < 14) | ((hour == 14) & (minute == 0)))
                  )
    
    ds_night   = ds.where(mask_night, drop=True)
    ds_day     = ds.where(mask_day,   drop=True)
    
    # return ds, ds_day, ds_night
    if sel=='all':   return ds
    if sel=='night': return ds_night
    if sel=='day':   return ds_day

def cmap_windspeed():
    """ Custom colormap: UIBK wind speed """
    
    cmap = np.array([[255, 255, 255],
                    [255, 252, 203],
                    [224, 243, 139],
                    [171, 231, 131],
                    [109, 220, 136],
                    [0, 208, 149],
                    [0, 197, 165],
                    [0, 185, 180],
                    [0, 173, 193],
                    [0, 159, 204],
                    [0, 144, 212],
                    [55, 127, 216],
                    [112, 108, 216],
                    [145, 89, 211],
                    [168, 69, 201],
                    [183, 50, 188],
                    [192, 35, 173],
                    [195, 48, 93],
                    [210, 103, 73],
                    [246, 139, 69],
                    [255, 204, 79]], dtype=np.float32)

    cmap /= 255.0 # Normalize the colormap to [0, 1] range

    return LinearSegmentedColormap.from_list("windspeed", cmap)

