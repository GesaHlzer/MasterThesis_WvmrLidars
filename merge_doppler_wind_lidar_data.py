# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:29:00 2026

@author: alleh
"""


import os
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime, timezone, timedelta 

def read_sl88_stare(start, end): #08-07 to 09-09
    """
    start : pd.Timestamp   – exakter Startzeitpunkt
    end   : pd.Timestamp   – exakter Endzeitpunkt
    """

    # Dates from Timestamps
    date_start = start.date()
    date_end   = end.date()
    date_range = pd.date_range(date_start, date_end, freq='D')

    # ---- SETTINGS
    
    # specify window size for moving average and moving variance 
    window_size = 5160  # number of datapoints that are approx 1 h # 3601  # seconds = 1 hour
                                                            #  2580 # 1801  # seconds = 0.5 hour 
    # specify data cleaning options 
    apply_deltavrad_threshold = 'yes' # 'no'
    apply_vrad_threshold = 'yes' 
    apply_int_threshold = 'yes' 
    apply_nan_threshold = 'yes' 
    apply_vradmean_threshold = 'yes' 
    apply_vradvar_threshold = 'yes'
    
    # ---- read all SL88 files within given date/time range
    base_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'SL88', 'SL88_stare')
    
    all_datasets = []
    count_duplicates = 0
   
    for date in date_range:
        ncdir = os.path.join(base_dir, date_start.strftime('%Y%m%d'))
        
        if not os.path.isdir(ncdir):
            print(f"Folder not found, skip dir: {ncdir}")
            continue
        
        files = sorted([os.path.join(ncdir, f) for f in os.listdir(ncdir) if f.endswith('.nc')])
        
        for f in files:
            ds = xr.open_dataset(f)

            decimal_hours   = ds["decimal_time"].values
            datetime_coords = pd.Timestamp(date.date()) + pd.to_timedelta(decimal_hours, unit="h")

            ds = ds.rename({'gate_centers': 'height'})
            ds = ds.swap_dims({"NUMBER_OF_GATES": "height"})
            ds = ds.assign_coords(time=("NUMBER_OF_RAYS", datetime_coords))
            ds = ds.swap_dims({"NUMBER_OF_RAYS": "time"})

            time_index     = ds.get_index("time")
            duplicated_mask = time_index.duplicated(keep="first")
            count_duplicates += duplicated_mask.sum()
            ds = ds.isel(time=~duplicated_mask)

            all_datasets.append(ds)
    
    if not all_datasets:
        raise ValueError("No Data found in time period.")

    
    if count_duplicates > 0:
        print(f"Removed {count_duplicates} duplicate time entries in SL88 stare data.")
        
    ds = xr.concat(all_datasets, dim = 'time')
    ds = ds.sel(time=((ds.time >= start) & (ds.time <= end)))
    
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

def read_slxr142(start, end):
    """
    start : pd.Timestamp   – exakter Startzeitpunkt
    end   : pd.Timestamp   – exakter Endzeitpunkt
    """
    
    base_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'SLXR142')

    date_range = pd.date_range(start.date(), end.date(), freq='D')
    all_datasets = []

    for date in date_range:
        ncdir  = os.path.join(base_dir, date.strftime('%Y%m'))
        ncfile = os.path.join(ncdir, date.strftime('%Y%m%d') + '.nc')

        if not os.path.isfile(ncfile):
            print(f"File not found, skip: {ncfile}")
            continue

        nc = netCDF4.Dataset(ncfile, 'r')

        height = nc.variables['height'][:]
        time   = nc.variables['time'][:-1]
        ff     = nc.variables['ff'][:, :-1]
        dd     = nc.variables['dd'][:, :-1]
        intensity = nc.variables['intensity'][:, :-1]

        datetimes = np.array([
            datetime.fromtimestamp(float(t))#, tz=timezone.utc)
            if not np.ma.is_masked(t) and not np.isnan(t)
            else None
            for t in time
        ])

        ff        = np.array(ff)
        dd        = np.array(dd)
        intensity = np.array(intensity)

        u_wind = -ff * np.sin(np.radians(dd))
        v_wind = -ff * np.cos(np.radians(dd))

        threshold_indices = intensity < 1.0045
        ff[threshold_indices] = np.nan
        dd[threshold_indices] = np.nan

        ds = xr.Dataset(
            {
                'ff':        (['height', 'time'], ff),
                'dd':        (['height', 'time'], dd),
                'u_wind':    (['height', 'time'], u_wind),
                'v_wind':    (['height', 'time'], v_wind),
                'intensity': (['height', 'time'], intensity),
            },
            coords={'time': datetimes, 'height': height}
        )

        all_datasets.append(ds)
        nc.close()

    if not all_datasets:
        raise ValueError("No data in period.")

    ds = xr.concat(all_datasets, dim='time')
    ds = ds.sel(time=((ds.time >= start) & (ds.time <= end)))


    return ds

if __name__ ==  "__main__":
    
    # # --- SL88 vertical stare
    # start = pd.Timestamp("2024-08-07 00:00")
    # end   = pd.Timestamp("2024-09-09 00:00")
    # ds = read_sl88_stare(start, end)
    # save_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'SL88', 'SL88_stare')
    # filename = f"SL88_stare_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}.nc"
    
    # --- SLXR142
    start = pd.Timestamp("2024-07-18 00:00")
    end   = pd.Timestamp("2024-10-23 00:00")
    ds = read_slxr142(start, end)
    save_dir  = os.path.join(os.path.dirname(os.getcwd()), 'data', 'SLXR142')
    filename  = f"SLXR142_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}.nc"
    
    
    save_path = os.path.join(save_dir, filename)
    ds.to_netcdf(save_path)
    print(f"saved {save_path}")