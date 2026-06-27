# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:19:01 2026

@author: alleh
"""

import numpy as np
import xarray as xr
from pathlib import Path
from basic_plot_funcions import haversine
# import os
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from scipy.odr import ODR, Model, RealData

def fuse_DA10_100m(base_dir, version="orig"):
    
    #base_dir = Path(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\DA10-5\2024")
    months   = ["06", "07", "08", "09", "10"]
    nc_files = sorted([ f for month in months
                          for f in (base_dir / month / "netCDF").glob("DA10-5_*_60s_100m_wvmr_profile.nc")
                          ])
    print(f"Found {len(nc_files)} files:")
    # for f in nc_files:
    #     print(" ", f.name)
    
    # Open and concatenate along the time dimension
    ds = xr.open_mfdataset(
        nc_files,
        combine="by_coords",   # uses the time coordinate to sort and concat
        parallel=True,         # reads files in parallel (needs dask)
    )
    
    # Extract from the first timestep and drop the time dimension if same for all time steps:
        
    if np.all(ds.zsl.values == ds.zsl.values[0]):
        ds = ds.assign_coords(zsl=ds["zsl"].isel(time=0).drop_vars("time"))
        
    if np.all(ds.lat.values == ds.lat.values[0]):
        ds = ds.assign_coords(lat=ds["lat"].isel(time=0).drop_vars("time"))
        
    if np.all(ds.lon.values == ds.lon.values[0]):
        ds = ds.assign_coords(lon=ds["lon"].isel(time=0).drop_vars("time"))
        
    if np.all(ds.height_bnds.values == ds.height_bnds.values[0]):
        height_bnds_fixed = ds["height_bnds"].isel(time=0).drop_vars("time")
        ds = ds.drop_vars("height_bnds").assign(height_bnds=height_bnds_fixed)
    
    
    # Fix height coordinate: 0. -> 25.
    new_height = ds.height.values.copy()
    new_height[0] = 25.0
    
    # Fix height_bnds: [-50., 50.] -> [0., 50.]
    new_height_bnds = ds.height_bnds.values.copy()
    new_height_bnds[0, 0] = 0.0
    
    # Assign back
    ds = ds.assign_coords(height=new_height)
    ds["height_bnds"] = xr.DataArray(new_height_bnds, dims=["height", "nv"])
        
    #print("\nMerged dataset:")
    #print(ds)
 
    return ds         

def ppl_100m_bin(file):
    ppl_orig = xr.open_dataset(file)
    # --- Convert altitude to height AGL
    ppl_orig["height"] = ppl_orig["height"] - 3
    ppl_orig["ground_elevation"] = 577
     
    # --- Define bin edges and center                       
    bin_centers = np.concatenate([[25.0], np.arange(100.0, 4001.0, 100)]) # [25, 100, 200, ..., 4000]
    bin_edges = np.concatenate([[0.0], np.arange(50.0, 4051.0, 100)])     # [0, 50, 150, 250, ..., 4050]
    height_bnds = np.column_stack([bin_edges[:-1], bin_edges[1:]])
     
    # --- Bin-average
    h = ppl_orig["height"]
    bin_means = []
    wvmr = ppl_orig["wvmr_filtered"].where(ppl_orig['height'] <= ppl_orig['wvmr_max_range'])
    
    for lower, upper in height_bnds:
        mask = ((h >= lower) & (h < upper))
        wvmr_bin = wvmr.where(mask)
        mean_profile = wvmr_bin.mean(dim="height", skipna=True)
        bin_means.append(mean_profile) if len(wvmr_bin) > 0 else np.nan
    bin_means = xr.concat(bin_means, dim="height")

        
    
    # --- Build output Dataset
    
    data_binned = xr.Dataset({
    "wvmr": (("height", "time"), bin_means.values)
    },
    coords={
        "time":    ppl_orig.coords["time"],
         "height":  bin_centers,
         },
     attrs={"ground_elevation": 577,
            "timestep":         "1200s"
         }
     )
     
    data_binned["height_bnds"] = xr.DataArray(height_bnds, dims=["height", "nv"])

    return data_binned

def match_vertical_all(rsondes, dial_orig, dial_0_88, ppl20m): 
     
    dial    = dial_orig.copy()
    dial088 = dial_0_88.copy()
    ppl     = ppl20m.copy()
    raso    = rsondes.copy()
    
    #  Move timestamps to the middle of the 20 min averaging period 
    dial['time']    = dial_orig['time'] - np.timedelta64(10, 'm')  
    dial088['time'] = dial_0_88['time'] - np.timedelta64(10, 'm')
    ppl['time']     = ppl20m['time'] - np.timedelta64(10, 'm')
    
    # Calculate distance between the lidars and the radiosonde
    distance_rs   =  haversine(dial.lat.item(), dial.lon.item(), raso['lat'].values, raso['lon'].values)
    raso['distance'] = xr.DataArray(distance_rs, dims=['launch', 'height'])
    
    # Select relevant data
    dial = dial.rename({"water_vapor_mean":"wvmr"})
    dial = dial[['wvmr']] 
    
    dial088['wvmr'] = dial088['water_vapor'].where(dial088['height'] <= dial088['water_vapor_max_range_mean'])
    dial088 = dial088[['wvmr']]
    
    ppl = ppl[['wvmr']]
    
    raso = raso.rename({"mean_wvmr": "raso_wvmr"})
    raso = raso[['time', 'height', 'height_bnds', 'raso_wvmr', 'launch','date', 'day_night', 'distance']]
    
    processed_data = []
    for i in raso.launch.values:
        try:
            #i = 1  # choose one radiosonde launch for testing (example i = 73)
            # Select launch and discard all NaNs from the combined dataset
            rs = raso.isel(launch=i) 
            rs = rs.where(rs.time.notnull(), drop=True)
            
            gridtime = rs.time.values
            
            mask_da = ((dial.time >= (gridtime.min() - np.timedelta64(10,  'm'))) & 
                        (dial.time <= (gridtime.max() + np.timedelta64(10,  'm'))))
            mask_pl = ((ppl.time >= (gridtime.min() - np.timedelta64(10,  'm'))) & 
                        (ppl.time <= (gridtime.max() + np.timedelta64(10,  'm'))))
            
            da    = dial.sel(time=mask_da)  
            da088 = dial088.sel(time=mask_da) 
            pl    = ppl.sel(time=mask_pl) 
                       
            # Interpolate at these times for all heights
            dial_interp = da["wvmr"].interp(time=gridtime, method="linear")
            dial_interp = dial_interp.values.diagonal()
            dial088_interp = da088["wvmr"].interp(time=gridtime, method="linear")
            dial088_interp = dial088_interp.values.diagonal()
            
            if pl.sizes["time"] != 0: 
                ppl_interp = pl["wvmr"].interp(time=gridtime, method="linear")
                ppl_interp = ppl_interp.values.diagonal()
            
                ds_new = xr.Dataset({
                   'time'        : (('height'), gridtime),
                   'distance'    : (('height'), rs['distance'].values),
                   'rs_wvmr'     : (('height'), rs['raso_wvmr'].values),
                   'dial_wvmr'   : (('height'), dial_interp),
                   'dial088_wvmr': (('height'), dial088_interp),
                   'rl_wvmr'     : (('height'), ppl_interp),
                   'height_bnds' : (('height', 'nv'), rs['height_bnds'].values),
                   },
                   coords={
                        'height'   : raso['height'].values,
                        'launch'   : rs.launch.item(),
                        'date'     : rs.date.values,
                        'day_night': rs.day_night.item(),
                        },
                    attrs={
                        'elevation': 577, # m ASL
                        })
            else:
                ds_new = xr.Dataset({
                   'time'        : (('height'), gridtime),
                   'distance'    : (('height'), rs['distance'].values),
                   'rs_wvmr'     : (('height'), rs['raso_wvmr'].values),
                   'dial_wvmr'   : (('height'), dial_interp),
                   'dial088_wvmr': (('height'), dial088_interp),
                   'height_bnds' : (('height', 'nv'), rs['height_bnds'].values),
                   },
                   coords={
                        'height'   : raso['height'].values,
                        'launch'   : rs.launch.item(),
                        'date'     : rs.date.values,
                        'day_night': rs.day_night.item(),
                        },
                    attrs={
                        'elevation': 577, # m ASL
                        })
                
            processed_data.append(ds_new)
            
 
        except Exception as e:
            print(f"Error with launch nr {i}: {e}")
            
            
    data = xr.concat(processed_data, dim='launch')
    
    if (data["height_bnds"] == data["height_bnds"].isel(launch=0)).all():
        data = data.assign(height_bnds=data["height_bnds"].isel(launch=0))
    else:
        print("height_bnds differs between launches!")

    return data

def match_dials_all(dial_orig, dial_0_86, dial_0_88, ppl20m, dt=1200):
    
    dial    = dial_orig.copy()
    dial088 = dial_0_88.copy()
    dial086 = dial_0_86.copy()
    ppl     = ppl20m.copy()
    
    # #  Move timestamps to the middle of the 20 min averaging period 
    # dial['time']    = dial_orig['time'] - np.timedelta64(10, 'm')  
    # dial088['time'] = dial_0_88['time'] - np.timedelta64(10, 'm')
    # ppl['time']     = ppl20m['time'] - np.timedelta64(10, 'm')
    
    # --- Select measurement period of PPL
    start = np.datetime64('2024-08-22T23:00')
    end   = np.datetime64('2024-09-09T01:00')
    
    ppl     =  ppl   .sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))
    dial    = dial   .sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm'))) 
    dial088 = dial088.sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))     
    dial086 = dial086.sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))     

    # --- Define new grid 
    gridtime = np.arange(start, end, np.timedelta64(dt, 's'), dtype='datetime64[ns]')
    gridheight = dial.height.values
    height_bnds  = dial.height_bnds.values
    
    # Select relevant data
    dial = dial.rename({"water_vapor_mean":"wvmr"})
    dial = dial[['wvmr']] 
    dial088['wvmr'] = dial088['water_vapor'].where(dial088['height'] <= dial088['water_vapor_max_range_mean'])
    dial088 = dial088[['wvmr']]
    dial086['wvmr'] = dial086['water_vapor'].where(dial086['height'] <= dial086['water_vapor_max_range_mean'])
    dial086 = dial086[['wvmr']]
    ppl = ppl[['wvmr']]
    
    # make sure all times are unique and handle NaT
    vals_dial,   idx_dial      = np.unique(dial['time'].values, return_index=True)
    dial                       = dial.isel(time=np.sort(idx_dial))
    vals_dial088, idx_dial088  = np.unique(dial088['time'].values, return_index=True)
    dial088                    = dial088.isel(time=np.sort(idx_dial088))
    vals_dial086, idx_dial086  = np.unique(dial086['time'].values, return_index=True)
    dial086                    = dial086.isel(time=np.sort(idx_dial086))
    vals_ppl,   idx_ppl        = np.unique(ppl['time'].values, return_index=True)
    ppl                        = ppl.isel(time=np.sort(idx_ppl))
    
    # --- Interpolate raso, dial & ppl on the height & time grid 
    da    =  dial.interp({'time': gridtime}, method='linear')
    da088 =  dial088.interp({ 'time': gridtime}, method='linear')
    da086 =  dial086.interp({ 'time': gridtime}, method='linear')
    pl    =  ppl.interp({'time': gridtime}, method='linear')
    
    # Asemble in Dataset
    
    data = xr.Dataset({
        'dial_wvmr'    : (('time','height'), da.wvmr.values),
        'dial088_wvmr' : (('time','height'), da088.wvmr.values),
        'dial086_wvmr' : (('time','height'), da086.wvmr.values),
        'rl_wvmr'      : (('time','height'), pl.wvmr.values.T),
        'height_bnds'  : (('height', 'nv'), height_bnds),
        }, 
        coords={
            'height': gridheight,
            'time'  : gridtime,
        },
        attrs={
            'longnitude_lidars': dial.longitude.item(),
            'latitude_lidars': dial.latitude.item(),
            'elevation': 577, # m a.m.s.l.
        })

    return data

rsondes   = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\radiosondes_ibk_binned.nc")
dial_orig = fuse_DA10_100m(Path(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\DA10-5\2024"))
dial_0_88 = fuse_DA10_100m(Path(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\post_processed_profiles\sim_calibration_0_88\DA10-5\2024"))
dial_0_86 = fuse_DA10_100m(Path(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\post_processed_profiles\sim_calibration_0_86\DA10-5\2024"))
ppl20m    = ppl_100m_bin(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc")

# data_1d = match_vertical_all(rsondes, dial_orig, dial_0_86, ppl20m)
# data_1d.to_netcdf(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1d_all_with_CorrectedDial086.nc")

# data_1d = match_vertical_all(rsondes, dial_orig, dial_0_88, ppl20m)
# data_1d.to_netcdf(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1d_all_with_CorrectedDial088.nc")

data_2d = match_dials_all(dial_orig, dial_0_86, dial_0_88, ppl20m, dt=1200)
data_2d.to_netcdf(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2d_all_with_CorrectedDial.nc")
