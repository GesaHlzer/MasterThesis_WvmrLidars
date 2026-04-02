# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:58:03 2026

@author: alleh
"""

import xarray as xr
from datetime import datetime
import os
import pandas as pd
import numpy as np
import glob

from metpy.units import units
import metpy.calc as mpcalc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from basic_plot_funcions import savefig

def open_wyoming_csv_raso(csvfile):
        #csvfile = os.path.join(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\raso\{date.strftime('%Y%m%d%H')}-11120.csv")
        ds_wyoming = pd.read_csv(csvfile, delimiter=',', skipinitialspace=True)
        ds_wyoming = ds_wyoming.set_index('time') 
        ds_wyoming = ds_wyoming.to_xarray()
        ds_wyoming['time'] = ds_wyoming['time'].astype('datetime64[ns]')
        ds_wyoming = ds_wyoming.rename({'pressure': 'air_pressure',
                                        'geopotential height': 'geopotential_height',
                                        'temperature': 'air_temperature',
                                        # latitude already 'latitude'
                                        # longitude already 'longitude'
                                        'dew point temperature': 'dew_point_temperature',
                                        'wind direction': 'wind_from_direction',
                                        'wind speed': 'wind_speed',
                                        'relative humidity': 'relative_humidity',
                                        'mixing ratio': 'water_vapor_mixing_ratio',
                                        #'ice point temperature': 'ice_point_temperature',
                                        #'humidity wrt ice': 'humidity_wrt_ice',
                                        })
        # ds_wyoming["dew_point_temperature"] = ds_wyoming["dew_point_temperature"] + 273.15 # degC -> K
        # ds_wyoming['air_temperature'] = ds_wyoming['air_temperature'] + 273.15 # degC -> K
        # ds_wyoming['air_pressure'] = ds_wyoming['air_pressure'] * 100 # hPa -> Pa
        # ds_wyoming = ds_wyoming.drop_vars(['humidity wrt ice', 'ice point temperature'])
        ds_wyoming["height"] = (6356766 * ds_wyoming['geopotential_height']) / (6356766 - ds_wyoming['geopotential_height'] )
        
        ds_wyoming.attrs['source'] = 'Wyoming CSV files'
        ds_wyoming.attrs['serial_number'] = 'Wyoming-' + os.path.basename(csvfile).split('.')[0]
        ds_wyoming.attrs['station_nr'] = csvfile.split('-')[1].split('.')[0]
        ds_wyoming.attrs['creation_date'] = ds_wyoming['time'].values[0]
        ds_wyoming.attrs['start_latitude'] = ds_wyoming['latitude'].values[0]
        ds_wyoming.attrs['start_longitude'] = ds_wyoming['longitude'].values[0]
    
        return ds_wyoming

def open_compare_file(comparefile):
    #comparefile = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\compare\DA10-5_compare2024082313.csv"
    ds = pd.read_csv(comparefile, delimiter=',', skipinitialspace=True)
    ds = ds.iloc[:, :5]
    ds.columns = ["height", "raso_wvmr", "dial_wvmr", "distance", "difference"]
    ds = ds.set_index('height') 
    ds = ds.to_xarray()
    return ds

def vertical_plot(raso, dial, dial2, dial3, date):
     
    # --- Select Closest DIAL Timestamp  for each RS
    
    #filter raso below 4050 m
    raso['height_agl'] = raso['height'] - 577
    raso_low = raso.where(raso.height_agl < 4050, drop=True)
    
    # match dial to raso timestamps
    matched_dial = dial.sel(time=raso_low.time.values[1], method="nearest")
    
    # Only heights below or equal to the valid range of MR determination
    valid_mask = matched_dial['height'] <= matched_dial.water_vapor_max_range.values
    matched_dial = matched_dial.where(valid_mask, drop=True)
    
    # match dial to raso timestamps
    matched_dial2 = dial2.sel(time=raso_low.time.values[1], method="nearest")
     
    # # create a fine height grid
    # dz = 2.4 
    # height_grid = np.arange(np.nanmin(dial_heights), np.nanmax(dial_heights)+dz, dz)
    
    # # height grid of datasets
    # rs_heights = rs_ds['heightAGL'].values.astype(float)
    # rs_mr = rs_ds['mr'].values
   
    # # Ensure that the heights are sorted in rows         
    # sorted_idx_rs = np.argsort(rs_heights)
    # rs_heights_sorted = rs_heights[sorted_idx_rs]
    # rs_mr_sorted = rs_mr[sorted_idx_rs]
    
    # # Generate interpolation functions (linear, with NaN outside the measured range)
    # interp_rs = interp1d(rs_heights_sorted, rs_mr_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
    # rs_mr_new= interp_rs(height_grid)
    
    # # Select rs_mr_interp values corresponding to heights on the dial height grid
    # mask = np.isclose(height_grid[:, None], dial_heights, atol=1e-6)
    # mask = np.any(mask, axis=1)
    # rs_mr_new = rs_mr_new[mask]
    
    # # Select dial MR for valid heights
    # dial_mr = dial_ds['water_vapor'].where(valid_mask, drop=True).values
    # dial_mr_uncertainty = dial_ds['water_vapor_uncertainty'].where(valid_mask, drop=True).values
    
    # rs_mr_interp.append(rs_mr_new)
    # dial_mr_interp.append(dial_mr)
    # dial_mr_uncert_interp.append(dial_mr_uncertainty)
    # heights.append(dial_heights)
        
    Fontsize = 25 #22 #25
    Hmax=4#3 #12
    
    fig, ax = plt.subplots(figsize=(10, 15)) #10,15 #10,25 
    fig.suptitle(f'{date.strftime("%Y%m%d UTC%H")}', fontsize=Fontsize+4)# +2
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(matched_dial.water_vapor, matched_dial.height, label="DA10 orig", c='darkorange',linewidth=3)
    ax.plot(matched_dial2.water_vapor_mean, matched_dial2.height, label="DA10 100m mean", c='red',linewidth=3)
    ax.plot(raso_low.water_vapor_mixing_ratio, raso_low.height_agl, label="Raso", c='black',linewidth=3)
    ax.plot(dial3.dial_wvmr, dial3.height, label="DA10 comparefile", c='purple',linewidth=3)
    
    ax.set_xlabel(r"water vapor mixing ratio (g kg$^{-1}$)", fontsize=Fontsize)
    ax.set_ylabel("height (m AGL)",                 fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize)
    ax.set_ylim(0, Hmax* 1000)
    ax.set_xlim(0, 18) #18.
    ax.grid(True)
    ax.legend(fontsize=Fontsize-2, loc= 'upper right')
    
    # - Label the crest height
    ax.text(0.06, 1.700+Hmax/250, 'crest', fontsize=Fontsize, color="gray", alpha=0.5)
    ax.axhline(y=1.700, color='slategray', linestyle='--', linewidth=1.1, alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.933) #915 #933
    
    fig.show()
    
    return fig


# date = datetime(2024,8,23,2)
#csvfile = os.path.join(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\raso\{date.strftime('%Y%m%d%H')}-11120.csv")
# raso = open_wyoming_csv_raso(csvfile)

# dial = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
# dial = dial.sel(time=dial.time.dt.floor('D') == np.datetime64(date.date()))

# dial2 = xr.open_dataset(fr"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\Level2\uncalibrated\DA10-5\2024\{date.strftime('%m')}\netCDF\DA10-5_{date.strftime('%Y%m%d')}_60s_100m_wvmr_profile.nc")

#dial3 = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\compare\DA10-5_compare2024082313.csv"

# fig = vertical_plot(raso, dial, dial2, date)
# folderpath = os.path.join(os.path.dirname(os.getcwd()),'plots','VerticalPlots','wvmr_test_data')
# filename = f'dial1_vertical_{date.strftime("%Y%m%d_T%H")}_to{Hmax}km.png'
# # savefig(fig, folderpath, filename)

raso_files = sorted(glob.glob("C:/Users/alleh/Documents/+Uni_Innsbruck/+MasterThesis/data/raso/*.csv"))
raso_files = raso_files[70:90]

for csvfile in raso_files:
    
    #csvfile = raso_files[73]
    date = datetime.strptime(csvfile[-20:-10], '%Y%m%d%H')

    # --- Load data
    raso = open_wyoming_csv_raso(csvfile)

    dial = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
    dial = dial.sel(time=dial.time.dt.floor('D') == np.datetime64(date.date()))
    
    dial2 = xr.open_dataset(fr"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\Level2\uncalibrated\DA10-5\2024\{date.strftime('%m')}\netCDF\DA10-5_{date.strftime('%Y%m%d')}_60s_100m_wvmr_profile.nc")
    
    comparefile = fr"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\KIT_hiwi\compare\DA10-5_compare{date.strftime('%Y%m%d%H')}.csv"
    dial3 = open_compare_file(comparefile)
    
    fig = vertical_plot(raso, dial, dial2, dial3, date)
    folderpath = os.path.join(os.path.dirname(os.getcwd()),'KIT_hiwi','wvmr_test_data')
    filename = f'dial_test_vertical_{date.strftime("%Y%m%d_T%H")}.png'
    savefig(fig, folderpath, filename)

# # Select dial MR for valid heights
# dial_mr = dial_ds['water_vapor'].where(valid_mask, drop=True).values
# dial_mr_uncertainty = dial_ds['water_vapor_uncertainty'].where(valid_mask, drop=True).values

# rs_ds['heightAGL'] = rs_ds.height.values - dial_ds.elevation.values



        
    
