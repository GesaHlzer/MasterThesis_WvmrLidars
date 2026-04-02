# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 19:19:00 2026

@author: alleh
"""

import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from basic_plot_funcions import savefig

# Settings
Save          = True
Stations      = True
Hmax          = 3


# rs_list = read_raso_csv(date_beg, date_end)
# # ppl = xr.concat(read_ppl(date_beg, date_end, var='MR', avg_time='900s'), dim="time")
# dial_filename = os.path.join(os.path.dirname(os.getcwd()), 'data', 'DA10', 'dial_MR_2024-06-18_2024-10-22.nc')
# dial = xr.open_dataset(dial_filename)

# --- Load data

path = os.path.join(os.path.dirname(os.getcwd()), 'data', '1d_vertical_profiles.nc')
data = xr.open_dataset(path)




def plot_vertical_dial_raso(data_sel, Save, Stations, Hmax):   
    
    ds = data_sel.copy()
    ds['wvmr_rl2'] = xr.where(data.height < 0.1, np.nan, ds['wvmr_rl2'])
    ds['wvmr_aws'] = xr.where(ds['wvmr_aws'] < 0, np.nan, ds['wvmr_aws']) 
    Fontsize = 25 #22 #25
    Hmax=12#3 #12
    # figtitle = f'{ds.date.astype(str).item()[:10]} ({ds.day_night.values})'
    # figtitle = f'{ds.date.values[0].astype("datetime64[D]")} (02 UTC)'
    figtitle = f'Water vapor mixin ratio {ds.date.astype(str).item()[:10]} ({ds.day_night.values})'

    fig, ax = plt.subplots(figsize=(10, 15)) #10,15 #10,25 
    fig.suptitle(figtitle, fontsize=Fontsize+4)# +2
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(ds['wvmr_rs'].values.flatten(),  ds['height'],label="Radiosonde",c='black',     linewidth=3)
    ax.plot(ds['wvmr_rl2'].values.flatten(), ds['height'],label="PPL-1200s", c='dodgerblue',linewidth=3)
    ax.plot(ds['wvmr_dial'].values.flatten(),ds['height'],label="DA10",      c='darkorange',linewidth=3)
    
    if Stations: 
        valid = ~np.isnan(ds['wvmr_aws']) & ~np.isnan(ds['height_aws'])
        ax.plot(ds['wvmr_aws'].values[valid], ds['height_aws'].values[valid], marker='o', label="AWSs",
                c='black', linestyle=':', linewidth=2, markersize=8)
    ax.set_xlabel(r"water vapor mixing ratio (g kg$^{-1}$)", fontsize=Fontsize)
    ax.set_ylabel("height (km AGL)",                 fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize)
    ax.set_ylim(0, Hmax)
    ax.set_xlim(0, 14) #18.
    ax.grid(True)
     
    #- Create a second x-axis showing horizontal distance.
    ax2 = ax.twiny()
    ax2.set_xlabel("distance (km)", c = 'silver', fontsize=Fontsize)
    ax2.tick_params(axis='x', colors='silver', labelsize=Fontsize)
    ax2.set_xlim(0, 14)
     
    ax2.plot(ds['distance_rs'].values.flatten(), ds['height'], 
             label="Distance between RS and Lidars", 
             color='gray', linestyle=':', linewidth=3, alpha=0.3)
    if Stations: 
        ax2.plot(ds['distance_aws'].values[valid], ds['height_aws'].values[valid], 'o',
                 label="Distance between AWSs and Lidars",
                 color='gray', linestyle=':', ms=7, linewidth=3, alpha=0.3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1[:4] + lines2[:2], labels1[:4] + labels2[:2], fontsize=Fontsize-2, loc= 'upper right')
    
    # - Label the crest height
    ax.text(0.06, 1.700+Hmax/250, 'crest', fontsize=Fontsize, color="gray", alpha=0.5)
    ax.axhline(y=1.700, color='slategray', linestyle='--', linewidth=1.1, alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.933) #915 #933
    
    if Save:
        folderpath = os.path.join(os.path.dirname(os.getcwd()),
                                  'plots','VerticalPlots',
                                  'WVMR')
        if ds.day_night.values == 'day':
            filename = f'{ds.date.values[0].astype("datetime64[D]")}T12'
        elif ds.day_night.values == 'night':
            filename = f'{ds.date.values[0].astype("datetime64[D]")}T02'
        filename = f'1_{filename}_to{Hmax}km.png'   
        # filename = f'all_to_{Hmax}km' + filename + '_filtered.png'
        savefig(fig, folderpath, filename)
    
    plt.show()
    

# data['height'] = data['height'] / 1000
# data['height_aws'] = data['height_aws'] / 1000
# data = data.drop_sel({'station': 'Olympisch'})

#--- Plot all    
for i in data.launch.values:
    # i=87
    ds = data.sel(launch=i)
    
    print('\n',
          'Plotting... ',
          "Launch label:", i, 
          "Date:", ds.date.astype(str).item()[:10], 
          "When:", ds.day_night.values)
    
    ds = ds.drop_sel({'station': 'Olympisch'})
    
    plot_vertical_dial_raso(ds, Save, Stations, Hmax)
    
    
#--- Plot Selected

# date_beg = np.datetime64("2024-08-24")
# date_end = np.datetime64("2024-08-24")
# date = np.datetime64("2024-08-24")
           
# data_day = data.sel(launch=data.day_night=='day')   # day_sondes
# data_night = data.sel(launch=data.day_night=='night') # night_sondes
# data = data.where(data['date'] == date, drop=True)
                  
# plot_filtered_mr(data_night, Save, Stations, Hmax)  
    