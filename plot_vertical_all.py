

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 00:18:52 2025

@author: alleh
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from basic_plot_funcions import savefig
   
   
def plot_wvmr(data_sel, Save, Stations, Hmax):   
    
    Fontsize = 22
    
    ds = data_sel.copy()
    ds['height'] = ds['height'] / 1000
    ds['aws_height'] = ds['aws_height'] / 1000
    ds = ds.drop_sel({'station': 'Olympisch'})
    
    #    ds['rl2_wvmr'] = xr.where(data.height < 0.1, np.nan, ds['rl2_wvmr'])
    ds['aws_wvmr'] = xr.where(ds['aws_wvmr']<0, np.nan, ds['aws_wvmr']) 
    
    #nd = ds.day_night.values
    figtitle = f'{ds.date.astype(str).item()[:10]} ({ds.day_night.values})'
    #figtitle = f'Water vapor mixing ratio {ds.date.astype(str).item()[:10]} ({ds.day_night.values})'

    fig, ax = plt.subplots(figsize=(10, 15))
    fig.suptitle(figtitle, fontsize=Fontsize)
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(ds['rs_wvmr'],  ds['height'], label="Radiosonde",c='black', linewidth=3)
    ax.plot(ds['rl2_wvmr'], ds['height'], label="PPL-1200s", c='dodgerblue',linewidth=4) #dodgerblue
    ax.plot(ds['dial_wvmr'],ds['height'], label="DA10",      c='darkorange',linewidth=4) #'darkorange'
    
    if Stations: 
        aws_wvmr_clean = ds['aws_wvmr'].dropna('station')
        aws_height_clean = ds['aws_height'].sel(station=aws_wvmr_clean.station)
        ax.plot(aws_wvmr_clean, aws_height_clean, 'o', label="AWSs",
                   c='black', linestyle=':', linewidth=3, ms=8)
    ax.set_xlabel(r"water vapor mixing ratio (g kg$^{-1}$)", fontsize=Fontsize)
    ax.set_ylabel("height (km AGL)",                 fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize, size=8)
    ax.set_ylim(0, Hmax)
    ax.set_xlim(0, 17.5)
    ax.grid(True)
     
    #- Create a second x-axis showing horizontal distance.
    ax2 = ax.twiny()
    ax2.set_xlabel("distance (km)", c = 'gray', fontsize=Fontsize-1)
    ax2.tick_params(axis='x', colors='gray', labelsize=Fontsize-1, size=8)
    ax2.set_xlim(0, 8)
     
    ax2.plot(ds['rs_distance'].values, ds['height'], 
             label="Distance between RS and Lidars", 
             color='gray', linestyle=':', linewidth=3, alpha=0.4)
    if Stations: 
        aws_distance_clean = ds['aws_distance'].sel(station=aws_wvmr_clean.station)
        ax2.plot(aws_distance_clean, aws_height_clean, 'o',
                 label="Distance between AWSs and Lidars",
                 color='gray', linestyle=':', linewidth=3, alpha=0.4)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=Fontsize-2, loc= 'upper right')
    
    # - Label the crest height
    ax.text(0.05, 1.700+Hmax/200, 'crest', fontsize=Fontsize, color="gray")
    ax.axhline(y=1.700, color='gray', linestyle='-', linewidth=1.3)
    fig.tight_layout()
    
    if Save:
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots"
        if ds.day_night.values == 'day':
            filename = f'vertical_day_wvmr_{ds.date.values.astype("datetime64[D]")}T12'
        elif ds.day_night.values == 'night':
            filename = f'vertical_day_wvmr_{ds.date.values.astype("datetime64[D]")}T02'
        filename = f'{filename}__to{Hmax}km.png'  
        savefig(fig, folderpath, filename, show=True)
    
    plt.show()
     
def plot_temp(data_sel, Save, Stations, Hmax):
    
    ds = data_sel.copy()
    ds['height'] = ds['height'] / 1000
    ds['aws_height'] = ds['aws_height'] / 1000
    ds = ds.drop_sel({'station': 'Olympisch'})
    
   # t_ppl = ds['t_ppl'].values
   
    ds['aws_temp']     = xr.where(ds['aws_temp']<-200, np.nan, ds['aws_temp']) 
    ds['rl_temp']      = xr.where(ds['remp_rl']< 0, np.nan, ds['rl_temp']) 
    ds['rl2_temp']     = xr.where(ds['rl2_temp']<0, np.nan, ds['rl2_temp']) 
    
    # ds['t_ppl']      = ds['rl_temp'] - 273.15      
    # ds['t_ppl20min'] = ds['rl2_temp'] - 273.15 

    Fontsize = 22
    figtitle = f'Temperature {ds.date.astype(str).item()[:10]} ({ds.day_night.values})'
        
    fig, ax = plt.subplots(figsize=(10, 15))
    fig.suptitle(figtitle, fontsize=Fontsize+2)
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(ds['rs_temp'],  ds['height'], label="Radiosonde", c='black', linewidth=3)
    ax.plot(ds['rl2_temp'], ds['height'], label="PPL-1200s",  c='dodgerblue', linewidth=3)
    
    if Stations: 
        ax.plot(ds['aws_temp'], ds['aws_height'], 'o', label="AWSs",
                   c='black', linestyle=':', linewidth=3, ms=8)
    ax.set_xlabel("temperature (°C)", fontsize=Fontsize)
    ax.set_ylabel("height (km AGL)",                 fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize)
    ax.set_ylim(0, Hmax)
    ax.set_xlim(-60, 41)
    ax.grid(True)
     
    #- Create a second x-axis showing horizontal distance.
    ax2 = ax.twiny()
    ax2.set_xlabel("distance (km)", c = 'gray', fontsize=Fontsize)
    ax2.tick_params(axis='x', colors='gray', labelsize=Fontsize)
    ax2.set_xlim(0, 8)
     
    ax2.plot(ds['rs_distance'], ds['height'], 
             label="Distance between RS and Lidars", 
             color='cornflowerblue', linestyle='-.', linewidth=3, alpha=0.3)
    if Stations: 
        ax2.plot(ds['aws_distance'], ds['aws_height'], 'o',
                 label="Distance between AWSs and Lidars",
                 color='gray', linestyle=':', linewidth=3, alpha=0.3)
    
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, fontsize=Fontsize-2, loc= 'upper right')
    
    # - Label the crest height
    ax.text(-58, 1.700+Hmax/200, 'crest', fontsize=Fontsize, color="gray")
    ax.axhline(y=1.700, color='gray', linestyle='-', linewidth=1.3)
    fig.tight_layout()
    
    if Save:
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots"
        if ds.day_night.values == 'day':
            filename = f'vertical_day_temp_{ds.date.values.astype("datetime64[D]")}T12'
        elif ds.day_night.values == 'night':
            filename = f'vertical_day_temp_{ds.date.values.astype("datetime64[D]")}T02'
        filename = f'{filename}__to{Hmax}km.png'  
        savefig(fig, folderpath, filename)
    
    plt.show()
    
def plot_filtered_wvmr(data_sel, Save, Stations, Hmax):   
    
    Fontsize = 25 #22 #25
    #Hmax=12#3 #12
    
    ds = data_sel.copy()
    ds['height'] = ds['height'] / 1000
    ds['aws_height'] = ds['aws_height'] / 1000
    ds = ds.drop_sel({'station': 'Olympisch'})
    
    #ds['rl2_wvmr_filtered'] = xr.where(data.height < 0.1, np.nan, ds['rl2_wvmr'])
    ds['aws_wvmr'] = xr.where(ds['aws_wvmr'] < 0, np.nan, ds['aws_wvmr']) 
    
    # figtitle = f'{ds.date.astype(str).item()[:10]} ({ds.day_night.values})'
    # figtitle = f'{ds.date.values[0].astype("datetime64[D]")} (02 UTC)'
    figtitle = f'Water vapor mixing ratio {ds.date.astype(str).item()[:10]} ({ds.day_night.values[0]})'

    fig, ax = plt.subplots(figsize=(10, 15)) #10,15 #10,25 
    fig.suptitle(figtitle, fontsize=Fontsize+4)# +2
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(ds['rs_wvmr'].values.flatten(),  ds['height'],label="Radiosonde",c='black',     linewidth=3)
    ax.plot(ds['rl2_wvmr_filtered'].values.flatten(), ds['height'],label="PPL-1200s", c='dodgerblue',linewidth=3)
    ax.plot(ds['dial_wvmr'].values.flatten(),ds['height'],label="DA10",      c='darkorange',linewidth=3)
    
    if Stations: 
        valid = ~np.isnan(ds['aws_wvmr']) & ~np.isnan(ds['aws_height'])
        ax.plot(ds['aws_wvmr'].values[valid], ds['aws_height'].values[valid], marker='o', label="AWSs",
                c='black', linestyle=':', linewidth=2, markersize=8)
    ax.set_xlabel("water vapor mixing ratio (g/kg)", fontsize=Fontsize)
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
     
    ax2.plot(ds['rs_distance'].values.flatten(), ds['height'], 
             label="Distance between RS and Lidars", 
             color='gray', linestyle=':', linewidth=3, alpha=0.3)
    if Stations: 
        ax2.plot(ds['aws_distance'].values[valid], ds['aws_height'].values[valid], 'o',
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
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots"
        if ds.day_night.values == 'day':
            filename = f'vertical_wvmr_{ds.date.values[0].astype("datetime64[D]")}T12'
        elif ds.day_night.values == 'night':
            filename = f'vertical_wvmr_{ds.date.values[0].astype("datetime64[D]")}T02'
        filename = f'1_{filename}__to{Hmax}km.png'   
        # filename = f'all_to_{Hmax}km' + filename + '_filtered.png'
        savefig(fig, folderpath, filename)
    
    plt.show()
     
def plot_filtered_temp(data_sel, Save, Stations, Hmax):
    
   # t_ppl = ds['t_ppl'].values
    Fontsize = 22
    
    ds = data_sel.copy()
    ds['height'] = ds['height'] / 1000
    ds['aws_height'] = ds['aws_height'] / 1000
    ds = ds.drop_sel({'station': 'Olympisch'})
    
    ds['aws_temp']     = xr.where(ds['aws_temp'] < -200, np.nan, ds['aws_temp']) 
    ds['rl2_temp_filtered'] = xr.where(ds['rl2_temp'] < 0, np.nan, ds['rl2_temp']) 
    #ds['temp_ppl'] = ds['temp_ppl'] - 273.15 
    ds['rl2_temp_filtered'] = xr.where(data.height < 0.1, np.nan, ds['rl2_temp'])

    figtitle = f'Temperature {ds.date.values[0].astype("datetime64[D]")} ({ds.day_night.values})'
        
    fig, ax = plt.subplots(figsize=(10, 15))
    fig.suptitle(figtitle, fontsize=Fontsize+2)
    
    # - Create axis showing the vertical WVMR ratios
    ax.plot(ds['rs_temp'],  ds['height'], label="Radiosonde", c='black', linewidth=3)
    ax.plot(ds['rl2_temp_filtered'], ds['height'], label="PPL-1200s",  c='dodgerblue',linewidth=2)
    
    if Stations: 
        valid = ~np.isnan(ds['aws_wvmr']) & ~np.isnan(ds['aws_height'])
        ax.plot(ds['aws_temp'].values[valid], ds['aws_height'].values[valid], 'o', label="AWSs",
                   c='black', linestyle=':', linewidth=3, ms=8)
    ax.set_xlabel("temperature (°C)", fontsize=Fontsize)
    ax.set_ylabel("height (km AGL)",                 fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize)
    ax.set_ylim(0, Hmax)
    ax.set_xlim(-60, 41)
    ax.grid(True)
     
    #- Create a second x-axis showing horizontal distance.
    ax2 = ax.twiny()
    ax2.set_xlabel("distance (km)", c = 'gray', fontsize=Fontsize)
    ax2.tick_params(axis='x', colors='gray', labelsize=Fontsize)
    ax2.set_xlim(0, 8)
     
    ax2.plot(ds['rs_distance'], ds['height'], 
             label="Distance between RS and Lidars", 
             color='gray', linestyle='-.', linewidth=3, alpha=0.3)
    if Stations: 
        ax2.plot(ds['aws_distance'], ds['aws_height'], 'o',
                 label="Distance between AWSs and Lidars",
                 color='gray', linestyle=':', linewidth=3, alpha=0.3)
    
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, fontsize=Fontsize-2, loc= 'upper right')
    
    # - Label the crest height
    
    ax.text(-58, 1.700+Hmax/200, 'crest', fontsize=Fontsize, color="gray")
    ax.axhline(y=1.700, color='gray', linestyle='-', linewidth=1.3)
    fig.tight_layout()
    
    if Save:
        folderpath = os.path.join(os.path.dirname(os.getcwd()),
                                  'plots','VerticalPlots',
                                  'Temperature_PPL20m-AWS_Raso')
        if ds.day_night.values == 'day':
            filename = f'{ds.date.values[0].astype("datetime64[D]")}T12'
        elif ds.day_night.values == 'night':
            filename = f'{ds.date.values[0].astype("datetime64[D]")}T02'
        filename = f'Temp_{filename}_to{Hmax}km.png'
        savefig(fig, folderpath, filename)
    
    plt.show()

     
# --- Load data

data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")

# Settings
Save          = False
Stations      = True
Hmax          = 4


date = np.datetime64("2024-08-24")

mask = (data['day_night'] == 'night') & (data['date'].values == date)
# mask = (data['day_night'] == 'day') & (data['date'].values == date)

data_sel = data.sel(launch=mask).squeeze('launch')

#plot_filtered_wvmr(data_sel, Save, Stations, Hmax)
plot_wvmr(data_sel, Save, Stations, Hmax)


#%%
# Settings
Save          = True
Stations      = True
Hmax          = 4

data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")

for i in data.launch.values:
    # i=87
    ds = data.sel(launch=i)#.squeeze('launch')
    
    print('\n',
          'Plotting... ',
          "Launch label:", i+1, 
          "Date:", ds.date.astype(str).item()[:10], 
          "When:", ds.day_night.values)
    
    plot_wvmr(ds, Save, Stations, Hmax)

