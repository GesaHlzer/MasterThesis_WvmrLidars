# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:01:26 2026

@author: alleh
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from basic_plot_funcions import savefig, classify_daytime

start = np.datetime64("2024-08-23")
end = np.datetime64("2024-09-09")

# -----------------------------------------------------------

ppl20m = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc"
ppl10s = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_90.0%valid.nc"
dial = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc"


def plot_max_range_occurrence(lidar, start, end, hmax=12):
    """
    Plottet für jede Höhe wie oft max_range diese Höhe erreicht (in %).
    Getrennt nach day, night, twilight und gesamt.
    """
    
    if lidar == 'PPL (20min)': 
        ds = xr.open_dataset(ppl20m)
        max_range =  ds.wvmr_max_range
    elif lidar == 'PPL (10s)': 
        ds = xr.open_dataset(ppl10s)
        max_range = ds.wvmr_max_range
    else:  
        ds = xr.open_dataset(dial) 
        max_range = ds.water_vapor_max_range
        
    # mask_time = (( max_range.time >= np.datetime64("2024-06-30")) 
    #                 & (max_range.time <= np.datetime64("2024-07-01")) )
    # maxr = max_range.sel(time=mask_time).values
        
    day_class = classify_daytime(ds)
    mr = max_range
    heights = ds.height.values  # alle Höhenwerte

    # Zeitmaske
    mask_time = (ds.time >= start) & (ds.time <= end)
    
    # Subsets
    subsets = {
        'all':       mask_time,
        'night':     mask_time & (day_class == 'night'),
        'twilight':  mask_time & (day_class == 'twilight'),
        'day':       mask_time & (day_class == 'day'),
    }
    
    colors = {'all': 'black', 'night': 'royalblue', 'twilight': 'tomato', 'day': 'orange'}
    labels = {'all': 'All',   'night': 'Night',     'twilight': 'Twilight',     'day': 'Day'}
    linestyles = {'all': '-',       'night': '--',       'twilight': ':'  ,     'day': '-.'}
    markers    = {'all': 'o',       'day': 's',        'night': '^',       'twilight': 'D'}
    
    fig, ax = plt.subplots(figsize=(6, 7))

    for key, mask in subsets.items():
        mr_sub = mr.sel(time=mask).values  # (n_time,)
        n      = len(mr_sub)
        
        # Für jede Höhe: Anteil der Zeitschritte wo max_range >= height
        occurrence = np.array([np.sum(mr_sub >= h) / n * 100 for h in heights])
        
        ax.plot(occurrence, heights / 1000, 
                #marker=markers[key], markevery=50, markersize=7,
                linestyle=linestyles[key], color=colors[key], 
                label=f"{labels[key]} (n={n})", linewidth=2)

    ax.set_xlabel('Occurrence (%)', fontsize=14)
    ax.set_ylabel('Height (km AGL)', fontsize=14)
    ax.set_title(f'{lidar} Water Vapor Max Range Occurrence\n{start} – {end}', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, hmax)
    fig.tight_layout()
    plt.show()
    
    return fig

# --- Compare Daytime Range for the different lidars:

lidar = 'DA10 (1 min)' # 'PPL (20min)' 'PPL (10s)' 'DA10 (1 min)'

# for DA10
# start = np.datetime64("2024-06-18")
# end = np.datetime64("2024-10-22")
# fig = plot_max_range_occurrence(lidar, start, end, hmax=4)


fig = plot_max_range_occurrence(lidar, start, end)
folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
filename = f"{lidar}_{start}_{end}.png"
savefig(fig, folderpath, filename, dpi=300, show=True)


def plot_max_range_occurrence_compare(start, end, daytime='all'):
    """
    Vergleicht max_range occurrence der drei LiDAR Datensätze in einem Plot.
    
    daytime: 'all', 'day', 'night', 'twilight'
    """
    # Datensätze laden
    datasets = {
        'PPL (20min)': (xr.open_dataset(ppl20m), 'wvmr_max_range'),
        'PPL (10s)':   (xr.open_dataset(ppl10s), 'wvmr_max_range'),
        'DA10':        (xr.open_dataset(dial),   'water_vapor_max_range'),
    }

    colors     = {'PPL (20min)': 'blue', 'PPL (10s)': 'violet', 'DA10': 'orange'}
    linestyles = {'PPL (20min)': '--',      'PPL (10s)': '-.',       'DA10': '-'}

    fig, ax = plt.subplots(figsize=(7, 9))

    for lidar_name, (ds, mr_var) in datasets.items():
        day_class = classify_daytime(ds)
        mr        = ds[mr_var]
        heights   = ds.height.values

        mask_time = (ds.time >= start) & (ds.time <= end)

        if daytime == 'all':
            mask = mask_time
        else:
            mask = mask_time & (day_class == daytime)

        mr_sub     = mr.sel(time=mask).values
        n          = len(mr_sub)
        occurrence = np.mean(mr_sub[:, None] >= heights[None, :], axis=0) * 100

        ax.plot(occurrence, heights / 1000,
                color=colors[lidar_name], linestyle=linestyles[lidar_name],
                linewidth=2, label=f"{lidar_name} (n={n})")

    daytime_label = daytime.capitalize()
    ax.set_xlabel('Occurrence (%)', fontsize=14)
    ax.set_ylabel('Height (km AGL)', fontsize=14)
    ax.set_title(f'Water Vapor Max Range Occurrence — {daytime_label}\n{start} – {end}', fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 12)

    fig.tight_layout()
    plt.show()
    return fig

# --- Compare Lidars for different daytimes:
    
# daytime='all' # 'all' 'day' 'night' 'twilight'

# fig = plot_max_range_occurrence_compare(start, end, daytime=daytime)
# folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
# filename   = f"compare_{daytime}_{start}_{end}.png"
# savefig(fig, folderpath, filename, dpi=300, show=True)
