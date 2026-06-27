# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:01:26 2026

@author: alleh
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from basic_plot_funcions import savefig, classify_daytime, load_sun_times

start = np.datetime64("2024-08-23")
end = np.datetime64("2024-09-09")
sun_file = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\SSD_IMGI_SSDundDAEMMERUNG.txt"
# -----------------------------------------------------------

ppl20m = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_75.0%valid.nc"
ppl10s = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_50.0%valid.nc"
dial = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc"

#via dytime
def plot_max_range_occurrence(lidar, start, end, hmax=12):
    """
    Plottet für jede Höhe wie oft max_range diese Höhe erreicht (in %).
    Getrennt nach day, night, twilight und gesamt.
    """
    
    if lidar == 'PPL (20-min)': 
        ds = xr.open_dataset(ppl20m)
        max_range =  ds.wvmr_max_range
    elif lidar == 'PPL (10-s)': 
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

# lidar = 'DA10 (1 min)' # 'PPL (20min)' 'PPL (10s)' 'DA10 (1 min)'

# for DA10
# start = np.datetime64("2024-06-18")
# end = np.datetime64("2024-10-22")
# fig = plot_max_range_occurrence(lidar, start, end, hmax=4)


# fig = plot_max_range_occurrence(lidar, start, end)
# folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
# filename = f"{lidar}_{start}_{end}.png"
# savefig(fig, folderpath, filename, dpi=300, show=True)


def plot_max_range_occurrence_compare(start, end, daytime='all'):
    """
    Vergleicht max_range occurrence der drei LiDAR Datensätze in einem Plot.
    
    daytime: 'all', 'day', 'night', 'twilight'
    """
    Fontsize = 20 #17
    # Datensätze laden
    datasets = {
        'PPLS (20-min)': (xr.open_dataset(ppl20m), 'wvmr_max_range'),
        'PPLS (10-s)':   (xr.open_dataset(ppl10s), 'wvmr_max_range'),
        'DA10':        (xr.open_dataset(dial),   'water_vapor_max_range'),
    }

    # colors     = {'PPLS (20-min)': 'blue', 'PPLS (10-s)': '#2ca02c', 'DA10': 'orange'}
    colors  = {'DA10': '#ff7f0e', 'PPLS (10-s)': '#2ca02c', 'PPLS (20-min)': '#1f77b4'}
    #linestyles = {'PPLS (20-min)': '--',   'PPLS (10-s)': '-',      'DA10': '-.'}
    linestyles = {'PPLS (20-min)': '-',   'PPLS (10-s)': '-',      'DA10': '-'}   
    fig, ax = plt.subplots(figsize=(7, 7))

    for lidar_name, (ds, mr_var) in datasets.items():
        #lidar_name, (ds, mr_var) = 'PPLS (10s)', (xr.open_dataset(ppl10s), 'wvmr_max_range')
        
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
    ax.set_xlabel('occurrence (%)',  fontsize=Fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
    #ax.set_title(f'Water Vapor Max Range Occurrence — {daytime_label}\n{start} – {end}', fontsize=14)
    # ax.text(0.02, 0.07, f'{daytime_label}', 
    #         transform=ax.transAxes, fontsize=Fontsize+3, verticalalignment='top',
    #    bbox=dict(boxstyle='round',edgecolor='none',facecolor='whitesmoke', alpha=0.9))
    ax.legend(fontsize=Fontsize-2, loc='upper right')
    ax.tick_params(axis='both', labelsize=Fontsize-1)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 12)

    fig.tight_layout()
    plt.show()
    return fig

# --- Compare Lidars for different daytimes:
    
# daytime='all' # 'all' 'day' 'night' 'twilight'
# daytime='day'
# daytime='night'
# daytime='twilight'

# fig = plot_max_range_occurrence_compare(start, end, daytime=daytime)
# folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
# filename   = f"Range_compare_v3_{daytime}_{start}_{end}.png"
# savefig(fig, folderpath, filename, dpi=300, show=True)


def plot_typical_range_during_day(start, end):
    
    
    sun_times = load_sun_times(sun_file)
    # Filter sun times to the study period
    mask_sun = (sun_times.index >= pd.Timestamp(start)) & (sun_times.index <= pd.Timestamp(end))
    df_period = sun_times.loc[mask_sun]
     
    def mean_hour(series):
        """Average fractional UTC hour across dates."""
        return series.dt.hour + series.dt.minute / 60
 
    avg_beg_daem       = mean_hour(df_period['BeginnDaemmerung']).mean()
    avg_eff_sunrise    = mean_hour(df_period['effektiverSonnenaufgang']).mean()
    avg_eff_sunset     = mean_hour(df_period['effektiverSonnenuntergang']).mean()
    avg_end_daem       = mean_hour(df_period['EndeDaemmung']).mean()
    
    datasets = {
    'DA10':         (xr.open_dataset(dial),  'water_vapor_max_range'),
    'PPLS (10-s)':  (xr.open_dataset(ppl10s), 'wvmr_max_range'),
    'PPLS (20-min)':(xr.open_dataset(ppl20m), 'wvmr_max_range'),
    }

    colors  = {'DA10': '#ff7f0e', 'PPLS (20-min)': '#1f77b4', 'PPLS (10-s)': '#2ca02c'}#'mediumvioletred',}
    #colors  = {'DA10': 'darkorange', 'PPLS (20min)': 'dodgerblue',   'PPLS (10s)': '#2ca02c'}
    markers = {'DA10': 'o',       'PPLS (10-s)': 's',        'PPLS (20-min)': '^'}
 
    results = {}
    
    for name, (ds, var) in datasets.items():
        da = ds[var]
        mask = (da.time >= start) & (da.time <= end)
        da   = da.sel(time=mask)
     
        # Convert to pandas Series
        s = da.to_series().dropna()
        s.index = pd.to_datetime(s.index)
     
        # Group by UTC hour
        hourly = s.groupby(s.index.hour).mean()
        hourly.loc[24] = hourly.loc[0]
        hourly = hourly/1000 # m->km
        results[name] = hourly
    
    Fontsize = 18 # v1: 15 , v2 14, v3 18
    fig, ax = plt.subplots(figsize=(10, 5)) # v1: 10,4 , v2 7, 4, v3 10,5
 
    for name, hourly in results.items():
        ax.plot(hourly.index, hourly.values,
                color=colors[name], marker=markers[name],
                linewidth=1.8, markersize=5, label=name)
     
    # Vertical lines for sun events — labels as inline text, not legend entries
    # (x, color, linestyle, label, ha, x_offset_hours)
    # ha and x_offset control which side of the line the text sits on
    sun_lines = [
        (avg_beg_daem,    '#555555', '--', 'Begin\ntwilight',  'right', -0.15),
        (avg_eff_sunrise, '#e67e00', '-',  'Eff.\nsunrise',    'left',   0.15),
        (avg_eff_sunset,  '#e67e00', '-',  'Eff.\nsunset',     'right', -0.15),
        (avg_end_daem,    '#555555', '--', 'End\ntwilight',    'left',   0.15),
    ]
    #y_top = ax.get_ylim()[1]  # will be updated after data is plotted; use transform instead
    
    for x, col, ls, label, ha, xoff in sun_lines:
        ax.axvline(x, color=col, linestyle=ls, linewidth=1.4)
        # Place text near the top of the axes using axes-fraction y
        ax.text(x + xoff, 0.98, label,
                transform=ax.get_xaxis_transform(),   # x in data, y in axes [0,1]
                color=col, fontsize=Fontsize-1, ha=ha, va='top',
                linespacing=1.3)
     
    ax.set_xlabel('hour (UTC)', fontsize=Fontsize)
    ax.tick_params(axis='both', labelsize=Fontsize)
    ax.set_ylabel('average max. range (km AGL)', fontsize=Fontsize)
    # ax.set_title('Diurnal variation of water vapour max range\n'
                 # f'2024-08-23 – 2024-09-09', fontsize=Fontsize)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.35)
     
    # Legend for the three datasets only, placed in the centre of the plot
    ax.legend(fontsize=Fontsize-2, loc='upper center', bbox_to_anchor=(0.5, 0.8))

     
    fig.tight_layout()
    plt.show()
    return fig

# fig = plot_typical_range_during_day(start, end)
# folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
# filename   = f"Range_during_dayhours_{start}_{end}_v0.png"
# savefig(fig, folderpath, filename, dpi=300, show=True)

def save_max_range_percentile_table(start, end, output_path='max_range_percentiles_new.txt'):
    """
    For each dataset and daytime class, find the height at which
    occurrence equals 5, 10, 25, 50, 75, 90, 95 %.
    
    'Occurrence at height h' = fraction of profiles whose max_range >= h,
    so the percentile height is where occurrence crosses the target %.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    datasets = {
        'PPLS (20-min)': (xr.open_dataset(ppl20m), 'wvmr_max_range'),
        'PPLS (10-s)':   (xr.open_dataset(ppl10s), 'wvmr_max_range'),
        'DA10':          (xr.open_dataset(dial),   'water_vapor_max_range'),
    }
    daytime_classes = ['all', 'day', 'night', 'twilight']
    target_occ      = [5, 10, 25, 50, 75, 90, 95]   # occurrence % targets

    lines = []
    lines.append(f"Max-range height at given occurrence levels")
    lines.append(f"Period: {start} – {end}")
    lines.append(f"Occurrence = fraction of profiles with max_range >= height")
    lines.append("")

    for lidar_name, (ds, mr_var) in datasets.items():
        lines.append("=" * 60)
        lines.append(f"Dataset: {lidar_name}")
        lines.append("=" * 60)

        day_class = classify_daytime(ds)
        mr        = ds[mr_var]
        heights   = ds.height.values          # metres
        mask_time = (ds.time >= start) & (ds.time <= end)

        for daytime in daytime_classes:
            if daytime == 'all':
                mask = mask_time
            else:
                mask = mask_time & (day_class == daytime)

            mr_sub = mr.sel(time=mask).values
            n      = len(mr_sub)

            # occurrence curve: for each height, % of profiles reaching it
            occurrence = np.mean(mr_sub[:, None] >= heights[None, :], axis=0) * 100
            # occurrence is monotonically decreasing with height → can interpolate
            # flip so x (occurrence) is increasing for interp1d
            occ_flip = occurrence[::-1]
            hgt_flip = heights[::-1]

            # Remove duplicate occurrence values to allow interpolation
            _, idx = np.unique(occ_flip, return_index=True)
            occ_u  = occ_flip[idx]
            hgt_u  = hgt_flip[idx]

            lines.append(f"\n  Daytime class : {daytime.capitalize()}  (n={n})")
            lines.append(f"  {'Occurrence (%)':>15}  {'Height (m)':>12}  {'Height (km)':>12}")
            lines.append(f"  {'-'*45}")
            for occ_target in target_occ:
                idx_nearest = np.argmin(np.abs(occurrence - occ_target))
                h = heights[idx_nearest]
                lines.append(f"  {occ_target:>15}  {h:>12.1f}  {h/1000:>12.3f}")


        lines.append("")

    text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Saved → {output_path}")
    return text

# Run it
folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\rangeanalysis"
save_max_range_percentile_table(start, end, output_path=folderpath+r'\max_range_percentiles_new.txt')


