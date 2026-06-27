# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:54:45 2025

@author: alleh
"""
import os
import numpy  as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import matplotlib.dates  as mdates
import matplotlib.ticker as ticker
import colormaps
import scipy.ndimage as nd
from skimage.morphology import remove_small_objects
from basic_plot_funcions import savefig, grid_edges
from colormaps import cmap_wvmr
# Define Functions
def ppl_timeseries(data, var, date, valid_range=None):
    # PPL  2024-8-14 (bzw. 7) - 2024-9-8
    
    fontsize = 18
    Hmax     = 6 # km 
        
    print(f"\n Making PPL {var} plot...")
    
    # Extend x (time array) and y (height array) to include first and last edges
    time = data['time'].values
    heights = data['height']
    t, h = grid_edges(time, heights)
    
    if var == 'MR':
        param = data.to_numpy().transpose()
        par_cmap = cmap_wvmr() #'Blues'
        param_label = 'water vapor mixing ratio (g/kg)'
        vmin=0
        vmax=15
    if var == 'T':
        param = data.to_numpy().transpose()
        par_cmap = colormaps.cmap_bluered40()
        param_label = 'temperature (K)' 
        vmin=220
        vmax=310
    if var == 'BR':
        param = data.to_numpy().transpose()
        par_cmap = 'viridis' #'twilight_shifted' 
        param_label = 'backscatter ratio'
        vmin=0
        vmax=3
        
    param = np.ma.masked_invalid(param)  # Ensures NaNs are masked
    start = np.datetime64(date, 'ns')
    end = np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
    # ----  Plot Data  
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Pixel Plot
    pcm = ax.pcolormesh(t, h, param, shading='flat', vmin=vmin, vmax=vmax, cmap=par_cmap) 
    #pcm = ax.pcolormesh(t, h, param, shading='flat', cmap=par_cmap) 

    # Format Colorbar
    cbar = plt.colorbar(pcm, ax=ax, pad=0.03, extend='neither')
    cbar.ax.tick_params(direction='out', labelsize=fontsize, size=10)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    cbar.set_label(param_label, size=fontsize)
    
    if valid_range is not None:
        ax.plot(time, valid_range/1000, color='r', linewidth=2)
    
    # Format Axes
    ax.set_xlim([start, end])
    ax.set_ylim([0, Hmax])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(direction='out', labelsize=fontsize)
    
    # Add Title and Labels
    ax.set_title(f"Purple Pulse Lidar: {date.strftime('%Y-%m-%d')}", fontsize=fontsize)
    ax.set_xlabel('time (UTC)', fontsize=fontsize)
    ax.set_ylabel('height (km AGL)', fontsize=fontsize)
    
    ax.set_facecolor([0.8, 0.8, 0.8])
    plt.tight_layout()
    plt.show()
    
    return fig

def max_valid_height(array, dt, window=200, min_valid_fraction=0.8, smoothing_10s=6):
    """
    Sucht von oben nach unten den ersten Höhenbereich wo
    mindestens min_valid_fraction der Werte im Fenster gültig sind.
    
    window: amount of vertical pixels → 200 * 3.75m = 750m
    min_valid_fraction: fraction that needs to be non-nans
    """
    heights = array.height
    # (n_time, n_height) bool array: True = valid
    valid = (~np.isnan(array)).values.astype(np.float32)
    
    # Rollende Summe über Höhenachse
    cumsum = np.cumsum(valid, axis=1)
    # Anzahl gültiger Werte in jedem Fenster [h-window:h]
    windowed = cumsum[:, window:] - cumsum[:, :-window]  # shape: (n_time, n_height-window)
    fraction = windowed / window  # shape: (n_time, n_height-window)
    
    # Von oben nach unten: letzter Index wo fraction >= min_valid_fraction
    # flip damit wir von oben suchen
    flipped = fraction[:, ::-1] >= min_valid_fraction  # True wo valide
    
    # Erster True-Wert pro Zeitschritt (= höchste valide Höhe)
    idx = np.argmax(flipped, axis=1)  # shape: (n_time,)
    
    # Zurückrechnen auf ursprünglichen Höhenindex
    # height_idx = (len(heights) - window) - 1 - idx #bottom of window
    height_idx = (len(heights) - int(window/2)) - 1 - idx #bottom of window
    # height_idx = (len(heights) - 1) - idx # top of window
    
    # Fallback: wo kein valider Bereich gefunden → unterste Höhe
    no_valid = ~flipped.any(axis=1)
    height_idx[no_valid] = 0
    
    max_range = heights[height_idx].values
    
    if dt == '10s':
        smoothing_window = smoothing_10s #6 =1 min
        max_range = pd.Series(max_range).rolling(window=smoothing_window, center=True, 
                                                 min_periods=1).mean().values
    
    return max_range
    
def filter_mr(ds, dt):
    # -------------- Settings ------------------   
    # specify data cleaning options: 'no'/ 'yes'
    apply_discard_lowest                 = 'yes' 
    apply_mr_threshold                   = 'yes' 
    apply_br_threshold                   = 'yes'
    apply_local_median_outlier_detection = 'no' if dt=='10s' else 'yes'
    apply_gradient_based_spike_detection = 'yes' #'no' if dt=='10s' else 'yes'
    apply_cleanup_of_isolated_pixels     = 'yes'
    # ------------------------------------------
    mr = ds.wvmr.copy()
    # Filter
    if apply_discard_lowest == 'yes':
        """
        Sets all values below the lower_limit to NaN.
        """
        lower_limit = 200# 300 if dt=='10s' else 200
        mr = xr.where(mr.height < lower_limit, np.nan, mr)
        mr = mr.transpose()
        
    if apply_mr_threshold == 'yes':   
        """
        Flags any value that exceeds an absolute threshold as NaN
        """
        upper_mr_threshold = 25
        lower_mr_threshold = -0.5 #include noise
        mr = mr.where((mr >= lower_mr_threshold) & (mr <= upper_mr_threshold))
    
    if apply_local_median_outlier_detection == 'yes':
        """ 
        Calculates the local median for each point within a  
        time × elevation window 
        If a value deviates by more than 8 K from the local median, 
        it is marked as an outlier and set to NaN 
        """
        arr = mr.to_numpy()
        nan_mask = np.isnan(arr)
        filled = np.where(nan_mask, 0, arr)
        
        if dt == '10s': # not using it here: too noisy and inefficient
            # Local median (1min × 375m window in time × height)
            local_med = nd.median_filter(filled, size=(720, 100), mode='nearest')
            #local_med = nd.median_filter(filled, size=(3, 100), mode='nearest')
            threshold = 30  # g/kg, tune to your noise level
        else:
            # Local median (2h x 375 m window in time × height)
            local_med = nd.median_filter(filled, size=(3, 100), mode='nearest')
            threshold = 6  # g/kg, tune to your noise level
        
        # Deviation and threshold
        dev = np.abs(filled - local_med)
        spikes = dev > threshold
        
        # Apply mask back to xarray
        cleaned = np.where(spikes, np.nan, arr)
        mr = xr.DataArray(cleaned, coords=mr.coords, dims=mr.dims)
    
    if apply_gradient_based_spike_detection  == 'yes':
        """
        Calculates the gradient in the time direction (dt) and the height direction (dz). 
        If the gradient exceeds a threshold per step in either direction, the point is set to NaN. 
        Detects abrupt jumps between adjacent data points.
        """
        if dt == "10s":
            # not using it here: too noisy and inefficient
            threshold_dt = 20 # g/kg 
        else:
            threshold_dt = 8 # g/kg
            
        arr = mr.to_numpy()
        dt_grad = np.abs(np.diff(arr, axis=0, prepend=arr[:1,:]))
        dz_grad = np.abs(np.diff(arr, axis=1, prepend=arr[:,:1]))
        spike_mask = (dt_grad > threshold_dt) | (dz_grad > 1)  # thresholds in g/kg per step
        arr_spk = np.where(spike_mask, np.nan, arr)
        mr = xr.DataArray(arr_spk, coords=mr.coords, dims=mr.dims)
        
    if apply_br_threshold == 'yes':
        """
        Filters based on the backscatter ratio (br). Only values where 
        0 ≤ br ≤ 5 are retained. Discards areas with too weak a signal 
        (clouds, heavy aerosols/cloud/rain, or no scatterer)."
        """
        upper_br_threshold = 10 if dt=="10s" else 30
        lower_br_threshold = 0.5
        mr = mr.where((ds.br >= lower_br_threshold) & (ds.br <= upper_br_threshold))
    
    if apply_cleanup_of_isolated_pixels == 'yes':
        """
        Removes contiguous valid clusters. Prevents 
        small isolated islands of data points from remaining in the otherwise 
        filtered area—likely artifacts.
        """
        if dt == "10s":
            isolated_pixel_size = 6000 # (e.g 2 min (12 dt) x 188m (50 dh))
        else:
            isolated_pixel_size = 100 #(e.g. 20 min (1 dt) x 188m (50 dh))
        mask = ~np.isnan(mr.to_numpy())  # True = valid data

        # Remove tiny clusters (<200 pixels) 
        clean_mask = remove_small_objects(mask, min_size=isolated_pixel_size)
        # clean_mask = remove_small_holes(clean_mask, area_threshold=20)

        # Apply back
        cleaned_arr = np.where(clean_mask, mr, np.nan)
        mr = xr.DataArray(cleaned_arr, coords=mr.coords, dims=mr.dims)
    
    return mr

def filter_t(ds, dt):
    # -------------- Settings ------------------   
    # specify data cleaning options: 'no'/ 'yes'
    apply_discard_lowest                     = 'yes'
    apply_valid_value_range_threshold        = 'yes' 
    apply_local_median_outlier_detection     = 'yes'
    apply_gradient_based_spike_detection     = 'no'
    apply_br_threshold                       = 'yes'
    apply_cleanup_of_isolated_pixels         = 'yes'
    # ------------------------------------------
    temp = ds.temp.copy()
    
    # Filter
    if apply_valid_value_range_threshold == 'yes':   
        """ 
        Sets all temperature values outside the range of 200–317 K to NaN.
        """
        # Flags any value that exceeds an absolute threshold as NaN
        upper_temp_threshold = 317 #  47 °C
        lower_temp_threshold = 200 # -73 °C
        temp = temp.where((temp >= lower_temp_threshold) & (temp <= upper_temp_threshold))
    
    if apply_local_median_outlier_detection == 'yes':
        """ 
        Calculates the local median for each point within a  
        time × elevation window 
        If a value deviates by more than 8 K from the local median, 
        it is marked as an outlier and set to NaN 
        """
        # Convert to NumPy, fill NaNs with a placeholder
        arr = temp.to_numpy()
        nan_mask = np.isnan(arr)
        filled = np.where(nan_mask, 0, arr)
        
        if dt == '10s':
            # Local median (30min × 375m window in time × height)
            local_med = nd.uniform_filter(filled, size=(180, 100), mode='nearest')
            threshold = 25  # °C or 4/8
        else:
            # Local median (2h x 375 m window in time × height)
            local_med = nd.median_filter(filled, size=(6, 200), mode='nearest')
            threshold = 8  # °C or 4
        
        # Deviation and threshold
        dev = np.abs(filled - local_med)
        spikes = dev > threshold
        
        # Apply mask back to xarray
        cleaned = np.where(spikes, np.nan, arr)
        temp = xr.DataArray(cleaned, coords=temp.coords, dims=temp.dims)
    
    if apply_gradient_based_spike_detection  == 'yes':
        """
        Calculates the gradient in the time direction (dt) and the height direction (dz). 
        If the gradient exceeds a threshold per step in either direction, the point is set to NaN. 
        Detects abrupt jumps between adjacent data points.
        """
        
        threshold = 20 # K

        
        arr = temp.to_numpy()
        dt_grad = np.abs(np.diff(arr, axis=0, prepend=arr[:1,:]))
        dz_grad = np.abs(np.diff(arr, axis=1, prepend=arr[:,:1])) 
        spike_mask = (dt_grad > threshold) | (dz_grad > threshold)
        arr_spk = np.where(spike_mask, np.nan, arr)
        temp = xr.DataArray(arr_spk, coords=temp.coords, dims=temp.dims)
        
    if apply_br_threshold == 'yes':
        """
        Filters based on the backscatter ratio (br). Only values where 
        0.8 ≤ br ≤ 5 are retained. Discards areas with too weak a signal 
        (clouds, heavy aerosols, or no scatterer)."
        """
        upper_br_threshold = 10 if dt=="10s" else 30
        lower_br_threshold = 0.5

        temp = temp.where((ds.br >= lower_br_threshold) & (ds.br <= upper_br_threshold))

        
    if apply_cleanup_of_isolated_pixels == 'yes':
        """
        Removes contiguous valid clusters smaller than 300 pixels. Prevents 
        small isolated islands of data points from remaining in the otherwise 
        filtered area—likely artifacts.
        """
        if dt == "10s":
            isolated_pixel_size = 6000
        else:
            isolated_pixel_size = 100
            
        mask = ~np.isnan(temp.to_numpy())  # True = valid data
        
        # Remove tiny clusters (<200 pixels)
        clean_mask = remove_small_objects(mask, min_size=isolated_pixel_size)
        
        # Apply back
        cleaned_arr = np.where(clean_mask, temp, np.nan)
        temp = xr.DataArray(cleaned_arr, coords=temp.coords, dims=temp.dims)
       
    if apply_discard_lowest == 'yes':
        """
        Sets all values below 100 m to NaN. Removes the near-range of the LiDAR 
        where the overlap function is not yet complete (overlap region).
        """
        temp = xr.where(temp.height < 300, np.nan, temp) #meter
        
    temp = temp.transpose()

    return temp
    
def filter_br(ds, dt):
    br = ds.br.copy()
    br = br.where(br['height']> 80) #bsr.where(bsr.height < 10, np.nan, bsr)
    br =  br.where(br > 0) #bsr_filtered.where(bsr_filtered == 0, np.nan, bsr_filtered)
    br =  br.where(br < 5) #bsr_filtered.where(bsr_filtered == 0, np.nan, bsr_filtered)
    
    if dt == '10s':
        mask = ~np.isnan(br.to_numpy())  # True = valid data
        clean_mask = remove_small_objects(mask, min_size=50)
        #clean_mask = remove_small_holes(clean_mask, area_threshold=40)

    else: # dt = 1200s
        mask = ~np.isnan(br.to_numpy())  # True = valid data
        clean_mask = remove_small_objects(mask, min_size=2)
        #clean_mask = remove_small_holes(clean_mask, area_threshold=40)

    # Apply back
    cleaned_arr = np.where(clean_mask, br, np.nan)
    br = xr.DataArray(cleaned_arr, coords=br.coords, dims=br.dims)
    
    return  br


# Load Data
ppl10s = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\PPL_10s_gl97m.nc")
ppl20m = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\PPL_1200s_gl97m.nc")

#%% --- Adjust filter

# dt = '10s'
# dt = '1200s'
# ds = ppl10s.copy() if dt == '10s' else ppl20m.copy()

# date = datetime(2024, 8, 24)
# start = np.datetime64(date, 'ns') - np.timedelta64(20, 's')
# end =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
# ds = ds.sel(time=slice(start, end)) 

# # #original data
# # br = ds.br
# # mr = ds.wvmr
# # temp = ds.temp

# # # ppl_timeseries(br,'BR', date)
# # ppl_timeseries(mr,'MR', date)
# # # ppl_timeseries(temp,'T', date)

# # # # filtered data
# # # br_filtered = filter_br(ds, dt)
# # # ppl_timeseries(br_filtered,'BR', date)

# # t_filtered = filter_t(ds, dt)
# # t_range = max_valid_height(t_filtered, dt, window=200, min_valid_fraction=0.75) #750 m
# # ppl_timeseries(temp,'T', date, valid_range=t_range)
# # ppl_timeseries(t_filtered, 'T', date, valid_range=t_range) 

# mr = ds.wvmr
# mr_filtered = filter_mr(ds, dt)
# mr_range = max_valid_height(mr_filtered, dt, window=100, 
#                             min_valid_fraction=0.75) 
# ppl_timeseries(mr,'MR', date, valid_range=mr_range)
# ppl_timeseries(mr_filtered, 'MR', date, valid_range=mr_range)
    

# # ---Save new dataset

# ds_new = ds.copy()
# ds_new["temp_filtered"] = t_filtered
# ds_new["wvmr_filtered"] = mr_filtered
# ds_new["temp_max_range"] = (["time"], t_range)
# ds_new["wvmr_max_range"] = (["time"], mr_range)
# ds_new.attr["maxrange_setting"]= f"Top down, in a vertial +/-100 pixel window at least {min_valid_fraction*100}% non-nan values"
# ds_new.to_netcdf(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_{dt}_filtered.nc")
# dstest = xr.open_dataset(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_{dt}_filtered.nc")
# print(dstest)

#%% pure plotting
# or 
dt = '10s'
# dt = '1200s'

ppl1 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_50.0%valid.nc") 
ppl2 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_75.0%valid.nc") 

ds = ppl1.copy() if dt == '10s' else ppl2.copy()
mr = ds.wvmr
mr_filtered = ds.wvmr_filtered
mr_range = ds.wvmr_max_range

date = datetime(2024, 8, 24)
print(date.date())
start =  np.datetime64(date, 'ns') - np.timedelta64(1, 'm')
end   =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')

wvmr = mr.sel(time=slice(start, end)) 
wvmr_filtered = mr_filtered.sel(time=slice(start, end))
wvmr_range = mr_range.sel(time=slice(start, end))

    # fig=ppl_timeseries(wvmr, var)
fig1 = ppl_timeseries(wvmr,'MR', date, valid_range=wvmr_range)
fig2 = ppl_timeseries(wvmr_filtered, 'MR', date, valid_range=wvmr_range)
    
    
folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "PPL_filter")
filename1 = f"ppl{dt}_wvmr_{date.date()}_timeseries_validrange.png"
filename2 = f"ppl{dt}_wvmr_{date.date()}_timeseries_validrange_filtered.png"
savefig(fig2, folderpath, filename2)
savefig(fig1, folderpath, filename1)


#%%######################### Plotting ############################################

dt = '10s'
# dt = '1200s'
ds = ppl10s.copy() if dt == '10s' else ppl20m.copy()
min_valid_fraction=0.5 if dt == '10s' else 0.75
window=100
# --- wvmr

var =  'MR'
mr = ds.wvmr
mr_filtered = filter_mr(ds, dt)
mr_range = max_valid_height(mr_filtered, dt, window=window, 
                            min_valid_fraction=min_valid_fraction, 
                            smoothing_10s=6) 


date_beg = datetime(2024, 8, 23)
date_end = datetime(2024, 9, 8)
dates = [date_beg+timedelta(days=x) for x in range((date_end-date_beg).days+1)]  
  
for date in dates:
    # date = datetime(2024, 8, 31)
    print(date.date())
    start =  np.datetime64(date, 'ns') - np.timedelta64(1, 'm')
    end   =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
    
    
    wvmr = mr.sel(time=slice(start, end)) 
    wvmr_filtered = mr_filtered.sel(time=slice(start, end)) 
    wvmr_range = max_valid_height(wvmr_filtered, dt, window=100, min_valid_fraction=min_valid_fraction, smoothing_10s=6)#6) 

    # fig=ppl_timeseries(wvmr, var)
    fig1 = ppl_timeseries(wvmr,'MR', date, valid_range=wvmr_range)
    fig2 = ppl_timeseries(wvmr_filtered, 'MR', date, valid_range=wvmr_range)
    
    
    folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "PPL_filter")
    filename1 = f"ppl{dt}_wvmr_{date.date()}_timeseries_validrange_v2_{min_valid_fraction*100}%in{window}.png"
    filename2 = f"ppl{dt}_wvmr_{date.date()}_timeseries_validrange_v2_filtered_{min_valid_fraction*100}%in{window}.png"
    savefig(fig2, folderpath, filename2)
    savefig(fig1, folderpath, filename1)


# --- temp

var =  'T'
temp = ds.temp
t_filtered = filter_t(ds, dt)
t_range = max_valid_height(t_filtered, dt, window=window, 
                           min_valid_fraction=min_valid_fraction) 


# date_beg = datetime(2024, 8, 23)
# date_end = datetime(2024, 9, 8)
# dates = [date_beg+timedelta(days=x) for x in range((date_end-date_beg).days+1)]  
  
# for date in dates:
#     print(date.date())
#     start =  np.datetime64(date, 'ns') 
#     end   =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
    
#     temperature = t_filtered.sel(time=slice(start, end))
    
#     #fig = ppl_timeseries(temperature, var)
#     fig1 = ppl_timeseries(temp,'T', date, valid_range=t_range)
#     fig2 = ppl_timeseries(t_filtered, 'T', date, valid_range=t_range)
        
#     folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "PPL_filter")
#     filename1 = f"ppl{dt}_temp_{date.date()}_timeseries_validrange"
#     filename2 = f"ppl{dt}_temp_{date.date()}_timeseries_validrange_filtered"
#     savefig(fig2, folderpath, filename2)
#     savefig(fig1, folderpath, filename1)

# # ---Save new dataset

ds_new = ds.copy()
ds_new["temp_filtered"] = t_filtered
ds_new["wvmr_filtered"] = mr_filtered
ds_new["temp_max_range"] = (["time"], t_range)
ds_new["wvmr_max_range"] = (["time"], mr_range)
ds_new.attrs["maxrange_setting"]= f"Top down, in a vertial +/-100 pixel window at least {min_valid_fraction*100}% non-nan values"

ds_new.to_netcdf(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_{dt}_filtered_{min_valid_fraction*100}%valid.nc")
dstest = xr.open_dataset(rf"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_{dt}_filtered_{min_valid_fraction*100}%valid.nc")
print(dstest)
