# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:32:46 2026

@author: alleh
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from basic_plot_funcions import savefig, grid_edges

# data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl20min__dt1200s_dh10m_hmax6000m.nc")
data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl10s__dt60s_dh10m_hmax6000m.nc") 

#date  = np.datetime64("2024-08-24")

def diff_plot_combined(data, date):
    
    start = date
    end   = date + np.timedelta64(1, 'D')
    
    # dial = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
    # dial         = dial.sel(time=slice(start, end)) 
    # dial_orig = dial['water_vapor'].where(dial['height'] < dial['water_vapor_max_range']).values.T
    # t1, h1       = grid_edges(dial['time'], dial['height'])
    
    # ppl  = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc")
    # ppl          = ppl.sel(time=slice(start, end))
    # ppl_orig  = ppl['wvmr'].where(ppl['height'] < ppl['wvmr_max_range']).values.T
    # t2, h2       = grid_edges(ppl['time'], ppl['height'])
    
    ds        = data.sel(time=slice(start, end)) 
    ds_ppl_f  = ds['rl_wvmr_filtered'].where(data['height'] < data['rl_wvmr_maxrange']).values.T
    ds_da10_f = ds['dial_wvmr']       .where(data['height'] < data['dial_wvmr_maxrange']).values.T
    diff      = ds_da10_f - ds_ppl_f
    t, h      = grid_edges(ds['time'], ds['height'])
    
    par_cmap  = 'Blues'
    par_cmap2 = 'PuOr'
    
    Fontsize  = 19
    hmax      = 3.5
    endt      = end - np.timedelta64(1, 'ns')
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(18, 5*3))
    
    # 1st subplot (dial)
    ax1.set_title(f"DIAL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    pcm1  = ax1.pcolormesh(t, h, ds_da10_f, shading='flat', cmap=par_cmap, vmin=2, vmax=18)
    cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.03, norm='log')
    cbar1.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
    cbar1.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar1.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
    # 2st subplot (ppl)
    ax2.set_title(f"PPL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    pcm2  = ax2.pcolormesh(t, h, ds_ppl_f, shading='flat', cmap=par_cmap, vmin=2, vmax=18)
    cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.03, norm='log')
    cbar2.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
    cbar2.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar2.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
    # 3st subplot (dial-ppl)
    ax3.set_title(f"DIAL - PPL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    ax3.set_xlabel('time (UTC)', fontsize=Fontsize)
    pcm3  = ax3.pcolormesh(t, h, diff, shading='flat', cmap=par_cmap2, vmin=-3, vmax=3)                
    cbar3 = plt.colorbar(pcm3, ax=ax3, pad=0.03, norm='log')
    cbar3.set_label(r'Δwvmr (g kg$^{-1})$', size=Fontsize)
    cbar3.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar3.ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
    
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.tick_params(direction='out', labelsize=Fontsize)
        
        ax.set_xlim([start, end])
        ax.set_ylim([0, hmax])
        #ax.set_xlabel('time (UTC)', fontsize=Fontsize)
        ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
        ax.set_facecolor([0.8, 0.8, 0.8])
    
    fig.align_ylabels()  
    fig.tight_layout()
    plt.show()
    return fig
    
def diff_plot_combined_unf(data, date):
    
    start = date
    end   = date + np.timedelta64(1, 'D')
    
    # dial = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
    # dial         = dial.sel(time=slice(start, end)) 
    # dial_orig = dial['water_vapor'].where(dial['height'] < dial['water_vapor_max_range']).values.T
    # t1, h1       = grid_edges(dial['time'], dial['height'])
    
    # ppl  = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc")
    # ppl          = ppl.sel(time=slice(start, end))
    # ppl_orig  = ppl['wvmr'].where(ppl['height'] < ppl['wvmr_max_range']).values.T
    # t2, h2       = grid_edges(ppl['time'], ppl['height'])
    
    ds        = data.sel(time=slice(start, end)) 
    ds_ppl_f  = ds['rl_wvmr_filtered'].where(data['height'] < data['rl_wvmr_maxrange']).values.T
    ds_da10_f = ds['dial_wvmr']       .where(data['height'] < data['dial_wvmr_maxrange']).values.T
    diff      = ds_da10_f - ds_ppl_f
    t, h      = grid_edges(ds['time'], ds['height'])
    
    ds_ppl    = ds['rl_wvmr'].values.T
    ds_da10   = ds['dial_wvmr'].values.T
    
    par_cmap  = 'Blues'
    par_cmap2 = 'PuOr'
    
    Fontsize  = 19
    hmax      = 3.5
    endt      = end - np.timedelta64(1, 'ns')
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(18, 5*3))
    
    # 1st subplot (dial)
    ax1.set_title(f"DIAL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    # pcm1  = ax1.pcolormesh(t, h, ds_da10_f, shading='flat', cmap=par_cmap, vmin=2, vmax=18)
    pcm1  = ax1.pcolormesh(t, h, ds_da10, shading='flat', cmap=par_cmap, vmin=2, vmax=18)
    ax1.plot(data['time'].values, data['dial_wvmr_maxrange'].values/1000, 'r')                
    # pcm1  = ax1.pcolormesh(t1, h1, ds_dial_orig, shading='flat', cmap=par_cmap, vmin=2, vmax=18)                
    # ax1.plot(dial['time'].values, dial['water_vapor_uncertainty'].values, 'r')
    cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.03, norm='log')
    cbar1.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
    cbar1.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar1.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
    # 2st subplot (ppl)
    ax2.set_title(f"PPL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    # pcm2  = ax2.pcolormesh(t, h, ds_ppl_f, shading='flat', cmap=par_cmap, vmin=2, vmax=18)
    pcm2  = ax2.pcolormesh(t, h, ds_ppl, shading='flat', cmap=par_cmap, vmin=2, vmax=18)  
    ax2.plot(data['time'].values, data['rl_wvmr_maxrange'].values/1000, 'r')          
    # pcm2  = ax2.pcolormesh(t2, h2, ds_ppl_orig, shading='flat', cmap=par_cmap, vmin=2, vmax=18)   
    # ax2.plot(ppl['time'].values, ppl['wvmr_max_range'].values, 'r')                  
    cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.03, norm='log')
    cbar2.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
    cbar2.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar2.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
    # 3st subplot (dial-ppl)
    ax3.set_title(f"DIAL - PPL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
    ax3.set_xlabel('time (UTC)', fontsize=Fontsize)
    pcm3  = ax3.pcolormesh(t, h, diff, shading='flat', cmap=par_cmap2, vmin=-3, vmax=3)                
    cbar3 = plt.colorbar(pcm3, ax=ax3, pad=0.03, norm='log')
    cbar3.set_label(r'Δwvmr (g kg$^{-1})$', size=Fontsize)
    cbar3.ax.tick_params(direction='out', labelsize=Fontsize)
    cbar3.ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
    
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.tick_params(direction='out', labelsize=Fontsize)
        
        ax.set_xlim([start, end])
        ax.set_ylim([0, hmax])
        #ax.set_xlabel('time (UTC)', fontsize=Fontsize)
        ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
        ax.set_facecolor([0.8, 0.8, 0.8])
    
    fig.align_ylabels()  
    fig.tight_layout()
    plt.show()
    return fig
 

dates = [np.datetime64("2024-08-23") + np.timedelta64(x, 'D') for x in range(0, 17)]

for date in dates: 
    try:
        # fig = diff_plot_combined(data, date)
        fig = diff_plot_combined_unf(data, date)
        
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\LidarComparison"
        filename = f"wvmr_diff_timeseries_h10m_t10s_{date}.png"
        savefig(fig, folderpath, filename, show=True)
        
    except Exception as e:
        print(f"Error plotting data: {e}")
        continue



#%%
# # timeseries
# def plot_mr_dial(ax, data, start, end, hmax, Fontsize = 22):
    
#     print("\n Making da10 mr plot...")
#     mr_da10 = data.wvmr_dial.to_numpy().transpose()

#     # Extend x (time array) and y (height array) to include first and last edges
#     time = data['time'].values
#     heights = data['height']
#     t, h = grid_edges(time, heights)
    
#     par_cmap = 'Blues' #'cubehelix_r' # colormaps.cmap_adv_div_brown_green() 
#     # Fontsize = 22
#     # hmax = 3.5
    
#     # ---- make Subplot
    
#     #fig, ax = plt.subplots(figsize=(18, 5))
#     pcm = ax.pcolormesh(t, h, mr_da10, shading='flat', cmap=par_cmap, vmin=2, vmax=18)                
#     cbar = plt.colorbar(pcm, ax=ax, pad=0.03, norm='log')
#     cbar.set_label(r'wvmr in g kg$^{-1}$', size=Fontsize)
#     cbar.ax.tick_params(direction='out', labelsize=Fontsize)
#     cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
#     ax.set_xlim([start, end])
#     ax.set_ylim([0, hmax])
#     ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#     ax.tick_params(direction='out', labelsize=Fontsize)

#     # Add Title and Labels
#     dialABS_title_text = f"DIAL: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}" # .strftime('%H UTC %d-%m-%Y')
#     ax.set_title(dialABS_title_text, fontsize=Fontsize)
#     #ax.set_xlabel('time (UTC)', fontsize=Fontsize)
#     ax.set_ylabel('height in km AGL', fontsize=Fontsize)
    
#     ax.set_facecolor([0.8, 0.8, 0.8])
    
#     # plt.tight_layout()
#     # plt.show()
    
#     return ax  

# def plot_mr_rl(ax, data, start, end, hmax, fontsize = 22):
    
#     #print("Making da10 mr plot...")
#     mr_ppl = data.wvmr_rl.to_numpy().transpose()

#     # Extend x (time array) and y (height array) to include first and last edges
#     time = data['time'].values
#     heights = data['height']
#     t, h = grid_edges(time, heights)
    
#     par_cmap = 'Blues'# 'cubehelix_r' # colormaps.cmap_adv_div_brown_green() 
#     # fontsize = 22
#     # hmax = 3.5
    
#     # ---- make Subplot
#     #'water vapor mixing ratio (g/kg)' #r'wvmr in g kg$^{-1}$'
#     # fig, ax = plt.subplots(figsize=(18, 5))
#     pcm = ax.pcolormesh(t, h, mr_ppl, shading='flat', cmap=par_cmap, vmin=2, vmax=18)                
#     cbar = plt.colorbar(pcm, ax=ax, pad=0.03, norm='log')
#     cbar.set_label(r'wvmr in g kg$^{-1}$', size=fontsize)
#     cbar.ax.tick_params(direction='out', labelsize=fontsize)
#     cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
    
#     ax.set_xlim([start, end])
#     ax.set_ylim([0, hmax])
#     ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#     ax.tick_params(direction='out', labelsize=fontsize)

#     # Add Title and Labels
#     dialABS_title_text = f"PPLS: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}" # .strftime('%H UTC %d-%m-%Y')
#     ax.set_title(dialABS_title_text, fontsize=fontsize)
#     #ax.set_xlabel('time (UTC)', fontsize=fontsize)
#     ax.set_ylabel('height in km AGL', fontsize=fontsize)
    
#     ax.set_facecolor([0.8, 0.8, 0.8])
    
#     # plt.tight_layout()
#     # plt.show()
    
#     return ax  

# def plot_mr_diff(ax, data, start, end, hmax, fontsize = 22):
    
#     # --- for independent plotting: uncomment
#     # hmax = 3.5
#     # date  = datetime(2024, 8, 23)
#     # start =  np.datetime64(date, 'ns') 
#     # end   =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
#     # data  = dataset.sel(time=slice(start, end)) 
#     # fig, ax = plt.subplots(figsize=(18, 5))
    
#     print("Making mr diff plot...")
#     mr_diff = data.wvmr_diff.to_numpy().transpose()

#     # Extend x (time array) and y (height array) to include first and last edges
#     time = data['time'].values
#     heights = data['height']
#     t, h = grid_edges(time, heights)
    
#     par_cmap = 'PuOr' #'viridis' #
#     cmin, cmax = -2.5, 3
#     # fontsize = 22
    
#     # ---- make Subplot
#     pcm = ax.pcolormesh(t, h, mr_diff, shading='flat', 
#                         cmap=par_cmap, vmin=cmin, vmax=cmax)                
#     cbar = plt.colorbar(pcm, ax=ax, pad=0.03, norm='log')
#     cbar.set_label(r'Δwvmr in g kg$^{-1}$', size=fontsize)
#     cbar.ax.tick_params(direction='out', labelsize=fontsize)
#     cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
    
#     ax.set_xlim([start, end])
#     ax.set_ylim([0, hmax])
#     ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#     ax.tick_params(direction='out', labelsize=fontsize)

#     # Add Title and Labels
#     title_text = f"DIAL - PPLS: {start.astype('datetime64[m]').astype(str)} - {end.astype('datetime64[m]').astype(str)}" # .strftime('%H UTC %d-%m-%Y')
#     ax.set_title(title_text, fontsize=fontsize)
#     ax.set_xlabel('time in UTC', fontsize=fontsize)
#     ax.set_ylabel('height in km AGL', fontsize=fontsize)
    
#     ax.set_facecolor([0.8, 0.8, 0.8])
    
#     # plt.tight_layout()
#     # plt.show()
    
#     return ax  

# Fontsize = 23
# hmax     = 3.5
# date     = np.datetime64("2024-08-24")
# dates    = [np.datetime64("2024-08-23") + np.timedelta64(x, 'D') for x in range(0, 17)]

# for date in dates:
#     print(date)
#     start =  np.datetime64(date, 'ns') 
#     end   =  np.datetime64(date, 'ns') + np.timedelta64(1, 'D')
#     ds    = data.sel(time=slice(start, end)) 

#     try:
#         n = 3
#         fig, (ax1, ax2, ax3) = plt.subplots(nrows=n, figsize=(18, 5*n))
        
#         ax1 = plot_mr_dial(ax1, ds, start, end, hmax, fontsize=Fontsize)
#         ax2 = plot_mr_rl  (ax2, ds, start, end, hmax, fontsize=Fontsize)
#         ax2 = plot_mr_diff(ax3, ds, start, end, hmax, fontsize=Fontsize)
#         fig.align_ylabels()  
#         fig.tight_layout()
#         plt.show()
        
        
#         folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\LidarComparison"
#         filename = f"wvmr_diff_timeseries_h10m_t20min_{date.date()}.png"
#         savefig(fig, folderpath, filename)
        
#     except Exception as e:
#         print(f"Error plotting data: {e}")
#         continue

# #-------- Difference Timeseries -----------------

# # plot_mr_diff_timeseies(dataset, save)

# # folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Lidar_Comparison", "DA10_vs_PPL_per_day")
# # filename = f"da10_vs_ppl_from23-08.png"
# # savefig(fig, folderpath, filename)


# # mr_da10  = dataset.wvmr_dial
# # mr_ppl   = dataset.wvmr_rl
# # mr_diff  = dataset.wvmr_diff
# # startend = np.array([datetime(2024, 8, 23), datetime(2024, 8, 23)], dtype="datetime64[ns]")
# # mr_da10  = mr_da10.sel(time=slice(startend[0], startend[1]  + np.timedelta64(59,  'm')))   
# # mr_ppl  = mr_ppl.sel(time=slice(startend[0], startend[1]  + np.timedelta64(59,  'm')))   
# # xdatas  = mr_da10.copy()
# # ydatas  = mr_ppl.copy()


# # selected_day = mr_ppl.sel(time='2024-08-23')
# # xdata = xdatas.sel(time=day_str)
# # ydata = ydatas.sel(time=day_str)


# # fig = plot_x_y(mr_da10, mr_ppl)
# # folderpath = os.path.join(os.path.dirname(os.getcwd()), "plots", "Lidar_Comparison", "DA10_vs_PPL_per_day")
# # filename = f"da10_vs_ppl_from23-08.png"
# # savefig(fig, folderpath, filename)
    
   
# # plot_x_y_at_day(mr_da10, mr_ppl, save)
# # plot_x_y_at_h(mr_da10, mr_ppl, save)


