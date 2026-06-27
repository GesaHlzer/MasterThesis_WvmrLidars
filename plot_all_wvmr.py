# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:44:58 2026

@author: alleh
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from basic_plot_funcions import savefig, grid_edges
from colormaps import cmap_wvmr

data_dial = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc") 
data_ppl1 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_90.0%valid.nc") 
data_ppl2 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc") 



date  = np.datetime64("2024-08-24")
start = date - np.timedelta64(1, 'm')
end   = date + np.timedelta64(1, 'D')
 
ds_dial = data_dial.sel(time=slice(start, end))#.values#.T
ds_ppl1 = data_ppl1.sel(time=slice(start, end))#.values#.T
ds_ppl2 = data_ppl2.sel(time=slice(start, end))#.values#.T

da10_maxrange = ds_dial['water_vapor_max_range']/1000
ppl1_maxrange = ds_ppl1['wvmr_max_range']/1000
ppl2_maxrange = ds_ppl2['wvmr_max_range']/1000

ds_dial = ds_dial['water_vapor']#.where(ds_dial['height'] < ds_dial['water_vapor_max_range'])#.values.T
ds_ppl1 = ds_ppl1['wvmr_filtered']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T
ds_ppl2 = ds_ppl2['wvmr_filtered']#.where(ds_ppl2['height'] < ds_ppl2['wvmr_max_range'])#.values.T
# ds_ppl1 = ds_ppl1['wvmr']#.where(ds_ppl1['height'] < ds_ppl1['wvmr_max_range'])#.values.T
# ds_ppl2 = ds_ppl2['wvmr']#.where(ds_ppl2['height'] < ds_ppl2['wvmr_max_range'])#.values.T


td, hd    = grid_edges(ds_dial['time'], ds_dial['height'])
tr1, hr1  = grid_edges(ds_ppl1['time'], ds_ppl1['height'])
tr2, hr2  = grid_edges(ds_ppl2['time'], ds_ppl2['height'])
ds_da10_f = ds_dial.values.T
ds_ppl1_f = ds_ppl1.values.T
ds_ppl2_f = ds_ppl2.values.T


filename = f"wvmr_timeseries_{date}_newcolor_v5.png"
filename = f"rl_wvmr_timeseries_{date}_newcolor_v1.png"
par_cmap  = cmap_wvmr()#'Blues'
Fontsize  = 20
endt      = end - np.timedelta64(1, 'ns')
# hmax_height_ratios = [4, 6, 6] #km AGL
hmax_height_ratios = [6, 6] #km AGL

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(18, 5*3))
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(16, 3*6), gridspec_kw={'height_ratios': hmax_height_ratios})
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(16, 2*5), gridspec_kw={'height_ratios': hmax_height_ratios})

# 1st subplot (dial)
# ax1.set_title(f"DIAL: {start.astype('datetime64[m]').astype(str)} - {endt.astype('datetime64[m]').astype(str)}", fontsize=Fontsize)
# ax1.set_title(f"DIAL: {date}", fontsize=Fontsize)
# pcm1  = ax1.pcolormesh(td, hd, ds_da10_f, shading='flat', cmap=par_cmap, vmin=0, vmax=15)
# cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.03, norm='log')
# cbar1.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
# cbar1.ax.tick_params(direction='out', labelsize=Fontsize)
# cbar1.ax.yaxis.set_major_locator(ticker.MultipleLocator(3)) 
# ax1.plot(ds_dial['time'], da10_maxrange, 'r')

# 2st subplot (ppl20min)
ax2.set_title(f"PPLS-20min: {date}", fontsize=Fontsize)
pcm2  = ax2.pcolormesh(tr2, hr2, ds_ppl2_f, shading='flat', cmap=par_cmap, vmin=0, vmax=15)
cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.03, norm='log')
cbar2.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
cbar2.ax.tick_params(direction='out', labelsize=Fontsize)
cbar2.ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) 
ax2.plot(ds_ppl2['time'], ppl2_maxrange, 'r')

# 3st subplot (ppl10s)
ax3.set_title(f"PPLS-10s: {date}", fontsize=Fontsize)
ax3.set_xlabel('time (UTC)', fontsize=Fontsize)
pcm3  = ax3.pcolormesh(tr1, hr1, ds_ppl1_f, shading='flat', cmap=par_cmap, vmin=0, vmax=15)                
cbar3 = plt.colorbar(pcm3, ax=ax3, pad=0.03, norm='log')
cbar3.set_label(r'wvmr (g kg$^{-1})$', size=Fontsize)
cbar3.ax.tick_params(direction='out', labelsize=Fontsize)
cbar3.ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) 
ax3.plot(ds_ppl1['time'], ppl1_maxrange, 'r')
    
# for ax, hmax in zip([ax1, ax2, ax3], hmax_height_ratios):
for ax, hmax in zip([ax2, ax3], hmax_height_ratios):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(direction='out', labelsize=Fontsize)

    ax.set_xlim([start, end])
    ax.set_ylim([0, hmax])        # ← each axis gets its own limit
    ax.set_ylabel('height (km AGL)', fontsize=Fontsize)
    ax.set_facecolor([0.8, 0.8, 0.8])

fig.align_ylabels()  
fig.tight_layout()
plt.show()

folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\LidarComparison"
savefig(fig, folderpath, filename, show=True)
