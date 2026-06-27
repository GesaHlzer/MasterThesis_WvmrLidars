# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:31:49 2026

@author: alleh
"""
import numpy as np
import xarray as xr
from plot_one_to_one_lidar_raso import plot_raso_to_lidars
from plot_one_to_one_rl_dial import plot_rl_vs_dial
from plot_vertical_mean_difference import stat_computation, plot_stats_v2
from plot_difference_timeseries import diff_plot_combined

data_1d_088 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1d_all_with_CorrectedDial.nc")
data_1d_086 = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1d_all_with_CorrectedDial086.nc")
data_2d = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2d_all_with_CorrectedDial.nc")
# ds = data_1d.sel(launch=90)


# %% plot_rl_vs_dial
data = data_2d.copy()
daytime = ['all', 'night', 'day', 'twilight']
i=1
plot_rl_vs_dial(data, daytime=daytime[i], dial_var='dial_wvmr', rl_var='rl_wvmr')
plot_rl_vs_dial(data, daytime=daytime[i], dial_var='dial088_wvmr', rl_var='rl_wvmr')
plot_rl_vs_dial(data, daytime=daytime[i], dial_var='dial086_wvmr', rl_var='rl_wvmr')

# %% plot_vertical_mean_difference

## data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")
data = data_1d_088.copy()
data = data_1d_086.copy()

ds, daytime = data.sel(launch=data['day_night']== 'night'), 'night'# Night Sondes
ds, daytime = data.sel(launch=data['day_night'] == 'day') , 'day'  # Day Sondes
ds, daytime = data.copy()                                 , 'day&night'                  # All Sondes

lidar_var = 'dial_wvmr' 
lidar_var = 'dial088_wvmr'
# lidar_var = 'rl_wvmr'

hmax=4

stats = stat_computation(ds, lidar_var)
fig = plot_stats_v2(stats, lidar_var, daytime, hmax)

# %% diff_plot_combined(data_dial, data_ppl, date)

data = data_2d.copy()

data_ppl    = data['rl_wvmr']
data_dial   = data['dial_wvmr'] 
data_dial088 = data['dial088_wvmr']
data_dial086 = data['dial086_wvmr']

date  = np.datetime64("2024-08-31")
fig = diff_plot_combined(data_dial, data_ppl, date)
fig = diff_plot_combined(data_dial088, data_ppl, date)
fig = diff_plot_combined(data_dial086, data_ppl, date)

# %% plot_raso_to_lidars
data = data_1d_088.copy()
data086 = data_1d_086.copy()

fig = plot_raso_to_lidars(data,    daytime='day&night', dial_var='dial_wvmr',    rl_var='rl_wvmr')
fig = plot_raso_to_lidars(data,    daytime='day&night', dial_var='dial088_wvmr', rl_var='rl_wvmr')
fig = plot_raso_to_lidars(data086, daytime='day&night', dial_var='dial088_wvmr', rl_var='rl_wvmr')
