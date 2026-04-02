# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:33:12 2026

@author: alleh
"""
import xarray as xr
import numpy as np
from scipy import stats
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from basic_plot_funcions import savefig


data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")
# data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh100m.nc")

# start = np.datetime64("2024-08-23")
# end = np.datetime64("2024-09-09")
# data = data.where((data['date'] >= start) & (data['date'] <= end), drop=True)

def plot_raso_to_lidars(ds, daytime='day&night'):
    
    Fontsize = 13
    
    if daytime=='night':
        data = ds.sel(launch=ds['day_night'] == 'night')    # Night Sondes
    elif daytime=='day': 
        data = ds.sel(launch=ds['day_night'] == 'day')   # Day Sondes
    else: 
        data = ds.copy() # All Sondes
    
    # --- flatten and drop NaN pairs 
    def clean_pair(x, y):
        x, y = x.values.ravel(), y.values.ravel()
        mask = ~np.isnan(x) & ~np.isnan(y)
        return x[mask], y[mask]
    
    #data['rl2_wvmr_filtered'] = data['rl2_wvmr_filtered'].where(data.height > 0.2, np.nan)
    rs_dial, dial = clean_pair(data['rs_wvmr'], data['dial_wvmr'])
    rs_rl2,  rl2  = clean_pair(data['rs_wvmr'], data['rl2_wvmr_filtered'])
    
    # ###### LINGRESS
    # slope_dial, intercept_dial, r_dial, p_value_rl, std_err_rl  = stats.linregress(rs_dial, dial)
    # slope_rl2,  intercept_rl2,  r_rl2,  p_value_rl, std_err_rl  = stats.linregress(rs_rl2,  rl2)
    
    # ########## ODR
    def linear(params, x): # Define linear model
        slope, intercept = params
        return slope * x + intercept
    model = Model(linear)
     # --- Fit dial
    odr_data_dial = RealData(rs_dial, dial)  # RealData(rs_dial, dial, sx=unc_x, sy=unc_y)
    odr_dial = ODR(odr_data_dial, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
    out_dial = odr_dial.run()
    slope_dial, intercept_dial = out_dial.beta
    r_dial = np.corrcoef(rs_dial, dial)[0, 1]  # ODR doesn't return R², compute separately
    # --- Fit rl2
    odr_data_rl2 = RealData(rs_rl2, rl2)
    odr_rl2 = ODR(odr_data_rl2, model, beta0=[1, 0])
    out_rl2 = odr_rl2.run()
    slope_rl2, intercept_rl2 = out_rl2.beta
    r_rl2 = np.corrcoef(rs_rl2, rl2)[0, 1]


    def fmt_intercept(slope, intercept):
        sign = '+' if intercept >= 0 else '-'
        return f'y = {slope:.2f}x {sign} {abs(intercept):.2f}'
    
    label_dialfit = (f'Linear Fit for DA10:\n'
                 f'{fmt_intercept(slope_dial, intercept_dial)} (R² = {r_dial**2:.2f})')
    label_rlfit   = (f'Linear Fit for PPL:\n'
                 f'{fmt_intercept(slope_rl2, intercept_rl2)} (R² = {r_rl2**2:.2f})')
    # label_dialfit = (f'Linear Fit for PPLS (R² = {r_dial**2:.2f}:)' + '\n' +
    #               f"Slope = {slope_dial:.2f}, " + '\n' +
    #               f'Intercept = {intercept_dial:.2f}' )
    # label_dialfit = ('Linear Fit for PPL:' + '\n' +
    #                  f'y={slope_dial:.2f}x+{intercept_dial:.2f} (R² = {r_dial**2:.2f})')
    # label_dialfit = rf'DA10 fit: y={slope_dial:.2f}x+{intercept_dial:.2f}, \n $R^2$={r_dial**2:.2f}'
    # label_rlfit = (f'Linear Fit for PPLS (R² = {r_rl2**2:.2f}):' + '\n' +
    #               f"Slope = {slope_rl2:.2f},"  + '\n'+
    #               f'Intercept = {intercept_rl2:.2f}' )
    # label_rlfit = ('Linear Fit for PPL:' + '\n' +
    #                f'y={slope_rl2:.2f}x+{intercept_rl2:.2f} (R² = {r_rl2**2:.2f})')
    # label_rlfit = rf'PPL fit: y={slope_rl2:.2f}x+{intercept_rl2:.2f},\n $R^2$={r_rl2**2:.2f}'
    
    # --- Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # scatter
    ax.scatter(rs_dial, dial, color='darkorange', alpha=0.2, s=3, label='DA10')
    ax.scatter(rs_rl2,  rl2,  color='dodgerblue', alpha=0.2, s=3, label='PPL (20 min)')
    
    # 1:1 line
    lim = [0, 16] # [0, max(rs_dial.max(), rs_rl2.max(), dial.max(), rl2.max())]
    ax.plot(lim, lim, 'k--', lw=1.5, label='1:1')
    
    # fit lines
    x_fit = np.linspace(lim[0], lim[1], 200)
    ax.plot(x_fit, slope_dial * x_fit + intercept_dial,
            color='crimson', lw=2, linestyle='-', label=label_dialfit) #orangered
    ax.plot(x_fit, slope_rl2 * x_fit + intercept_rl2, 
            color='blue', lw=2, linestyle='-',label=label_rlfit) #blue
    
    # ledgend
    # ax.legend(fontsize=Fontsize-2, loc= "lower right")
    legend_handles = [ Line2D([0], [0], marker='o', color='w', label='DA10',
                      markerfacecolor='darkorange', alpha=0.8, markersize=4),
                      Line2D([0], [0], marker='o', color='w', label='PPL (20 min)', 
                      markerfacecolor='dodgerblue', alpha=0.8, markersize=4) ]
    handles, labels = ax.get_legend_handles_labels() # restliche handles
    handles[0] = legend_handles[0] #replace first two
    handles[1] = legend_handles[1]
    ax.legend(handles, labels, fontsize=Fontsize-2, loc='lower right')
    
    ax.set_title(f'DA10 & PPL vs. Radiosonde ({daytime})', fontsize=Fontsize)
    ax.set_xlabel(r'radiosonde wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_ylabel(r'lidar wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.tick_params(size=5)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    plt.show()  
    
    return fig


for daytime in ['day&night', 'night', 'day']:
    fig = plot_raso_to_lidars(data, daytime=daytime)
    
    filename = f'one-to-one_lidar_raso__{daytime}_ODR_dh10m.png'
    folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\OneToOne"
    savefig(fig, folderpath, filename, dpi=300, show=True)
    

# Method       Minimizes                       Assumes
# linregress  vertical distance (y residuals)  x is exact, only y has error
# ODR         perpendicular distance           both x and y have error

