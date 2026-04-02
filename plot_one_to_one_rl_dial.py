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
from basic_plot_funcions import savefig, classify_daytime


# data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl20min__dt1200s_dh100m_hmax6000m.nc")
data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl20min__dt1200s_dh10m_hmax6000m.nc")
# data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl10s__dt60s_dh10m_hmax6000m.nc")

def plot_rl_vs_dial(data, daytime='all'):
    
    Fontsize = 20
    day_class = classify_daytime(data)
    
    if daytime ==  'night':
        ds = data.sel(time=day_class == 'night')
    elif daytime == 'day': 
        ds = data.sel(time=day_class == 'day')
    elif daytime == 'twilight': 
        ds = data.sel(time=day_class == 'twilight')  
    else: 
        ds = data.copy() # All
    
    # --- drop NaN pairs preserving height ---
    
    # take arrays
    rl_vals   = ds['rl_wvmr_filtered'].values      # (launch, height)
    dial_vals = ds['dial_wvmr'].values 
    height_2d = np.broadcast_to(data['height'].values[np.newaxis, :], rl_vals.shape)
    
    # mask
    mask_valid_dial = (height_2d <= ds['dial_wvmr_maxrange'].values[:, None])  # Broadcast to (time, range)
    mask_valid_rl   = (height_2d <= ds['rl_wvmr_maxrange'].values[:, None])
    maskNaN = ~np.isnan(rl_vals) & ~np.isnan(dial_vals)
    mask = mask_valid_dial & mask_valid_rl & maskNaN
    
    rl_clean   = rl_vals[mask]
    dial_clean = dial_vals[mask]
    h_clean    = height_2d[mask] / 1000  # km
    
    
    # ###### LINGRESS
    # slope, intercept, r, p_value, std_err  = stats.linregress(rl_clean, dial_clean)
    
    # ########## ODR
    def linear(params, x): # Define linear model
        slope, intercept = params
        return slope * x + intercept 
    model = Model(linear)
    
    odr_data = RealData(rl_clean, dial_clean)  # RealData(rs_dial, dial, sx=unc_x, sy=unc_y)
    odr = ODR(odr_data, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
    out = odr.run()
    slope, intercept = out.beta
    r = np.corrcoef(rl_clean, dial_clean)[0, 1]  # ODR doesn't return R², compute separately

    def fmt_intercept(slope, intercept):
        sign = '+' if intercept >= 0 else '-'
        return f'y = {slope:.2f}x {sign} {abs(intercept):.2f}'
    label_fit = (f'Linear Fit (R² = {r**2:.2f}):\n'
                 f'{fmt_intercept(slope, intercept)}')
    
    lim = [0, 16]  # max(rs_dial.max(), rs_rl2.max(), dial.max(), rl2.max())
    x_fit = np.linspace(lim[0], lim[1], 200)
    
    # --- Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc = ax.scatter(rl_clean, dial_clean, c = h_clean, label='wvmr measuements',
                    cmap='managua', alpha=0.4, s=10, vmin=0, vmax=3.75) 
    cbar = plt.colorbar(sc, ax=ax) # ,fraction=0.046, pad=0.03
    cbar.set_label("height (km AGL)", fontsize=Fontsize)
    cbar.ax.tick_params(direction='out', labelsize=Fontsize, size=9) 
    cbar.solids.set_alpha(1.0)
    
    ax.plot(lim, lim, 'k--', lw=1.5, label='1:1')
    ax.plot(x_fit, slope * x_fit + intercept, color='red', lw=2, alpha=0.7, linestyle='-', label=label_fit) #orangered
    
    # # ledgend
    #ax.legend(fontsize=Fontsize-2, loc= "lower right")
    legend_handles = [Line2D([0], [0], marker='o', color='w', label='wvmr measuements',
                      markerfacecolor='peru', alpha=0.8, markersize=6)]
    handles, labels = ax.get_legend_handles_labels() # restliche handles
    handles[0] = legend_handles[0] #replace first two
    ax.legend(handles, labels, fontsize=Fontsize-2, loc='lower right')
    
    ax.set_xlabel(r'PPL wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_ylabel(r'DA10 wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=Fontsize-1, size=7)
    ax.grid(alpha=0.5)
    # ax.set_title(f'DA10 vs. PPLS ({daytime})', fontsize=Fontsize)
    ax.text(0.05, 0.95, f' DA10 vs. PPL ({daytime})', 
            transform=ax.transAxes, fontsize=Fontsize+2, verticalalignment='top',
       bbox=dict(boxstyle='round',edgecolor='none',facecolor='whitesmoke', alpha=0.9))
    fig.tight_layout()
    plt.show()  
    
    return fig

for daytime in ['all', 'night', 'day', 'twilight']:
    fig = plot_rl_vs_dial(data, daytime=daytime)
    
    filename = f'one-to-one_rl20min_dial__dt20min_dh10m_{daytime}_ODR.png'
    folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\OneToOne"
    savefig(fig, folderpath, filename, dpi=300, show=True)
    

# Method      Minimizes                        Assumes
# Linregress  vertical distance (y residuals)  x is exact, only y has error
# ODR         perpendicular distance           both x and y have error
