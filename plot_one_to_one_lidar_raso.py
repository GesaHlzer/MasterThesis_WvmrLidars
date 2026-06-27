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


def plot_raso_to_lidars(data, daytime='day&night', dial_var='dial_wvmr', rl_var='rl2_wvmr_filtered'):
    
    if daytime=='night':
        ds = data.sel(launch=data['day_night'] == 'night') # Night Sondes
    elif daytime=='day': 
        ds = data.sel(launch=data['day_night'] == 'day')   # Day Sondes
    else: 
        ds = data.copy() # All Sondes
    
    # --- flatten and drop NaN pairs 
    def clean_pair(x, y):
        x, y = x.values.ravel(), y.values.ravel()
        mask = ~np.isnan(x) & ~np.isnan(y)
        return x[mask], y[mask]
    
    def count_sondes(x, y):
        """Count Radiosondes, with at least 1 (non-NaN) data entry."""
        x_arr = x.values  # shape: (launch, height)
        y_arr = y.values
        mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)  # (launch, height)
        valid_launches = mask.any(axis=1).sum()       
        return int(valid_launches)

    #data['rl2_wvmr_filtered'] = data['rl2_wvmr_filtered'].where(data.height > 0.2, np.nan)
    rs_dial, dial = clean_pair(ds['rs_wvmr'], ds[dial_var])
    rs_rl2,  rl2  = clean_pair(ds['rs_wvmr'], ds[rl_var])
    
    n_sondes_dial = count_sondes(ds['rs_wvmr'], ds[dial_var])
    n_sondes_rl2  = count_sondes(ds['rs_wvmr'], ds[rl_var])
    
    # ###### LINGRESS
    # slope_dial, intercept_dial, r_dial, p_value_rl, std_err_rl  = stats.linregress(rs_dial, dial)
    # slope_rl2,  intercept_rl2,  r_rl2,  p_value_rl, std_err_rl  = stats.linregress(rs_rl2,  rl2)
    
    # ########## ODR
    def linear(params, x): # Define linear model
        slope, intercept = params
        return slope * x + intercept
    model = Model(linear)
    
    # --- Fit dial
    print('DIAL:\n')
    odr_data_dial = RealData(rs_dial, dial)  # RealData(rs_dial, dial, sx=unc_x, sy=unc_y)
    odr_dial = ODR(odr_data_dial, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
    out_dial = odr_dial.run()
    slope_dial, intercept_dial = out_dial.beta
    r_dial = np.corrcoef(rs_dial, dial)[0, 1]  # ODR doesn't return R², compute separately
    
    out_dial.pprint()
    daslope, daintercept         = out_dial.beta
    daslope_err, daintercept_err = out_dial.sd_beta
    dacov                        = out_dial.cov_beta
    daresidual_variance          = out_dial.res_var        # The mean squared orthogonal residual — i.e. the average squared perpendicular distance from each point to the fit line, 
    dainv_condition              = out_dial.inv_condnum
    
    ### statistics
    # r_orig = np.corrcoef(rl_clean, dial_clean)[0, 1]  # Stichproben-Pearson-r mit N-1
    dar, dap_value = stats.pearsonr(rs_dial, dial) # Stichproben-Pearson-r mit N-1
    dar2   = dar**2 
    dadiff = dial - rs_dial
    dabias = np.mean(dadiff)                    # Mean Difference (signed)
    dastd  = np.std(dadiff, ddof=1)
    daMAE  = np.mean(np.abs(dadiff))            # Mean Absolute Error
    daRMSE = np.sqrt(np.mean(dadiff**2))        # Root Mean Square Error
    
    print(f"Amount of sondes included: {n_sondes_dial}")
    print(f"Amount of datapoints: {len(dial)}")
    print(f'R = {dar:.3f},\nR² = {dar2:.3f}')
    print(f'MD = DIAL-RASO = {dabias:.3f} g/kg')
    print(f'Sigma_mad = {dastd:.3f} g/kg')
    print(f'MAD  = {daMAE:.3f} g/kg')
    print(f'RMSD = {daRMSE:.3f} g/kg')
    print(f'Square root of Residual Variance: {np.sqrt(daresidual_variance):.3f}')
    print('-----------')
    print('')
    
    # --- Fit rl2
    print('RL2:\n')
    odr_data_rl2 = RealData(rs_rl2, rl2)
    odr_rl2 = ODR(odr_data_rl2, model, beta0=[1, 0])
    out_rl2 = odr_rl2.run()
    slope_rl2, intercept_rl2 = out_rl2.beta
    r_rl2 = np.corrcoef(rs_rl2, rl2)[0, 1]
    
    out_rl2.pprint()
    rlslope, rlintercept         = out_rl2.beta
    rlslope_err, rlintercept_err = out_rl2.sd_beta
    rlcov                        = out_rl2.cov_beta
    rlresidual_variance          = out_rl2.res_var        # The mean squared orthogonal residual — i.e. the average squared perpendicular distance from each point to the fit line, 
    rlinv_condition              = out_rl2.inv_condnum
    
    ### statistics
    # r_orig = np.corrcoef(rl_clean, dial_clean)[0, 1]  # Stichproben-Pearson-r mit N-1
    rlr, rlp_value = stats.pearsonr(rs_rl2, rl2) # Stichproben-Pearson-r mit N-1
    rlr2   = rlr**2 
    rldiff = rl2 - rs_rl2
    rlbias = np.mean(rldiff)                    # Mean Difference (signed)
    rlstd  = np.std(rldiff, ddof=1)
    rlMAE  = np.mean(np.abs(rldiff))            # Mean Absolute Error
    rlRMSE = np.sqrt(np.mean(rldiff**2))        # Root Mean Square Error
    
    print(f"Amount of sondes included: {n_sondes_rl2}")
    print(f"Amount of datapoints: {len(rl2)} ")
    print(f'R = {rlr:.3f},\nR² = {rlr2:.3f}')
    print(f'MD = PPLS-RASO = {rlbias:.3f} g/kg')
    print(f'Sigma_mad = {rlstd:.3f} g/kg')
    print(f'MAD  = {rlMAE:.3f} g/kg')
    print(f'RMSD = {rlRMSE:.3f} g/kg')
    print(f'Square root of Residual Variance: {np.sqrt(rlresidual_variance):.3f}')
    print('-----------')
    print('')
    
    def fmt_intercept(slope, intercept):
        sign = '+' if intercept >= 0 else '-'
        return f'y = {slope:.2f}x {sign} {abs(intercept):.2f}'
    
    label_dialfit = (f'DA10 fit: {fmt_intercept(slope_dial, intercept_dial)}')
    label_rlfit   = (f'PPLS  fit: {fmt_intercept(slope_rl2, intercept_rl2)}')
    # label_dialfit = (f'Linear Fit for DA10:\n'
    #              f'{fmt_intercept(slope_dial, intercept_dial)}')# (R² = {r_dial**2:.2f})')
    # label_rlfit   = (f'Linear Fit for PPL:\n'
                 # f'{fmt_intercept(slope_rl2, intercept_rl2)}')#' (R² = {r_rl2**2:.2f})')
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
    Fontsize = 15
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # scatter
    ax.scatter(rs_dial, dial, color='darkorange', 
               alpha=0.2, s=3, label='DA10')
    ax.scatter(rs_rl2,  rl2,  color='dodgerblue', 
               alpha=0.2, s=3, label='PPLS 20-min') #label=f'PPLS 20-min (nan<{yc})')
    
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
    legend_handles = [ Line2D([0], [0], marker='o', color='w',
                              # label=f'DA10 (n={len(dial)})',
                      markerfacecolor='darkorange', alpha=0.8, markersize=4),
                      Line2D([0], [0], marker='o', color='w', 
                             # label=f'PPLS 20-min (n={len(rl2)})', 
                      markerfacecolor='dodgerblue', alpha=0.8, markersize=4) ]
    handles, labels = ax.get_legend_handles_labels() # restliche handles
    handles[0] = legend_handles[0] #replace first two
    handles[1] = legend_handles[1]
    ax.legend(handles, labels, fontsize=Fontsize-3, loc='lower right')
    
    ax.text(0.05, 0.95, f'DA10 & PPLS vs. \nRadiosonde ({daytime})', 
            transform=ax.transAxes, fontsize=Fontsize+3, verticalalignment='top',
       bbox=dict(boxstyle='round',edgecolor='none',facecolor='whitesmoke', alpha=0.9))
    
    # ax.set_title(f'DA10 & PPLS vs. Radiosonde ({daytime})', fontsize=Fontsize)
    ax.set_xlabel(r'radiosonde wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_ylabel(r'lidar wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=Fontsize-1, size=6)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    plt.show()  
    
    return fig


# This part only runs when you execute the file directly:
if __name__ == "__main__":
    
    
    data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")
    # data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh100m.nc")
    
    # start = np.datetime64("2024-08-23")
    # end = np.datetime64("2024-09-09")
    # data = data.where((data['date'] >= start) & (data['date'] <= end), drop=True)
    # yc= 150
    # data['rl2_wvmr_filtered'] = xr.where(data.height < yc, np.nan, data['rl2_wvmr_filtered'])
    # data['rl2_wvmr_filtered'] = data['rl2_wvmr_filtered'].T
    for daytime in ['day&night', 'night', 'day']:
        print('')
        print(daytime)
        fig = plot_raso_to_lidars(data, daytime=daytime)
        
        filename = f'one-to-one_lidar_raso__{daytime}_ODR_dh10m_v2.png'
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\OneToOne"
        savefig(fig, folderpath, filename, dpi=300, show=True)
        
    # 
    # Method       Minimizes                       Assumes
    # linregress  vertical distance (y residuals)  x is exact, only y has error
    # ODR         perpendicular distance           both x and y have error

