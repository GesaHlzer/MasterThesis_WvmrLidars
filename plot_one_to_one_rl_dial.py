# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:33:12 2026

@author: alleh
"""
import xarray as xr
import numpy as np
from scipy import stats
from scipy.odr import ODR, Model, RealData, Data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from basic_plot_funcions import savefig, classify_daytime


def plot_rl_vs_dial(data, daytime='all', dial_var='dial_wvmr', rl_var='rl_wvmr_filtered'):
    
    print(daytime)
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
    rl_vals   = ds[rl_var].values      # (launch, height)
    dial_vals = ds[dial_var].values 
    height_2d = np.broadcast_to(data['height'].values[np.newaxis, :], rl_vals.shape)
    
    # mask
    if rl_var=='rl_wvmr_filtered' or rl_var=='rl2_wvmr_filtered':
        mask_valid_dial = (height_2d <= ds['dial_wvmr_maxrange'].values[:, None])  # Broadcast to (time, range)
        mask_valid_rl   = (height_2d <= ds['rl_wvmr_maxrange'].values[:, None])
        maskNaN = ~np.isnan(rl_vals) & ~np.isnan(dial_vals)
        mask = mask_valid_dial & mask_valid_rl & maskNaN
    else:
        mask = ~np.isnan(rl_vals) & ~np.isnan(dial_vals)
        
    rl_clean   = rl_vals[mask]
    dial_clean = dial_vals[mask]
    h_clean    = height_2d[mask] / 1000  # km
    
    ###### LINGRESS
    # slope, intercept, r, p_value, std_err  = stats.linregress(rl_clean, dial_clean)
    
    # ########## ODR
    def linear(params, x): # Define linear model
        slope, intercept = params
        return slope * x + intercept 
    model = Model(linear)

    odr_data = RealData(rl_clean, dial_clean)         # RealData(rs_dial, dial, sx=unc_x, sy=unc_y)
    odr      = ODR(odr_data, model, beta0=[1, 0])     # beta0: initial guess [slope, intercept]
    odr_out  = odr.run()
    
    odr_out.pprint()
   
    slope, intercept         = odr_out.beta
    slope_err, intercept_err = odr_out.sd_beta
    cov                      = odr_out.cov_beta
    residual_variance        = odr_out.res_var        # The mean squared orthogonal residual — i.e. the average squared perpendicular distance from each point to the fit line, 
    inv_condition            = odr_out.inv_condnum
    
    ### statistics
    # r_orig = np.corrcoef(rl_clean, dial_clean)[0, 1]  # Stichproben-Pearson-r mit N-1
    r, p_value = stats.pearsonr(rl_clean, dial_clean) # Stichproben-Pearson-r mit N-1
    r2 = r**2 
    diff = dial_clean - rl_clean
    bias = np.mean(diff)                    # Mean Difference (signed)
    std = np.std(diff, ddof=1)
    MAE  = np.mean(np.abs(diff))            # Mean Absolute Error
    RMSE = np.sqrt(np.mean(diff**2))        # Root Mean Square Error
    print(f'square root of Residual Variance: {np.sqrt(residual_variance):.3f}')
    print(f' R = {r:.3f},\n R² = {r2:.3f},\n p = {p_value:.3f}')
    print(f'MD = DIAL-PPLS = {bias:.3f} g/kg')
    print(f'sigma_mad = {std:.3f} g/kg')
    print(f'MAD  = {MAE:.3f} g/kg')
    print(f'RMSD = {RMSE:.3f} g/kg')
    print(f"amount of datapoints: {len(rl_clean)}")
    # # WEIGHTS (opt) 
      # Wie gut stimmen DA10 und PPL generell überein → Dichte-Gewichtung nach wvmr macht Sinn
      # Ist der Bias höhenabhängig → Höhengewichtung
      # Fit der alle Bedingungen gleich repräsentiert → beide kombiniert
      # Ungewichtet: "Wie gut stimmen DA10 und PPL über alle Messungen hinweg überein?" → repräsentativ für den tatsächlichen Betrieb
        ### Anzahl der Messwerte pro Höhenbin
        # height_bins = np.arange(0, h_clean.max() + 0.01, 0.01)  # 10m bins
        # bin_indices = np.digitize(h_clean, height_bins)
        # bin_counts  = np.bincount(bin_indices, minlength=len(height_bins))
        # # Gewicht = 1 / Anzahl im gleichen Bin → gleiche Gesamtgewichtung pro Höhe
        # weights_h = 1.0 / bin_counts[bin_indices].astype(float)
        ### Berechne Dichte nach wvmr
        # wvmr_bins = np.arange(0, max(rl_clean.max(), dial_clean.max()) + 0.1, 0.1)  # 0.1 g/kg bins
        # bin_indices_wvmr = np.digitize(rl_clean, wvmr_bins)
        # bin_counts_wvmr  = np.bincount(bin_indices_wvmr, minlength=len(wvmr_bins))
        # weights_wv = 1.0 / bin_counts_wvmr[bin_indices_wvmr].astype(float)
        ### Kombiiert
        # weights = weights_wv * weights_h  # aus Option 2 oben
        # odr_data2 = Data(rl_clean, dial_clean, we=weights, wd=weights)
        # odr2 = ODR(odr_data2, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
        # out2 = odr2.run()
        # slope2, intercept2 = out2.beta
        # odr_data_h = Data(rl_clean, dial_clean, we=weights_h, wd=weights_h)
        # odr_h = ODR(odr_data_h, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
        # out_h = odr_h.run()
        # slope_h, intercept_h = out_h.beta
        # odr_data_wv = Data(rl_clean, dial_clean,we=weights_wv, wd=weights_wv)
        # odr_wv = ODR(odr_data_wv, model, beta0=[1, 0])  # beta0: initial guess [slope, intercept]
        # out_wv = odr_wv.run()
        # slope_wv, intercept_wv = out_wv.beta   


    def label_fit(slope=slope, intercept=intercept, slope_err=slope_err, intercept_err=intercept_err, residual_variance=residual_variance):
        sign = '+' if intercept >= 0 else '-'
        # R² = {r**2:.2f})
        label_fit = (f'ODR fit: y = {slope:.3f}x {sign} {abs(intercept):.3f}')
        # label_fit = (f'Linear Fit (ODR) \n y = {slope:.2f}x {sign} {abs(intercept):.2f})
        # f"slope     = {slope:.3f} ± {slope_err:.3f} \n "
        # "intercept  = {intercept:.3f} ± {intercept_err:.3f} g/kg \n"
        #f"res. var.  = {residual_variance:.3f} (g/kg)²")
        return label_fit
    

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
    ax.plot(x_fit, slope * x_fit + intercept, color='red', lw=2, alpha=0.7, linestyle='-', label=label_fit()) #orangered
    # weights
    # ax.plot(x_fit, slope2 * x_fit + intercept2, color='red', lw=2, alpha=0.5, linestyle='-.', label=label_fit(slope2, intercept2, ' h & wv')) #orangered
    # ax.plot(x_fit, slope_h * x_fit + intercept_h, color='red', lw=2, alpha=0.5, linestyle=':', label=label_fit(slope_h, intercept_h, ' h')) #orangered
    # ax.plot(x_fit, slope_wv * x_fit + intercept_wv, color='red', lw=2, alpha=0.5, linestyle='--', label=label_fit(slope_wv, intercept_wv, ' wv')) #orangered

    
    # # ledgend
    #ax.legend(fontsize=Fontsize-2, loc= "lower right")
    legend_handles = [Line2D([0], [0], marker='o', color='w', label='wvmr measuements',
                      markerfacecolor='peru', alpha=0.8, markersize=6)]

    # Create handles manually
    handles, labels = ax.get_legend_handles_labels() # restliche handles
    handles[0] = legend_handles[0] #replace first two
    ax.legend(handles, labels, fontsize=Fontsize-2, loc='lower right')
    
    ax.set_xlabel(r'PPLS wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_ylabel(r'DA10 wvmr (g kg$^{-1}$)', fontsize=Fontsize)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=Fontsize-1, size=7)
    ax.grid(alpha=0.5)
    # ax.set_title(f'DA10 vs. PPLS ({daytime})', fontsize=Fontsize)
    ax.text(0.05, 0.95, f' DA10 vs. PPLS ({daytime})', 
            transform=ax.transAxes, fontsize=Fontsize+2, verticalalignment='top',
       bbox=dict(boxstyle='round',edgecolor='none',facecolor='whitesmoke', alpha=0.9))
    fig.tight_layout()
    plt.show()  

    return fig, odr_out, r


# This part only runs when you execute the file directly:
if __name__ == "__main__":
    
    # data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl20min__dt1200s_dh100m_hmax6000m.nc")
    data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl20min__dt1200s_dh10m_hmax6000m.nc")
    # data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl10s__dt60s_dh10m_hmax6000m.nc")

    for daytime in ['all', 'night', 'day', 'twilight']:
        fig, fit, r = plot_rl_vs_dial(data, daytime=daytime)
        print('')
        filename = f'one-to-one_rl20min_dial__dt20min_dh10m_{daytime}_ODR_v5.png'
        folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\OneToOne"
        savefig(fig, folderpath, filename, dpi=300, show=True)
        
        # fit.print()
        # r2 = r**2 
        
    # Method      Minimizes                        Assumes
    # Linregress  vertical distance (y residuals)  x is exact, only y has error
    # ODR         perpendicular distance           both x and y have error
    
    
    