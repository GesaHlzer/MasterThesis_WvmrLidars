# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:23:33 2026

@author: alleh
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from basic_plot_funcions import savefig

data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh10m.nc")

ds, daytime = data.sel(launch=data['day_night']== 'night'), 'night'# Night Sondes
ds, daytime = data.sel(launch=data['day_night'] == 'day')  , 'day'  # Day Sondes
ds, daytime = data.copy()                                  , 'day&night'                  # All Sondes

lidar_var = 'dial_wvmr' 
lidar_var = 'rl2_wvmr'
lidar_var = 'rl2_wvmr_filtered'

hmax=12

def stat_computation(data, lidar_var):
    
    # Compute differences per launch (lidar - radiosonde)
    diff = data[lidar_var] - data['rs_wvmr']  # shape: (launch, height)
    
    # Compute number of valid (non-NaN) pairs at each height
    N = diff.count(dim='launch')  # (height,)
    # n = N.values
    
    # Mean bias δ = 1/N * Σ(diff) at each height
    md = diff.mean(dim='launch', skipna=True)  
    
    # Mean absolute difference MAD = 1/N * |diff|  
    mad = np.abs(diff).mean(dim='launch', skipna=True)
    
    # Standard deviation σ = √[1/(N-1) * Σ(diff-mean_diff)²] of bias at each height
    std_diff = diff.std(dim='launch', skipna=True, ddof=1)  # ddof=1 bc sample std
    
    # Root mean square derivation RMSD Λ = √[1/N * Σ (diff)²] at each height
    rmsd = np.sqrt( (diff**2).mean(dim='launch', skipna=True) )  # (height,)
    
    # Bundle into a Dataset 
    stats = xr.Dataset({
        'md':           md,             # mean difference
        'mad':          mad,            # absolute mean difference
        'std':          std_diff,       # std of differences
        'rsmd':         rmsd,           # RMSD
        'N':            N,              # sample count
        },
        attrs={'lidar': lidar_var})
    
    stats['height'] = data['height']
    
    return stats

def plot_stats(data, stats, daytime):
    
    height = stats['height'].values / 1000  # convert m → km if you prefer, else keep in m
    lidar = 'DA10' if lidar_var=='dial_wvmr' else 'PPL (20 min)'
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 7), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    
    # Left panel: Λ, δq, σ
    ax = axes[0]
   
    # RMSD Λ
    ax.plot(stats['rsmd'].values, height,color='black', lw=1.5, label=r'$RMSD$')
    
    # Mean difference μ
    ax.plot(stats['md'].values, height, color='darkred', 
            lw=1.2, linestyle='--', label=r'$MD = wvmr_{lidar} - wvmr_{raso}$')
    
    # Shaded envelope: μ ± ε
    ax.fill_betweenx( height,
                     (stats['md'] - stats['std']).values,
                     (stats['md'] + stats['std']).values,
                     alpha=0.4, color='indianred', label=r'$MD \pm \sigma$'
                     )
    # # Mean absolute  difference |μ|
    # ax.plot(stats['mad'].values, height,color='darkgreen', lw=1.2, linestyle='--', label=r'$MAD = |lidar_{wvmr} - raso_{wvmr}|$')
    
    # # Std of differences 
    # ax.plot(stats['std'].values, height,
    #         color='black', lw=1.0, linestyle=':', label=r'$\sigma$')
    
    ax.axvline(0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel(r'$\Delta_{wvmr}$ (g kg$^{-1}$)')
    ax.set_ylabel('height (m AGL)')
    ax.set_xlim(-2, 2)
    #ax.set_ylim(0, 10000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(r'$\Delta_{wvmr}$ ('+lidar+ ' - Raso) ' + f'{daytime}', fontsize=11)
    
    # --- Right panel: N --
    ax2 = axes[1]
    ax2_twin = ax2.twiny()
    mean_rs = data['rs_wvmr'].mean(dim='launch', skipna=True)
    l1, = ax2.plot(mean_rs, height, color='steelblue', lw=1.5, 
                   label=r'$\overline{wvmr_{raso}}$')
    ax2.set_xlabel(r'$\overline{wvmr_{raso}}$, g kg$^{-1}$', color='steelblue')
    ax2.tick_params(axis='x', labelcolor='steelblue')
    
    # N on top x-axis
    l2, = ax2_twin.plot(stats['N'].values, height, color='olivedrab', 
                        lw=1.5, label='N')
    ax2_twin.set_xlabel('N', color='olivedrab')
    ax2_twin.tick_params(axis='x', labelcolor='olivedrab', size=10)
    # ax2_twin.xaxis.set_major_locator(ticker.MultipleLocator())   # ticks at 0,5,10,15,20...
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2_twin.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # combined legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=9, loc='upper right')
    ax2.grid(alpha=0.3)
    
    # ax2.plot(stats['N'].values, height,
    #          color='yellowgreen', lw=1.5, label='N')
    # ax2.set_xlabel('N')
    # ax2.set_xlim(0, stats['N'].values.max() * 1.1)
    # ax2.legend(fontsize=8)
    # ax2.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    fig.tight_layout()    
    plt.show()
    
    return fig

def plot_stats_v2(stats, daytime, hmax):
    
    height = stats['height'].values / 1000  # convert m → km if you prefer, else keep in m
    lidar = 'DA10' if lidar_var=='dial_wvmr' else 'PPL'
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    
    
    # RMSD Λ
    l10, = ax.plot(stats['rsmd'].values, height,color='black', lw=1.5, label=r'$RMSD$')
    
    # Mean difference μ
    l11, = ax.plot(stats['md'].values, height, color='darkred', 
            lw=1.2, linestyle='--', label=r'MD')
    
    # Shaded envelope: μ ± ε
    l12 = ax.fill_betweenx( height,
                     (stats['md'] - stats['std']).values,
                     (stats['md'] + stats['std']).values,
                     alpha=0.4, color='indianred', label=r'$MD \pm \sigma$'
                     )
    #ax.set_title(f'Deviation ({lidar} - Raso): {daytime} soundings', fontsize=11)
    #     #ax.set_title(rf'$\Delta$wvmr = wvmr$_{{\mathrm{{{lidar}}}}}$ - wvmr$_{{Raso}}$ {daytime}', fontsize=11)

    ax.grid(alpha=0.3)
    ax.axvline(0, color='black', lw=0.8, linestyle=':')
    ax.set_xlabel(fr'$\Delta wvmr^{{{lidar}}}_{{{daytime}}}$ (g kg$^{{-1}}$)', fontsize=11)
    ax.set_ylabel('height (m AGL)', fontsize=11)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, hmax)
    ax.tick_params(labelsize=11, size=5)
    #
    
    
    # --------- N on top x-axis
    ax2 =  ax.twiny()
    
    l2, = ax2.plot(stats['N'].values, height, color='steelblue', 
                        lw=1.5, alpha=0.6, linestyle='-', label='N')
    ax2.set_xlabel('N', color='steelblue')
    ax2.tick_params(axis='x', labelcolor='steelblue', labelsize=11, size=5)
    
    # ---------combined legend
    lines = [l10, l11, l12, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, loc='upper left')

    fig.tight_layout()    
    plt.show()
    
    return fig

stats = stat_computation(ds, lidar_var)
fig = plot_stats_v2(stats, daytime, hmax)

filename = f'vertical_bias_{lidar_var}_to{hmax}km_{daytime}.png'
folderpath = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\vertical_bias"
savefig(fig, folderpath, filename, dpi=300, show=True)