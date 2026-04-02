"""
Created on Wed Jul 23 00:18:52 2025

@author: alleh
"""
import numpy as np
import xarray as xr
from basic_plot_funcions import haversine

# ============ Settings for the new dataset ==================================

rltime = '10s' # '10s' '20min' '1200s'      # select PPL dataset

hmax = 6000  # meter 4000 12000             # up to which height 
dh   = 10    # meter
dt   = 60    # sec

#==============================================================================

# --- Load data ---

da10_wvmr  =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
da10_abs   =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\\dial_abs.nc")
awstations =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\stationsdata.nc")
ppl10s     =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_90.0%valid.nc")
ppl20m     =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_90.0%valid.nc")


# Check time-height grids of the different datasets
#---------------------------------------------------------------------------------------------------------------------
# timesteps at end of averaging period
td  = da10_wvmr.time.values.astype('datetime64[ns]').astype(str)  # dt= 60s-65s  (20 min avg of 5s retrivals)
tdb = da10_abs.time.values.astype('datetime64[ns]').astype(str)   # dt= 60s-65s  (5 min avg)

tp  = ppl10s.time.values.astype('datetime64[ns]').astype(str)     # dt= 10s-11s  (10 sec avg)
tp2 = ppl20m.time.values.astype('datetime64[ns]').astype(str)     # dt= 20min40s (20 min avg of 10s retrevials)

ts  = awstations.time.values.astype('datetime64[ns]').astype(str) # dt= 10min    (10 min avg)

# vertically averaged using Gaussian-like weighting functions in Δz = 10m at surface/100m at 200mAGL/500m at 3000mAGL
hd  = da10_wvmr.height.values      # AGL 577  # dh =  9.6 m,   h = [57.6, 67.2, 76.8, ...  4012.8,   4022.4,  4032.]
hdb = da10_abs.height.values       # AGL 577  # dh =  4.8 m,   h = [48. , 52.8, 57.6, ... 17990.4,  17995.2, 18000.]
                                   
# vertically averaged with gliding mean in a Δh = 97.5m window
hp  = ppl10s.height.values         # AGL 574  # dh = 3.75 m,   h = [3.75, 7.5, 11.25, ... 11992.5, 11996.25, 12000.]
hp2 = ppl20m.height.values         # AGL 574  # dh = 3.75 m,   h = [3.75, 7.5, 11.25, ... 11992.5, 11996.25, 12000.]
                                   
hs  = awstations.altitude.values   # MSL 0    # [2270., 1921., 1566., 1208.,  907.,  665.,  611.,  635.,  579.]
#---------------------------------------------------------------------------------------------------------------------


dial    = da10_wvmr.copy()
dialabs = da10_abs.copy()
aws     = awstations.copy()
rl      = ppl10s.copy() if rltime == '10s' else ppl20m.copy() # if rltime=20m

def fuse_dial_raman_aws(rl, dial, dialabs, aws, hmax, dh, dt, rltime):
    
    dialabs = dialabs.sortby("time") # ensure that time increases is monoton
    
    # --- Move timestamps to the midlle of their averaging period to ensure correct alignment
    dial['time']    = dial['time']   - np.timedelta64(10, 'm')  # (20min avg)
    dialabs['time'] = dialabs['time']- np.timedelta64(150,'s')  # (5min avg)
    aws['time']     = aws['time']    - np.timedelta64(5,  'm')  # (10min avg)
    if rltime == '10s':
        rl['time']  = rl['time']     - np.timedelta64(5,  's')  # (10s retrevial)
    else: #rltime == '1200s'
        rl['time']  = rl['time']     - np.timedelta64(10, 'm')  # (20min avg)
    

    # --- Select measurement period of PPL
    start = np.datetime64('2024-08-23T00:00')
    end   = np.datetime64('2024-09-09T00:00')
    
    pl   = rl     .sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))
    da   = dial   .sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm'))) 
    dabs = dialabs.sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))     
    ws   = aws    .sel(time=slice(start-np.timedelta64(1, 'm'), end+np.timedelta64(1, 'm')))

    # --- Define new grid
    gridheight = np.arange(0, hmax+1 , dh, dtype='int64')
    gridtime = np.arange(start, end, np.timedelta64(dt, 's'), dtype='datetime64[ns]')
    
    
    # --- Calculate the distance from the stations to the lidars in km
    ws['distance_aws'] =  haversine(da.latitude.item(), da.longitude.item(), 
                                    ws['lat'].values, ws['lon'].values)
    
    # --- Adjust layout
    dabs = dabs.rename({"beta_att": "bsr"})
    da = da.rename({"water_vapor":                "wvmr",
                    "water_vapor_uncertainty":    "wvmr_unc",
                    "water_vapor_max_range":      "wvmr_max_range"     })
    
    # Mask da10 water-vapor above water_vapor_max_range   
    #da['wvmr']     = da['wvmr']    .where(da['height'] <= da['wvmr_max_range'])
    #da['wvmr_unc'] = da['wvmr_unc'].where(da['height'] <= da['wvmr_max_range'])
    
    # Unify 'height' coordinates with 577 m ASL (=da.elevation) as ground reference
    pl['height'] = pl['height'] - 3            # 574 -> 577 m ASL
    ws['height'] = ws['altitude'] - 577        # 0 -> 577 m ASL
    
    
    # Keep only relevant data 
    da = da[['wvmr', 'wvmr_unc', 'wvmr_max_range', 'longitude', 'latitude', 'surface_pressure', 'surface_temperature', 'surface_water_vapor']] 
    dabs = dabs[['bsr']].drop_vars(["latitude", "longitude"]) 
    pl   = pl[['wvmr', 'temp', 'br', 'temp_filtered', 'wvmr_filtered', 'temp_max_range', 'wvmr_max_range']]
    ws   = ws[['wvmr', 'temp',  'height', 'distance_aws', 'shortcut']]
    
    
    # make sure all times are unique and handle NaT
    vals_da,   idx_da   = np.unique(da['time'].values, return_index=True)
    da                  = da.isel(time=np.sort(idx_da))
    vals_da,   idx_da   = np.unique(da['time'].values, return_index=True)
    da                  = da.isel(time=np.sort(idx_da))
    vals_dabs, idx_dabs = np.unique(dabs['time'].values, return_index=True)
    dabs                = dabs.isel(time=np.sort(idx_dabs))
    vals_pl,   idx_pl   = np.unique(pl['time'].values, return_index=True)
    pl                  = pl.isel(time=np.sort(idx_pl))
    
    # --- Interpolate raso, dial & ppl on the height & time grid 
    da   =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
    dabs =  dabs.interp({'height': gridheight, 'time': gridtime}, method='linear')
    pl   =  pl.interp({'height': gridheight, 'time': gridtime}, method='linear')
    
    # Asemble in Dataset
    
    ds_lidars = xr.Dataset({
        'dial_wvmr'         : (('time','height'), da.wvmr.values),
        'dial_wvmr_unc'     : (('time','height'), da.wvmr_unc.values),
        'dial_wvmr_maxrange': (('time'), da.wvmr_max_range.values),
        'dial_bsr'          : (('time','height'), dabs.bsr.values),
        'dial_surface_press': (('time'), da.surface_pressure.values),
        'dial_surface_temp' : (('time'), da.surface_temperature.values),
        'dial_surface_wvmr' : (('time'), da.surface_water_vapor.values),
        'rl_wvmr'           : (('time','height'), pl.wvmr.values),
        'rl_wvmr_filtered'  : (('time','height'), pl.wvmr_filtered.values),
        'rl_wvmr_maxrange'  : (('time'), pl.wvmr_max_range.values),
        'rl_br'             : (('time','height'), pl.br.values),
        'rl_temp'           : (('time','height'), pl.temp.values),
        'rl_temp_filtered' : (('time','height'), pl.temp_filtered.values),
        'rl_temp_maxrange'  : (('time'), pl.temp_max_range.values),
        }, 
        coords={
            'height': gridheight,
            'time'  : gridtime,
        },
        attrs={
            'longnitude_lidars': da.longitude.item(),
            'latitude_lidars': da.latitude.item(),
            'elevation': 577, # m a.m.s.l.
            'dial_merging region': dial.merging_region.values[0]
        })
     
    # --- Interpolate station data on the height & time grid 
    
    # compute station grid heights
    closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
    gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
    
    # pull aws data at those times
    wvmr_ws = ws['wvmr'].interp(time=gridtime, method='linear')
    temp_ws = ws['temp'].interp(time=gridtime, method='linear')
    
    #Assemble as new Dataset
    ds_aws = xr.Dataset(
        {
             'aws_wvmr'    : (('station', 'time'), wvmr_ws.values),               # (station,)
             'aws_temp'    : (('station', 'time'), temp_ws.values),                # (station,)
             'aws_height'  : (('station'), gridheightWS),
             'aws_shortcut': (('station'), ws['shortcut'].values),
             'aws_distance': (('station'), ws['distance_aws'].values),
             },
             coords={'station': ws['station'].values,
                     'time'   : gridtime,
            }
    )
    
    # Merge with other gidded data
    data = xr.merge([ds_lidars, ds_aws])
    
    # add metadata
    data.attrs['description'] = (
        f'DA10, PPL ({rltime}) and AWs water-vapor-mixing‐ratio and temperature data'
        f'on a uniform {dh} m x {dt} s grid'
    )
    data.dial_wvmr.attrs    .update(units='g kg-1'  , long_name='DA10 wv mixing ratio')
    data.dial_wvmr_unc.attrs.update(units='g kg-1'  , long_name='DA10 wvmr uncertainty')
    # 'dial_wvmr_maxrange'
    data.dial_bsr.attrs     .update(units='m-1 sr-1', long_name='DA10 attenuated backscatter profile')
    # 'dial_surface_press'
    # 'dial_surface_temp' 
    # 'dial_surface_wvmr' 
    data.rl_temp.attrs      .update(units='°C'      , long_name='PPL temperature')
    data.rl_wvmr.attrs      .update(units='g kg-1'  , long_name='PPL wv mixing ratio')
    data.rl_br.attrs        .update(units=' '       , long_name='PPL backscatter ratio')
    # 'rl_wvmr_filtered'
    # 'rl_temp_filtered'
    # 'rl_wvmr_maxrange'
    # 'rl_temp_maxrange'
    data.aws_temp.attrs     .update(units='°C'      , long_name='AWS temperature')
    data.aws_wvmr.attrs     .update(units='g kg-1'  , long_name='AWS mixing ratio')
    data.aws_distance.attrs .update(units='g kg-1'  , long_name='Distance of the stations to the lidars')

    return data


dts = fuse_dial_raman_aws(rl, dial, dialabs, aws, hmax, dh, dt, rltime)
outpath_ts = fr'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\2D_mereged_data_rl{rltime}__dt{dt}s_dh{dh}m_hmax{hmax}m.nc'

dts.to_netcdf(outpath_ts)
print(f'saved joint dataset as {outpath_ts}')
    
dstest = xr.open_dataset(outpath_ts)
print(dstest)    





    
 



