"""
Created on Wed Jul 23 00:18:52 2025

VERTICAL PROFILES - Radiosonde with DA10, PPL & AWS       

@author: alleh
"""
import numpy as np
import xarray as xr
from basic_plot_funcions import haversine


# ============ Settings for the new dataset ==================================
dh   = 10    # meter
#==============================================================================

# --- Load data ---
da10_wvmr  =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\dial_wvmr.nc")
da10_abs   =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\\dial_abs.nc")
awstations =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\stationsdata.nc")
ppl10s     =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_10s_filtered_50.0%valid.nc")
ppl20m     =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\PPL\ppl_1200s_filtered_75.0%valid.nc")
rsondes    =  xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\radiosondes.nc")


# Check time-height grids of the different datasets

#---------------------------------------------------------------------------------------------------------------------
# timesteps at end of averaging period
td  = da10_wvmr.time.values.astype('datetime64[ns]').astype(str)  # dt= 60s-65s  (20 min avg of 5s retrivals)
tdb = da10_abs.time.values.astype('datetime64[ns]').astype(str)   # dt= 60s-65s  (5 min avg)

tp  = ppl10s.time.values.astype('datetime64[ns]').astype(str)     # dt= 10s-11s  (10 sec avg)
tp2 = ppl20m.time.values.astype('datetime64[ns]').astype(str)     # dt= 20min40s (20 min avg of 10s retrevials)

ts  = awstations.time.values.astype('datetime64[ns]').astype(str) # dt= 10min    (10 min avg)
tr = rsondes.time.values.astype('datetime64[ns]').astype(str)          # dt = 1 s     (instantaneous measurement)

# vertically averaged using Gaussian-like weighting functions in Δz = 10m at surface/100m at 200mAGL/500m at 3000mAGL
hd  = da10_wvmr.height.values      # AGL 577  # dh =  9.6 m,   h = [57.6, 67.2, 76.8, ...  4012.8,   4022.4,  4032.]
hdb = da10_abs.height.values       # AGL 577  # dh =  4.8 m,   h = [48. , 52.8, 57.6, ... 17990.4,  17995.2, 18000.]
                                   
# vertically averaged with gliding mean in a Δh = 97.5m window
hp  = ppl10s.height.values         # AGL 574  # dh = 3.75 m,   h = [3.75, 7.5, 11.25, ... 11992.5, 11996.25, 12000.]
hp2 = ppl20m.height.values         # AGL 574  # dh = 3.75 m,   h = [3.75, 7.5, 11.25, ... 11992.5, 11996.25, 12000.]
                                   
hs  = awstations.altitude.values   # MSL 0    # [2270., 1921., 1566., 1208.,  907.,  665.,  611.,  635.,  579.]

hr = rsondes.altitude.values          # dz = about 4 m         z =[586., 591., 596., ...,  nan,  nan,  nan],
                                      #                           [586., 588., 591., ...,  nan,  nan,  nan],
                                      #                           [591., 595., 600., ...,  nan,  nan,  nan],
#------------------------------------------------------------------------------------------------------------------

def adjust_data(da10_wvmr, da10_abs, awstations, ppl10s, ppl20m, rsondes):
    
    dial    = da10_wvmr.copy()
    dialabs = da10_abs.copy()
    aws     = awstations.copy()
    rl      = ppl10s.copy()
    rl2     = ppl20m.copy()
    rs      = rsondes.copy()
        
    # --- Move timestamps to the midlle of their averaging period to ensure correct alignment
    
    dial['time']    = dial['time']   - np.timedelta64(10, 'm')  # (20min avg)
    dialabs['time'] = dialabs['time']- np.timedelta64(150,'s')  # (5min avg)
    aws['time']     = aws['time']    - np.timedelta64(5,  'm')  # (10min avg)
    rl['time']      = rl['time']     - np.timedelta64(5,  's')  # (10s retrevial)
    rl2['time']     = rl2['time']    - np.timedelta64(10, 'm')  # (20min avg)
    # rs measures on time stamp in-situ
    
    
    # --- Align 'height' coordinates with 577 m ASL as ground reference
    
    rl['height']  = rl['height']  - 3           # 574 -> 577 m ASL
    rl2['height'] = rl2['height'] - 3           # 574 -> 577 m ASL
    aws['height'] = aws['altitude'] - 577       # 0 -> 577 m ASL
    rs['height']  = rs['altitude'] - 577        # 0 -> 577 m ASL
    
    # Crop to relavent measurement values 
    #rs = rs.where(rs['height']  <= 12000.)
    #dialabs = dialabs.where(rs['height']  <= 12000.)
    
    
    # --- Discard first launch (not in data)
    rs = rs.sel(launch=rs.launch[rs.launch != 0])

    # --- Rename variables to chosen convention
    dial = dial.rename({"water_vapor"            :"wvmr",
                        "water_vapor_uncertainty":"wvmr_unc",
                        "water_vapor_max_range"  :"wvmr_max_range"})
    
    dialabs = dialabs.rename({"beta_att": "bsr"})
    
    rs = rs.rename({"mr": "wvmr",
                    "t":  "temp"})
    
    # --- Mask da10 and ppl water-vapor above water_vapor_max_range
    
    dial['wvmr']         = dial['wvmr']        .where(dial['height'] <= dial['wvmr_max_range'])
    dial['wvmr_unc']     = dial['wvmr_unc']    .where(dial['height'] <= dial['wvmr_max_range'])
    
    rl['wvmr']           = rl['wvmr']          .where(rl['height']  <= rl['wvmr_max_range'])
    rl['wvmr_filtered']  = rl['wvmr_filtered'] .where(rl['height']  <= rl['wvmr_max_range'])
    rl['temp']           = rl['temp']          .where(rl['height']  <= rl['temp_max_range'])
    rl['temp_filtered']  = rl['temp_filtered'] .where(rl['height']  <= rl['temp_max_range'])
    
    rl2['wvmr']          = rl2['wvmr']         .where(rl2['height'] <= rl2['wvmr_max_range'])
    rl2['wvmr_filtered'] = rl2['wvmr_filtered'].where(rl2['height'] <= rl2['wvmr_max_range'])
    rl2['temp']          = rl2['temp']         .where(rl2['height'] <= rl2['temp_max_range'])
    rl2['temp_filtered'] = rl2['temp_filtered'].where(rl2['height'] <= rl2['temp_max_range'])
    
    # --- Calculate the distance from the in-situ measurements to the lidars in km
    aws['distance'] =  haversine(dial.latitude.item(), dial.longitude.item(), 
                                    aws['lat'].values, aws['lon'].values)
    
    distance_rs   =  haversine(dial.latitude.item(), dial.longitude.item(), 
                                    rs['lat'].values, rs['lon'].values)
    rs['distance'] = xr.DataArray(distance_rs, dims=['launch', 'index'])
    

    # Keep only relevant data 
    dial    = dial[['wvmr', 'wvmr_unc', 'wvmr_max_range', 'longitude', 'latitude', 'surface_pressure', 'surface_temperature', 'surface_water_vapor']] 
    dialabs = dialabs[['bsr']].drop_vars(["latitude", "longitude"]) 
    rl      = rl[['wvmr', 'temp', 'br', 'temp_filtered', 'wvmr_filtered', 'temp_max_range', 'wvmr_max_range']]
    rl2     = rl2[['wvmr', 'temp', 'br', 'temp_filtered', 'wvmr_filtered', 'temp_max_range', 'wvmr_max_range']]
    aws     = aws[['wvmr', 'temp',  'height', 'distance', 'shortcut']]
    rs      = rs[['time', 'height', 'wvmr', 'temp', 'distance', 'p', 'lon', 'lat', 'launch','date', 'day_night']]

    return dial, dialabs, rl, rl2, aws, rs

def raman_dial_aws_raso(i, rs, dial, dialabs, rl, rl2, aws, dh): 
    
    # --- select the corresponding lidar timesteps
    
    # derive key times from radiosonde
    time_rs0km  = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
    time_rs2km  = rs.time.where(rs.height > 2000,  drop=True).values[0]
    time_rs4km  = rs.time.where(rs.height > 4000,  drop=True).values[0]
    time_rs12km = rs.time.where(rs.height > 12000, drop=True).values[0]
    
    mask_da   = ((dial.time >= (time_rs0km  - np.timedelta64(1,  'm'))) & 
                (dial.time <= (time_rs4km  + np.timedelta64(1,  'm'))))
    mask_dabs = ((dialabs.time >= (time_rs0km  - np.timedelta64(1,  'm'))) & 
                (dialabs.time <= (time_rs12km  + np.timedelta64(1,  'm'))))
    mask_pl   = ((rl.time   >= (time_rs0km  - np.timedelta64(10, 's'))) & 
                (rl.time   <= (time_rs12km + np.timedelta64(10, 's'))))
    mask_pl2  = ((rl2.time  >= (time_rs0km  - np.timedelta64(20, 'm'))) & 
                (rl2.time  <= (time_rs12km + np.timedelta64(20, 'm'))))
    mask_ws   = ((aws.time  >= (time_rs0km  - np.timedelta64(10, 'm'))) & 
                (aws.time  <= (time_rs2km  + np.timedelta64(10, 'm')))) 
    
    # --- Extract the matching time windows in each remote-sensing instrument
    #     as new variable
    
    da   = dial.sel(time=mask_da) 
    dabs = dialabs.sel(time=mask_dabs)
    pl   = rl.sel(time=mask_pl)
    pl2  = rl2.sel(time=mask_pl2)
    ws   = aws.sel(time=mask_ws)
    
            
    # --- define a fixed height & time grid:
    
    gridheight = np.arange(0, 12001, dh, dtype='int64')
               
    # Interpolate raso on new height grid
    
    rs["time"] = rs["time"].astype('int64') # Cast time to int64 (Nonoseconds since 1970-01-01) 
    rs = rs.swap_dims({"index": "height"})
    rs = rs.interp({'height': gridheight}, method='linear')
    rs["time"] = rs["time"].astype('datetime64[ns]') # Cast back
    
    # Define time grid
    gridtime = rs.time.values.astype('datetime64[ns]')
    
    # make sure all times are unique and handle NaT
    vals_da, idx_da = np.unique(da['time'].values, return_index=True)
    da = da.isel(time=np.sort(idx_da))
    
    vals_dabs, idx_dabs = np.unique(dabs['time'].values, return_index=True)
    dabs = dabs.isel(time=np.sort(idx_dabs))
    
    vals_pl, idx_pl = np.unique(pl['time'].values, return_index=True)
    pl = pl.isel(time=np.sort(idx_pl))
    
    vals_pl2, idx_pl2 = np.unique(pl2['time'].values, return_index=True)
    pl2 = pl2.isel(time=np.sort(idx_pl2))

    
    # --- Interpolate raso, dial & ppl on the height & time grid 
    
    # rs interp     # 5334->1200,dh ~4  ->10
    # da interp     # 415 ->1200,dh 9.6 ->10
    # dabs interp   #            dh 4.8 ->10
    # pl interp     # 3200->1200,dh 3.75->10
    # pl20 interp   # 3200->1200,dh 3.75->1
                    # kwargs={fill_value:'extrapolate'}
    
    da   = da.interp({'height': gridheight, 'time': gridtime}, method='linear')
    dabs = dabs.interp({'height': gridheight, 'time': gridtime}, method='linear')
    pl   = pl.interp({'height': gridheight, 'time': gridtime}, method='linear')
    pl2  = pl2.interp({'height': gridheight, 'time': gridtime}, method='linear')#,
    
    # Extract diagonal 
    da_mr     = da['wvmr'].values.diagonal()  
    da_mr_unc = da['wvmr_unc'].values.diagonal() 
    da_abs    = dabs['bsr'].values.diagonal() 
    da_sfc_p  = da['surface_pressure'].values
    da_sfc_t  = da['surface_pressure'].values
    da_sfc_mr = da['surface_pressure'].values
    
    ppl_t     = pl['temp'].values.diagonal()
    ppl_br    = pl['br'].values.diagonal()
    ppl_mr    = pl['wvmr'].values.diagonal() 
    ppl_mr_f  = pl['wvmr_filtered'].values.diagonal() 
    ppl_t_f   = pl['temp_filtered'].values.diagonal() 
    
    ppl2_t    = pl2['temp'].values.diagonal() 
    ppl2_mr   = pl2['wvmr'].values.diagonal() 
    ppl2_br   = pl2['br'].values.diagonal()
    ppl2_mr_f = pl2['wvmr_filtered'].values.diagonal() 
    ppl2_t_f  = pl2['temp_filtered'].values.diagonal()
    
    # Asemble in Dataset
    
    ds_joint = xr.Dataset({
        'time'             : (('height'), gridtime),
        'rs_wvmr'          : (('height'), rs['wvmr'].values),
        'rs_temp'          : (('height'), rs['temp'].values),
        'rs_distance'      : (('height'), rs['distance'].values),
        
        'dial_wvmr'        : (('height'), da_mr),
        'dial_wvmr_unc'    : (('height'), da_mr_unc),
        'dial_bsr'         : (('height'), da_abs),
        'dial_sfc_wvmr'    : (('height'), da_sfc_mr),
        'dial_sfc_p'       : (('height'), da_sfc_p),
        'dial_sfc_temp'    : (('height'), da_sfc_t),
        
        'rl_temp'          : (('height'), ppl_t),
        'rl_wvmr'          : (('height'), ppl_mr),
        'rl_br'            : (('height'), ppl_br),
        'rl_temp_filtered' : (('height'), ppl_t_f),
        'rl_wvmr_filtered' : (('height'), ppl_mr_f),
        
        'rl2_temp'         : (('height'), ppl2_t),
        'rl2_wvmr'         : (('height'), ppl2_mr),
        'rl2_br'           : (('height'), ppl2_br),
        'rl2_temp_filtered': (('height'), ppl2_t_f),
        'rl2_wvmr_filtered': (('height'), ppl2_mr_f),
        }, 
        coords={
            'height'   : gridheight,
            'launch'   : rs.launch.item(),
            'date'     : rs.date.values,
            'day_night': rs.day_night.item(),
            },
        attrs={
            'longnitude_lidars': da.longitude.item(),
            'latitude_lidars': da.latitude.item(),
            'elevation': 577, # m a.m.s.l.
            }
        )
     
    # --- Interpolate station data on the time grid and closest height on the grid

    # compute station grid heights
    closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
    gridheightWS = gridheight[closest_idx]  
    
    # Build a station‐indexed xarray of target heights
    gridheightWS = xr.DataArray(gridheightWS,
                        dims='station',
                        coords={'station': ws['station'].values},
                        name='height'
                        )
    # Interpolate radiosonde times onto station elevations 
    # → get time when sonde passes each station
    time_rs_at_aws = (rs['time']
                      .astype('int64')
                      .interp(height=gridheightWS, method='nearest')
                      .astype('datetime64[ns]')
                      )
    time_rs_at_aws.loc[{'station': 'Tawes'}] = time_rs0km #otherwise NaT

    # pull aws data at those times
    mr_ws = ws['wvmr'].interp(time=time_rs_at_aws, method='linear')
    t_ws  = ws['temp'].interp(time=time_rs_at_aws, method='linear')
    
    #Assemble as new Dataset
    ds_aws = xr.Dataset(
        {
             'aws_time'    : (('station'), time_rs_at_aws.values),      # (station,)
             'aws_wvmr'    : (('station'), mr_ws.values),               # (station,)
             'aws_temp'    : (('station'), t_ws.values),                # (station,)
             'aws_height'  : (('station'), gridheightWS.values),
             'aws_shortcut': (('station'), ws['shortcut'].values),
             'aws_distance': (('station'), ws['distance'].values),
             },
             coords={'station': ws['station'].values
            }
    )
    
    # Merge with other gidded data
    ds_new = xr.merge([ds_joint, ds_aws])
    
    return ds_new

def dial_aws_raso(i, rs, dial, dialabs, aws, dh): 
      
    # derive key times from radiosonde
    time_rs0km = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
    time_rs2km = rs.time.where(rs.height > 2000, drop=True).values[0]
    time_rs4km = rs.time.where(rs.height > 4000, drop=True).values[0]
    time_rs12km = rs.time.where(rs.height > 12000, drop=True).values[0]
    
    mask_da   = ((dial.time >= (time_rs0km - np.timedelta64(1,  'm'))) & 
                 (dial.time <= (time_rs4km + np.timedelta64(1,  'm'))))
    mask_dabs = ((dialabs.time >= (time_rs0km - np.timedelta64(1,  'm'))) & 
                 (dialabs.time <= (time_rs12km + np.timedelta64(1,  'm'))))
    mask_ws   = ((aws.time  >= (time_rs0km - np.timedelta64(10, 'm'))) & 
                 (aws.time  <= (time_rs2km + np.timedelta64(10, 'm')))) 
    
    # extract the matching time windows in each remote-sensing instrument
    da   = dial.sel(time=mask_da)     
    dabs = dialabs.sel(time=mask_dabs)        
    ws   = aws.sel(time=mask_ws)
    

    # --- define a fixed height & time grid:
    gridheight = np.arange(0, 12001, dh) #, dtype='int64')
                               
    # --- Interpolate rsb on new  height grid
    
    # rs interp     # 5334->1200,dh ~4  ->10
    rs["time"] = rs["time"].astype('int64') # Cast time to int64 (Nonoseconds since 1970-01-01) 
    rs = rs.swap_dims({"index": "height"})
    rs = rs.interp({'height': gridheight}, method='linear')
    rs["time"] = rs["time"].astype('datetime64[ns]') # Cast back
    
    # define time grid
    gridtime = rs.time.values.astype('datetime64[ns]')
    
    # --- Interpolate raso, dial & ppl on the height & time grid 
    
    # make sure all times are unique and handle NaT
    vals, idx = np.unique(da['time'].values, return_index=True)
    da = da.isel(time=np.sort(idx))
    
    vals_dabs, idx_dabs = np.unique(dabs['time'].values, return_index=True)
    dabs = dabs.isel(time=np.sort(idx_dabs))
    
    
    # da interp     # 415 ->1200,dh 9.6 ->10     
    da  =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
    dabs = dabs.interp({'height': gridheight, 'time': gridtime}, method='linear')
    
    # Extract diagonal 
    da_mr      = da['wvmr'].values.diagonal() #.where(~nat_mask).diagonal()
    da_mr_unc = da['wvmr_unc'].values.diagonal() #.where(~nat_mask).diagonal()
    da_abs    = dabs['bsr'].values.diagonal() 
    da_sfc_p  = da['surface_pressure'].values
    da_sfc_t  = da['surface_pressure'].values
    da_sfc_mr = da['surface_pressure'].values
    

    # Asemble in Dataset
    ds_joint = xr.Dataset({
        'time'             : (('height'), gridtime),
        'rs_wvmr'          : (('height'), rs['wvmr'].values),
        'rs_temp'          : (('height'), rs['temp'].values),
        'rs_distance'      : (('height'), rs['distance'].values),
        
        'dial_wvmr'        : (('height'), da_mr),
        'dial_wvmr_unc'    : (('height'), da_mr_unc),
        'dial_bsr'         : (('height'), da_abs),
        'dial_sfc_wvmr'    : (('height'), da_sfc_mr),
        'dial_sfc_p'       : (('height'), da_sfc_p),
        'dial_sfc_temp'    : (('height'), da_sfc_t),
        
        }, 
        coords={
            'height'   : gridheight,
            'launch'   : rs.launch.item(),
            'date'     : rs.date.values,
            'day_night': rs.day_night.item(),
            },
        attrs={
            'longnitude_lidars': da.longitude.item(),
            'latitude_lidars': da.latitude.item(),
            'elevation': 577, # m a.m.s.l.
            }
        )
    
                    
    # --- Interpolate station data on the height & time grid 
    
    # compute station grid heights
    closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
    gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
    
    # Build a station‐indexed xarray of target heights
    gridheightWS = xr.DataArray(gridheightWS,
                        dims='station',
                        coords={'station': ws['station'].values},
                        name='height'
                        )
    # Interpolate radiosonde times onto station elevations 
    # → get time when sonde passes each station
    time_rs_at_aws = (rs['time']
                      .astype('int64')
                      .interp(height=gridheightWS, method='nearest')
                      .astype('datetime64[ns]')
                      )
    time_rs_at_aws.loc[{'station': 'Tawes'}] = time_rs0km #otherwise NaT
    
    # pull aws data at those times
    mr_ws = ws['wvmr'].interp(time=time_rs_at_aws, method='linear')
    t_ws  = ws['temp'].interp(time=time_rs_at_aws, method='linear')
    
    #Assemble as new Dataset
    ds_aws = xr.Dataset(
        {
             'aws_time'    : (('station'), time_rs_at_aws.values),      # (station,)
             'aws_wvmr'    : (('station'), mr_ws.values),               # (station,)
             'aws_temp'    : (('station'), t_ws.values),                # (station,)
             'aws_height'  : (('station'), gridheightWS.values),
             'aws_shortcut': (('station'), ws['shortcut'].values),
             'aws_distance': (('station'), ws['distance'].values),
             },
             coords={'station': ws['station'].values
            }
    )
    
    # Merge with other gidded data
    ds_new = xr.merge([ds_joint, ds_aws])
    
    # # add metadata
    # data.attrs['description'] = (
    #     'joint radiosonde, DA10 & PPL & weather station water-vapor-mixing‐ratio and temperature '
    #     'on a uniform 4 m grid, time‐synced to radiosonde'
    #     )
    # data.wvmr_rs.attrs      .update(units='g/kg', long_name='RS mixing ratio')
    # data.temp_rs.attrs      .update(units='°C',   long_name='RS temperature')
    # data.wvmr_dial.attrs    .update(units='g/kg', long_name='DIAL mixing ratio')
    # data.wvmr_unc_dial.attrs.update(units='g/kg', long_name='DIAL mr uncertainty')
    # data.distance_rs.attrs  .update(units='km',   long_name='Distance of radiosonde to the lidars')
    # data.wvmr_aws.attrs     .update(units='g/kg', long_name='AWS mixing ratio')
    # data.temp_aws.attrs     .update(units='°C',   long_name='AWS temperature')
    # data.distance_aws.attrs .update(units='km',   long_name='Distance of weather stations to the lidars')
    
    return ds_new

def make_dataset(da10_wvmr, da10_abs, awstations, ppl10s, ppl20m, rsondes, dh):
        
    # --- Prepare data
    dial, dialabs, rl, rl2, aws, raso =  adjust_data(da10_wvmr, da10_abs, awstations, ppl10s, ppl20m, rsondes)
    processed_data = []
    
    # --- Combine data for each launch and add to "processed_data" list
    
    for i in raso.launch.values:
        #i = 61  # choose one radiosonde launch for testing (example i = 73)
        try:
            # Select launch and discard all NaNs from the combined dataset
            rs = raso.sel(launch=i)
            rs = rs.where(rs.time.notnull(), drop=True)
            
            # Check if PPL measurement available and run respective function
            print("Launch label:", i, 
                  "| Date:", rs.date.values.astype('datetime64[D]'),
                  "| When:", rs.day_night.values, '\n') 
            launchtime  = rs.time.values.min().astype('datetime64[m]')
            
            if launchtime in rl.time.values.astype('datetime64[m]'):
                print(f'Raman lidar available for {rs.date.values.astype("datetime64[D]")} {rs.day_night.values}')
                ds_i = raman_dial_aws_raso(i, rs, dial, dialabs, rl, rl2, aws, dh)  
            else:
                print('Raman lidar not in time')
                ds_i = dial_aws_raso(i, rs, dial, dialabs, aws, dh)
            
            # Append to list
            processed_data.append(ds_i)           
    
        except Exception as e:
            print(f"Error with launch nr {i}: {e}")
    
    # --- Combine list to one dataset andadd metadata
    data = xr.concat(processed_data, dim='launch')
     
    data.attrs['description'] = (f'Combined radiosonde, DA10 & PPL & weather station data on a uniform  {dh} m grid, time‐synced to radiosonde')
    
    data.rs_wvmr.attrs      .update(units='g kg-1',   long_name='Radiosonde water vapor mixing ratio')
    data.rs_temp.attrs      .update(units='°C',       long_name='Radiosonde temperature')
    data.rs_distance.attrs  .update(units='km',       long_name='Distance of radiosonde to the lidars')
    
    data.dial_wvmr.attrs    .update(units='g kg-1',   long_name='DA10 water vapor mixing ratio')
    data.dial_wvmr_unc.attrs.update(units='g kg-1',   long_name='DA10 wvmr uncertainty')
    data.dial_bsr.attrs     .update(units='m-1 sr-1', long_name='DA10 attenuated backscatter profile')
    # 'dial_surface_press'
    # 'dial_surface_temp' 
    # 'dial_surface_wvmr'
     
    data.rl_temp.attrs      .update(units='°C',       long_name='PPL (10s) temperature')
    data.rl_wvmr.attrs      .update(units='g kg-1',   long_name='PPL (10s) water vapor mixing ratio')
    data.rl_br.attrs        .update(units=' ',        long_name='PPL (10s) backscatter ratio')
    # 'rl_wvmr_filtered'
    # 'rl_temp_filtered'
    # 'rl_wvmr_maxrange'
    # 'rl_temp_maxrange'
    
    data.rl2_temp.attrs      .update(units='°C',       long_name='PPL (20min) temperature')
    data.rl2_wvmr.attrs      .update(units='g kg-1',   long_name='PPL (20min) water vapor mixing ratio')
    data.rl2_br.attrs        .update(units=' ',        long_name='PPL (20min) backscatter ratio')
    # 'rl2_wvmr_filtered'
    # 'rl2_temp_filtered'

    data.aws_temp.attrs     .update(units='°C'      , long_name='AWS temperature')
    data.aws_wvmr.attrs     .update(units='g kg-1'  , long_name='AWS mixing ratio')
    data.aws_distance.attrs .update(units='g kg-1'  , long_name='Distance of the stations to the lidars')
    data.aws_distance.attrs .update(units='km',   long_name='Distance of weather stations to the lidars')

    return data
    

data = make_dataset(da10_wvmr, da10_abs, awstations, ppl10s, ppl20m, rsondes, dh)

outpath = fr'C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\1D_vertical_profiles__dh{dh}m.nc'
data.to_netcdf(outpath)
print(f'saved joint dataset to {outpath}')
dstest = xr.open_dataset(outpath)
print(dstest)


#%%

# def fuse_raman_dial_aws_with_raso(dates, rsondes, da10, awstations, ppl, ppl2): 
    
#     dial = da10.copy()
#     aws  = awstations.copy()
#     rl2  = ppl2.copy()
#     rl   = ppl.copy()
#     raso = rsondes.copy()
    
#     raso = raso.where(raso['date'].isin(dates), drop=True)
    
#     # Mask da10 water-vapor above water_vapor_max_range
#     dial = dial.rename({"water_vapor"            :"mr",
#                         "water_vapor_uncertainty":"mr_unc",
#                         "water_vapor_max_range"  :"mr_maxrange"})
   
#     dial['mr']     = dial['mr']    .where(dial['height'] <= dial['mr_maxrange'])
#     dial['mr_unc'] = dial['mr_unc'].where(dial['height'] <= dial['mr_maxrange'])
    
#     processed_data = []
    
#     for i in raso.launch.values:
        
#         #i = 61  # choose one radiosonde launch for testing (example i = 73)
#         try: 
#             # --- select launch and discard all NaNs from the combined dataset
            
#             rs = raso.sel(launch=i)
#             rs = rs.where(rs.time.notnull(), drop=True)
#             print("Launch label:", i, 
#                   "Date:", rs.date.values.astype('datetime64[D]'), 
#                   "When:", rs.day_night.values, '\n')
            
#             # --- select the corresponding lidar timesteps
            
#             # derive key times from radiosonde
#             time_rs0km  = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
#             time_rs2km  = rs.time.where(rs.altitude > 2300,  drop=True).values[0]
#             time_rs4km  = rs.time.where(rs.altitude > 4000,  drop=True).values[0]
#             time_rs12km = rs.time.where(rs.altitude > 12000, drop=True).values[0]
            
#             mask_da  = ((dial.time >= (time_rs0km  - np.timedelta64(1,  'm'))) & 
#                         (dial.time <= (time_rs4km  + np.timedelta64(1,  'm'))))
#             mask_rl  = ((rl.time   >= (time_rs0km  - np.timedelta64(10, 's'))) & 
#                         (rl.time   <= (time_rs12km + np.timedelta64(10, 's'))))
#             mask_rl2 = ((rl2.time  >= (time_rs0km  - np.timedelta64(20, 'm'))) & 
#                         (rl2.time  <= (time_rs12km + np.timedelta64(20, 'm'))))
#             mask_aws = ((aws.time  >= (time_rs0km  - np.timedelta64(10, 'm'))) & 
#                         (aws.time  <= (time_rs2km  + np.timedelta64(10, 'm')))) 
            
#             # extract the matching time windows in each remote-sensing instrument
#             da  = dial.sel(time=mask_da)            
#             pl  = rl  .sel(time=mask_rl)
#             pl2 = rl2 .sel(time=mask_rl2)
#             ws  = aws .sel(time=mask_aws)
            
#             # --- Adjust layout
            
#             # Unify 'height' coordinates with 577 m ASL as ground reference
#             ws['height']  = ws['altitude'] - 577         # 0 -> 571 m ASL
#             rs['height']  = rs['altitude'] - 577         # 0 -> 571 m ASL
#             pl['height']  = pl['height']   - 3           # 574 -> 571 m ASL
#             pl2['height'] = pl2['height']  - 3           # 574 -> 571 m ASL
#             #pl2 = pl2.rename({'range': 'height'})
            
#             # Drop unnecassary data 
#             rs  =  rs[['time', 'height', 'mr', 't', 'p', 'lon', 'lat', 'launch', 'date', 'day_night']]
#             da  =  da[['mr', 'mr_unc', 'longitude', 'latitude'] ] 
#             pl  =  pl[['wvmr', 'temp', 'br']]
#             pl2 = pl2[['wvmr', 'temp', 'br']]
            
            
#             # --- define a fixed height & time grid:
            
#             gridheight = np.arange(0, 12001, 10, dtype='int64')
                       
#             # Interpolate rsb on new  height grid
#             rs["time"] = rs["time"].astype('int64') # Cast time to int64 (Nonoseconds since 1970-01-01) 
#             rs = rs.swap_dims({"index": "height"})
#             rs = rs.interp({'height': gridheight}, method='linear')
#             rs["time"] = rs["time"].astype('datetime64[ns]') # Cast back
            
#             # define time grid
#             gridtime = rs.time.values.astype('datetime64[ns]')
            
#             # make sure all times are unique and handle NaT
#             vals_da, idx_da = np.unique(da['time'].values, return_index=True)
#             da = da.isel(time=np.sort(idx_da))
            
#             vals_pl, idx_pl = np.unique(pl['time'].values, return_index=True)
#             pl = pl.isel(time=np.sort(idx_pl))
            
#             vals_pl2, idx_pl2 = np.unique(pl2['time'].values, return_index=True)
#             pl2 = pl2.isel(time=np.sort(idx_pl2))

            
#             # --- Interpolate raso, dial & ppl on the height & time grid 
            
#             # rs interp     # 5334->1200,dh ~4  ->10
#             # da interp     # 415 ->1200,dh 9.6 ->10
#             # pl interp     # 3200->1200,dh 3.75->10
#             # pl20 interp   # 3200->1200,dh 3.75->1
#                             # kwargs={fill_value:'extrapolate'}
            
#             da  =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
#             pl  =  pl.interp({'height': gridheight, 'time': gridtime}, method='linear')
#             pl2 = pl2.interp({'height': gridheight, 'time': gridtime}, method='linear')#,
            
#             # Extract diagonal 
#             da_mr     = da['mr']    .values.diagonal()  
#             da_mr_unc = da['mr_unc'].values.diagonal() 
#             ppl_t     = pl['temp']  .values.diagonal()
#             ppl_br    = pl['br']    .values.diagonal()
#             ppl_mr    = pl['wvmr']  .values.diagonal() 
#             ppl2_t    = pl2['temp'] .values.diagonal() 
#             ppl2_mr   = pl2['wvmr'] .values.diagonal() 
#             ppl2_br   = pl2['br']   .values.diagonal()
            
#             # Calculate the distance from the radiosonde to the lidars in km
#             distance_rs =  haversine(da.latitude.item(), da.longitude.item(), 
#                                      rs['lat'].values, rs['lon'].values)
            
#             # Asemble in Dataset
            
#             ds_joint = xr.Dataset({
#                 'time'           : (('height'), gridtime),
#                 'wvmr_rs'        : (('height'), rs['mr'].values),
#                 'temp_rs'        : (('height'), rs['t'].values),
#                 'wvmr_dial'      : (('height'), da_mr),
#                 'wvmr_unc_dial'  : (('height'), da_mr_unc),
#                 'temp_rl'        : (('height'), ppl_t),
#                 'wvmr_rl'        : (('height'), ppl_mr),
#                 'br_rl'          : (('height'), ppl_br),
#                 'temp_rl2'       : (('height'), ppl2_t),
#                 'mwvr_rl2'       : (('height'), ppl2_mr),
#                 'br_rl2'         : (('height'), ppl2_br),
#                 'distance_rs'    : (('height'), distance_rs),
                
#             }, coords={
#                 'height'   : gridheight,
#                 'launch'   : rs.launch.item(),
#                 'date'     : rs.date.values,
#                 'day_night': rs.day_night.item()
#             })
             
#             # --- Interpolate station data on the height & time grid 

#             # compute station grid heights
#             closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
#             gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
            
#             # Build a station‐indexed xarray of target heights
#             gridheightWS = xr.DataArray(gridheightWS,
#                                 dims='station',
#                                 coords={'station': ws['station'].values},
#                                 name='height'
#                                 )
#             # Interpolate radiosonde times onto station elevations 
#             # → get time when sonde passes each station
#             time_rs_at_aws = (rs['time']
#                               .astype('int64')
#                               .interp(height=gridheightWS, method='nearest')
#                               .astype('datetime64[ns]')
#                               )
#             time_rs_at_aws.loc[{'station': 'Tawes'}] = time_rs0km #otherwise NaT

#             # pull aws data at those times
#             mr_ws = ws['mr'] .interp(time=time_rs_at_aws, method='linear')
#             t_ws  = ws['t']  .interp(time=time_rs_at_aws, method='linear')


#             # Calculate the distance from the stations to the lidars in km
#             distance_aws =  haversine(da.latitude.item(), da.longitude.item(), 
#                                       ws['lat'].values, ws['lon'].values)
            
#             #Assemble as new Dataset
#             ds_aws = xr.Dataset(
#                 {
#                      'time_aws'    : (('station'), time_rs_at_aws.values),      # (station,)
#                      'wvmr_aws'    : (('station'), mr_ws.values),               # (station,)
#                      'temp_aws'    : (('station'), t_ws.values),                # (station,)
#                      'height_aws'  : (('station'), gridheightWS.values),
#                      'shortcut_aws': (('station'), ws['shortcut'].values),
#                      'distance_aws': (('station'), distance_aws),
#                      },
#                      coords={'station': ws['station'].values
#                     }
#             )
            
#             # Merge with other gidded data
#             ds_new = xr.merge([ds_joint, ds_aws])
            
#             #Append to list of all launches
#             processed_data.append(ds_new)
            
#         # Handle exceptions as not existing data by skipping them
#         except Exception as e:
#             print(f"Error with launch nr {rs.launch.item()}: {e}")
#             continue
    
#     data = xr.concat(processed_data, dim='launch')
    
#     # add metadata
#     data.attrs['description'] = (
#         'joint radiosonde, DA10 & PPL & weather station water-vapor-mixing‐ratio and temperature '
#         'on a uniform 4 m grid, time‐synced to radiosonde'
#         )
#     data.wvmr_rs.attrs      .update(units='g/kg', long_name='Radiosonde (RS) water vapor mixing ratio')
#     data.temp_rs.attrs      .update(units='°C',   long_name='Radiosonde (RS) temperature')
#     data.wvmr_dial.attrs    .update(units='g/kg', long_name='DIAL (DA10) water vapor mixing ratio')
#     data.wvmr_unc_dial.attrs.update(units='g/kg', long_name='DIAL (DA10) wvmr uncertainty')
#     data.br_rl.attrs        .update(              long_name='RamanLidar (PPL) 10s backscatter ratio')
#     data.temp_rl.attrs      .update(units='°C',   long_name='RamanLidar (PPL) 10s temperature')
#     data.wvmr_rl.attrs      .update(units='g/kg', long_name='RamanLidar (PPL) 10s water vapor mixing ratio')
#     data.br_rl2.attrs       .update(              long_name='RamanLidar (PPL) 1200s backscatter ratio')
#     data.temp_rl2.attrs     .update(units='°C',   long_name='RamanLidar (PPL) 1200s temperature')
#     data.wvmr_rl2.attrs     .update(units='g/kg', long_name='RamanLidar (PPL) 1200s water vapor mixing ratio')
#     data.distance_rs.attrs  .update(units='km',   long_name='Distance of radiosonde to the lidars')
#     data.wvmr_aws.attrs     .update(units='g/kg', long_name='AWS mixing ratio')
#     data.temp_aws.attrs     .update(units='°C',   long_name='AWS temperature')
#     data.distance_aws.attrs .update(units='km',   long_name='Distance of weather stations to the lidars')
    
#     return data


# def fuse_dial_aws_with_raso(dates, rsondes, da10, awstations): 
    
#     dial = da10.copy()
#     aws  = awstations.copy()
#     raso = rsondes.copy()
    
#     raso = raso.where(raso['date'].isin(dates), drop=True)
    
#     # Mask da10 water-vapor above water_vapor_max_range
#     dial = dial.rename({"water_vapor":            "mr",
#                     "water_vapor_uncertainty":    "mr_unc",
#                     "water_vapor_max_range":      "mr_maxrange"})
   
#     dial['mr']     = dial['mr']    .where(dial['height'] <= dial['mr_maxrange'])
#     dial['mr_unc'] = dial['mr_unc'].where(dial['height'] <= dial['mr_maxrange'])
    
#     processed_data = []
    
#     for i in raso.launch.values:
        
#         #i = 62  # choose one radiosonde launch for testing (example i = 73)
#         try: 
#             # --- select launch and discard all NaNs from the combined dataset
            
#             rs = raso.sel(launch=i)
#             rs = rs.where(rs.time.notnull(), drop=True)
#             print("Launch label:", i, 
#                   "Date:", rs.date.values.astype('datetime64[D]'), 
#                   "When:", rs.day_night.values, '\n')
            
#             # --- select the corresponding lidar timesteps
            
#             # derive key times from radiosonde
#             time_rs0km = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
#             time_rs2km = rs.time.where(rs.altitude > 2300, drop=True).values[0]
#             time_rs4km = rs.time.where(rs.altitude > 4000, drop=True).values[0]
            
#             mask_da  = ((dial.time >= (time_rs0km - np.timedelta64(1,  'm'))) & 
#                         (dial.time <= (time_rs4km + np.timedelta64(1,  'm'))))
#             mask_aws = ((aws.time  >= (time_rs0km - np.timedelta64(10, 'm'))) & 
#                         (aws.time  <= (time_rs2km + np.timedelta64(10, 'm')))) 
            
#             # extract the matching time windows in each remote-sensing instrument
#             da  = dial.sel(time=mask_da)            
#             ws  = aws.sel(time=mask_aws)
            
#             # --- Adjust layout
            
#             # Unify 'height' coordinates with 577 m ASL as ground reference
#             rs['height']  = rs['altitude'] - 577         # 0 -> 571 m ASL
#             ws['height']  = ws['altitude'] - 577         # 0 -> 571 m ASL
            
#             # Drop unnecassary data 
#             rs = rs[['time', 'height', 'mr', 't', 'p', 'lon', 'lat', 'launch','date', 'day_night']]
#             da = da[['mr', 'mr_unc', 'longitude', 'latitude'] ]            
            
#             # --- define a fixed height & time grid:
            
#             gridheight = np.arange(0, 12001, 10, dtype='int64')
                       
#             # Interpolate rsb on new  height grid
#             rs["time"] = rs["time"].astype('int64') # Cast time to int64 (Nonoseconds since 1970-01-01) 
#             rs = rs.swap_dims({"index": "height"})
#             rs = rs.interp({'height': gridheight}, method='linear')
#             rs["time"] = rs["time"].astype('datetime64[ns]') # Cast back
            
#             # define time grid
#             gridtime = rs.time.values.astype('datetime64[ns]')
            
#             # make sure all times are unique and handle NaT
#             vals, idx = np.unique(da['time'].values, return_index=True)
#             da = da.isel(time=np.sort(idx))
            
#             # --- Interpolate raso, dial & ppl on the height & time grid 
            
#             # rs interp     # 5334->1200,dh ~4  ->10
#             # da interp     # 415 ->1200,dh 9.6 ->10
            
#             da  =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
            
#             # Extract diagonal 
#             da_mr = da['mr'].values.diagonal() #.where(~nat_mask).diagonal()
#             da_mr_unc = da['mr_unc'].values.diagonal() #.where(~nat_mask).diagonal()
            
            
#             # Calculate the distance from the radiosonde to the lidars in km
#             distance_rs =  haversine(da.latitude.item(), da.longitude.item(), 
#                                      rs['lat'].values, rs['lon'].values)
            
#             # Asemble in Dataset
            
#             ds_joint = xr.Dataset({
#                 'time'         : (('height'),gridtime),
#                 'wvmr_rs'        : (('height'), rs['mr'].values),
#                 'temp_rs'         : (('height'), rs['t'].values),
#                 'wvmr_dial'      : (('height'), da_mr),
#                 'wvmr_unc_dial'  : (('height'), da_mr_unc),
#                 'distance_rs'  : (('height'), distance_rs),

#             }, coords={
#                 'height'   : gridheight,
#                 'launch'   : rs.launch.item(),
#                 'date'     : rs.date.values,
#                 'day_night': rs.day_night.item()
#             })
            
                            
#             # --- Interpolate station data on the height & time grid 

#             # compute station grid heights
#             closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
#             gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
            
#             # Build a station‐indexed xarray of target heights
#             gridheightWS = xr.DataArray(gridheightWS,
#                                 dims='station',
#                                 coords={'station': ws['station'].values},
#                                 name='height'
#                                 )
#             # Interpolate radiosonde times onto station elevations 
#             # → get time when sonde passes each station
#             time_rs_at_aws = (rs['time']
#                               .astype('int64')
#                               .interp(height=gridheightWS, method='nearest')
#                               .astype('datetime64[ns]')
#                               )
#             time_rs_at_aws.loc[{'station': 'Tawes'}] = time_rs0km #otherwise NaT

#             # pull aws data at those times
#             mr_ws = ws['mr'] .interp(time=time_rs_at_aws, method='linear')
#             t_ws  = ws['t']  .interp(time=time_rs_at_aws, method='linear')


#             # Calculate the distance from the stations to the lidars in km
#             distance_aws =  haversine(da.latitude.item(), da.longitude.item(), 
#                                       ws['lat'].values, ws['lon'].values)
            
#             #Assemble as new Dataset
#             ds_aws = xr.Dataset(
#                 {
#                      'time_aws'    : (('station'), time_rs_at_aws.values),      # (station,)
#                      'wvmr_aws'    : (('station'), mr_ws.values),               # (station,)
#                      'temp_aws'    : (('station'), t_ws.values),                # (station,)
#                      'height_aws'  : (('station'), gridheightWS.values),
#                      'shortcut_aws': (('station'), ws['shortcut'].values),
#                      'distance_aws': (('station'), distance_aws),
#                      },
#                      coords={'station': ws['station'].values
#                     }
#             )

#             # Merge with other gidded data
#             ds_new = xr.merge([ds_joint, ds_aws])
            
#             #Append to list of all launches
#             processed_data.append(ds_new)
        
#         # Handle exceptions as not existing data by skipping them
#         except Exception as e:
#             print(f"Error with launch nr {rs.launch.item()}: {e}")
#             continue
    
#     data = xr.concat(processed_data, dim='launch')
    
#     # add metadata
#     data.attrs['description'] = (
#         'joint radiosonde, DA10 & PPL & weather station water-vapor-mixing‐ratio and temperature '
#         'on a uniform 4 m grid, time‐synced to radiosonde'
#         )
#     data.wvmr_rs.attrs      .update(units='g/kg', long_name='RS mixing ratio')
#     data.temp_rs.attrs      .update(units='°C',   long_name='RS temperature')
#     data.wvmr_dial.attrs    .update(units='g/kg', long_name='DIAL mixing ratio')
#     data.wvmr_unc_dial.attrs.update(units='g/kg', long_name='DIAL mr uncertainty')
#     data.distance_rs.attrs  .update(units='km',   long_name='Distance of radiosonde to the lidars')
#     data.wvmr_aws.attrs     .update(units='g/kg', long_name='AWS mixing ratio')
#     data.temp_aws.attrs     .update(units='°C',   long_name='AWS temperature')
#     data.distance_aws.attrs .update(units='km',   long_name='Distance of weather stations to the lidars')
    
#     return data

# start1, end1 = datetime(2024, 6, 18), datetime(2024,  8,  6)
# start2, end2 = datetime(2024, 8,  7), datetime(2024,  9,  8)
# start3, end3 = datetime(2024, 9, 10), datetime(2024, 10, 22)

# dates1 = [start1 + timedelta(days=x) for x in range((end1 - start1).days + 1)] 
# dates2 = [start2 + timedelta(days=x) for x in range((end2 - start2).days + 1)] 
# dates3 = [start3 + timedelta(days=x) for x in range((end3 - start3).days + 1)] 

# dates1 = np.array(dates1, dtype="datetime64[ns]")
# dates2 = np.array(dates2, dtype="datetime64[ns]")
# dates3 = np.array(dates3, dtype="datetime64[ns]")
# i = [48, 67, 68, 79, 98]

# start, end = datetime(2024, 6, 19), datetime(2024, 10, 22)
# dates = [start + timedelta(days=x) for x in range((end - start).days + 1)]
# dates = np.array(dates, dtype="datetime64[ns]")

# ds1 = fuse_dial_aws_with_raso      (dates1, rsondes, da10, awstations)
# ds2 = fuse_raman_dial_aws_with_raso(dates2, rsondes, da10, awstations, ppl, ppl2)
# ds3 = fuse_dial_aws_with_raso      (dates3, rsondes, da10, awstations)
# ds4 = fuse_dial_aws_with_raso      (datesx, rsondes, da10, awstations)
# ds = xr.concat([ds1, ds2, ds3], dim= 'launch')
#%% Radiosonde with DA10 & AWS

# hmax = 6000 # m
# dh = 10      # m
# dt = 20      # min
# # aws     # dt 10 min               # hmax 2300  m
# # da      # dt  1 min  # dh 9.6  m  # hmax 4000  m
# # pl      # dt 20 min  # dh 3.75 m  # hmax 12000 m

# def fuse_dial_aws_with_raso(dates, raso, dial, aws): 
    
#     raso = raso.where(raso['date'].isin(dates), drop=True)
#     processed_data = []
    
#     for i in raso.launch.values:
        
#         #i = 61  # choose one radiosonde launch for testing (example i = 73)
#         try: 
#             # --- select launch and discard all NaNs from the combined dataset
            
#             rs = raso.sel(launch=i)
#             rs = rs.where(rs.time.notnull(), drop=True)
#             print("Launch label:", i, 
#                   "Date:", rs.date.values.astype('datetime64[D]'), 
#                   "When:", rs.day_night.values, '\n')
            
#             # --- select the corresponding lidar timesteps
            
#             # derive key times from radiosonde
#             time_rs0km = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
#             time_rs2km = rs.time.where(rs.altitude > 2300, drop=True).values[0]
#             time_rs4km = rs.time.where(rs.altitude > 4000, drop=True).values[0]
            
#             # extract the matching time windows in each remote-sensing instrument
#             da = dial.sel(time=slice(time_rs0km - np.timedelta64(1, 'm'),
#                                      time_rs4km  + np.timedelta64(1, 'm')))            
#             ws = aws.sel(time=slice(time_rs0km - np.timedelta64(10, 'm'),
#                                      time_rs2km  + np.timedelta64(10, 'm')))
            
#             # --- Adjust layout
            
#             # Mask da10 water-vapor above water_vapor_max_range
#             da = da.rename({"water_vapor":                "mr",
#                             "water_vapor_uncertainty":    "mr_unc",
#                             "water_vapor_max_range":      "mr_maxrange"     })
           
#             da['mr']     = da['mr']    .where(da['height'] <= da['mr_maxrange'])
#             da['mr_unc'] = da['mr_unc'].where(da['height'] <= da['mr_maxrange'])
            
#             # Unify 'height' coordinates with 577 m ASL as ground reference
#             rs['height']  = rs['altitude'] - 577         # 0 -> 571 m ASL
#             ws['height']  = ws['altitude'] - 577         # 0 -> 571 m ASL
            
#             # Drop unnecassary data 
#             rs = rs[['time', 'height', 'mr', 't', 'p', 'lon', 'lat', 'launch', 'date', 'day_night']]
#             da = da[['mr', 'mr_unc', 'longitude', 'latitude'] ]            
            
#             # --- define a fixed height & time grid:
            
#             gridheight = np.arange(0, 12001, 10, dtype='int64')
                       
#             # Interpolate rsb on new  height grid
#             rs["time"] = rs["time"].astype('int64') # Cast time to int64 (Nonoseconds since 1970-01-01) 
#             rs = rs.swap_dims({"index": "height"})
#             rs = rs.interp({'height': gridheight}, method='linear')
#             rs["time"] = rs["time"].astype('datetime64[ns]') # Cast back
            
#             # define time grid
#             gridtime = rs.time.values.astype('datetime64[ns]')
            
#             # make sure all times are unique and handle NaT
#             vals, idx = np.unique(da['time'].values, return_index=True)
#             da = da.isel(time=np.sort(idx))
            
#             # --- Interpolate raso, dial & ppl on the height & time grid 
            
#             # rs interp     # 5334->1200,dh ~4  ->10
#             # da interp     # 415 ->1200,dh 9.6 ->10
            
#             da  =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
            
#             # Extract diagonal 
#             da_mr = da['mr'].values.diagonal() #.where(~nat_mask).diagonal()
#             da_mr_unc = da['mr_unc'].values.diagonal() #.where(~nat_mask).diagonal()
            
            
#             # Calculate the distance from the radiosonde to the lidars in km
#             distance_rs =  haversine(da.latitude.item(), da.longitude.item(), 
#                                      rs['lat'].values, rs['lon'].values)
            
#             # Asemble in Dataset
            
#             ds_joint = xr.Dataset({
#                 'time'         : (('height'),gridtime),
#                 'wvmr_rs'        : (('height'), rs['mr'].values),
#                 'temp_rs'         : (('height'), rs['t'].values),
#                 'wvmr_dial'      : (('height'), da_mr),
#                 'wvmr_unc_dial'  : (('height'), da_mr_unc),
#                 'distance_rs'  : (('height'), distance_rs),

#             }, coords={
#                 'height'   : gridheight,
#                 'launch'   : rs.launch.item(),
#                 'date'     : rs.date.values,
#                 'day_night': rs.day_night.item()
#             })
            
                            
#             # --- Interpolate station data on the height & time grid 

#             # compute station grid heights
#             closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
#             gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
            
#             # Build a station‐indexed xarray of target heights
#             gridheightWS = xr.DataArray(gridheightWS,
#                                 dims='station',
#                                 coords={'station': ws['station'].values},
#                                 name='height'
#                                 )
#             # Interpolate radiosonde times onto station elevations 
#             # → get time when sonde passes each station
#             time_rs_at_aws = (rs['time']
#                               .astype('int64')
#                               .interp(height=gridheightWS, method='nearest')
#                               .astype('datetime64[ns]')
#                               )
#             time_rs_at_aws.loc[{'station': 'Tawes'}] = time_rs0km #otherwise NaT

#             # pull aws data at those times
#             mr_ws = ws['mr'] .interp(time=time_rs_at_aws, method='linear')
#             t_ws  = ws['t']  .interp(time=time_rs_at_aws, method='linear')


#             # Calculate the distance from the stations to the lidars in km
#             distance_aws =  haversine(da.latitude.item(), da.longitude.item(), 
#                                       ws['lat'].values, ws['lon'].values)
            
#             #Assemble as new Dataset
#             ds_aws = xr.Dataset(
#                 {
#                      'time_aws'    : (('station'), time_rs_at_aws.values),      # (station,)
#                      'mr_aws'      : (('station'), mr_ws.values),               # (station,)
#                      't_aws'       : (('station'), t_ws.values),                # (station,)
#                      'height_aws'  : (('station'), gridheightWS.values),
#                      'shortcut_aws': (('station'), ws['shortcut'].values),
#                      'distance_aws': (('station'), distance_aws),
#                      },
#                      coords={'station': ws['station'].values
#                     }
#             )

#             # Merge with other gidded data
#             ds_new = xr.merge([ds_joint, ds_aws])
            
#             #Append to list of all launches
#             processed_data.append(ds_new)
        
#         # Handle exceptions as not existing data by skipping them
#         except Exception as e:
#             print(f"Error with launch nr {rs.launch.item()}: {e}")
#             continue
    
#     data = xr.concat(processed_data, dim='launch')
    
#     # add metadata
#     data.attrs['description'] = (
#         'joint radiosonde, DA10 & PPL & weather station water-vapor-mixing‐ratio and temperature '
#         'on a uniform 4 m grid, time‐synced to radiosonde'
#         )
#     data.mr_rs.attrs      .update(units='g/kg', long_name='RS mixing ratio')
#     data.t_rs.attrs       .update(units='°C',   long_name='RS temperature')
#     data.mr_da10.attrs    .update(units='g/kg', long_name='DIAL mixing ratio')
#     data.mr_unc_da10.attrs.update(units='g/kg', long_name='DIAL mr uncertainty')
#     data.distance_rs.attrs.update(units='km',   long_name='Distance of radiosonde to the lidars')
#     data.mr_aws.attrs      .update(units='g/kg', long_name='AWS mixing ratio')
#     data.t_aws.attrs       .update(units='°C',   long_name='AWS temperature')
#     data.distance_aws.attrs.update(units='km',   long_name='Distance of weather stations to the lidars')
    
#     return data


# def fuse_dial_raman_aws(ppl2, da10, awstations, hmax, dh, dt):
#     start = np.datetime64('2024-08-14T00:00')
#     end = np.datetime64('2024-09-09T00:01')
    
#     # Select measurement period of PPL
#     pl = ppl2.copy()
#     da = da10.sel(time=slice(start, end))         
#     ws = awstations .sel(time=slice(start, end))
            
#     # --- define a fixed height & time grid
    
#     gridheight = np.arange(0, hmax+1 , dh, dtype='int64')
#     gridtime = np.arange(start, end, np.timedelta64(dt, 'm'), dtype='datetime64[ns]')
    
    
#     # --- Calculate the distance from the stations to the lidars in km
#     ws['distance_aws'] =  haversine(da.latitude.item(), da.longitude.item(), 
#                                     ws['lat'].values, ws['lon'].values)
    
#     # --- Adjust layout
            
#     # Mask da10 water-vapor above water_vapor_max_range
#     da = da.rename({"water_vapor":                "mr",
#                     "water_vapor_uncertainty":    "mr_unc",
#                     "water_vapor_max_range":      "mr_maxrange"     })
       
#     da['mr']     = da['mr']    .where(da['height'] <= da['mr_maxrange'])
#     da['mr_unc'] = da['mr_unc'].where(da['height'] <= da['mr_maxrange'])
    
#     # Unify 'height' coordinates with 577 m ASL as ground reference
#     pl['height'] = pl['height'] - 3            # 574 -> 571 m ASL
#     ws['height'] = ws['altitude'] - 577         # 0 -> 571 m ASL
    
    
#     # Keep only relevant data 
#     da = da[['mr', 'mr_unc', 'longitude', 'latitude'] ] 
#     pl = pl[['mr', 't']]
#     ws = ws[['mr', 't',  'height', 'distance_aws', 'shortcut']]
    
    
#     # make sure all times are unique and handle NaT
#     vals_da, idx_da = np.unique(da['time'].values, return_index=True)
#     da              = da.isel(time=np.sort(idx_da))
#     vals_pl, idx_pl = np.unique(pl['time'].values, return_index=True)
#     pl              = pl.isel(time=np.sort(idx_pl))
    
#     # --- Interpolate raso, dial & ppl on the height & time grid 
#     da  =  da.interp({'height': gridheight, 'time': gridtime}, method='linear')
#     pl  =  pl.interp({'height': gridheight, 'time': gridtime}, method='linear')
    
#     # Asemble in Dataset
    
#     ds_lidars = xr.Dataset({
#         'mr_da10'      : (('time','height'), da.mr.values),
#         'mr_unc_da10'  : (('time','height'), da.mr_unc.values),
#         't_ppl'        : (('time','height'), pl.t.values),
#         'mr_ppl'       : (('time','height'), pl.mr.values),
#         }, 
#         coords={
#         'height': gridheight,
#         'time'  : gridtime,
#         },
#         attrs={
#             'longnitude_lidars': da.longitude.item(),
#             'latitude_lidars': da.latitude.item()
#         })
     
#     # --- Interpolate station data on the height & time grid 
    
#     # compute station grid heights
#     closest_idx = np.abs(gridheight[:, None] - ws.height.values).argmin(axis=0)
#     gridheightWS = gridheight[closest_idx]  # shape = (n_stations,)
    
#     # pull aws data at those times
#     mr_ws = ws['mr'] .interp(time=gridtime, method='linear')
#     t_ws  = ws['t']  .interp(time=gridtime, method='linear')
    
    
#     #Assemble as new Dataset
#     ds_aws = xr.Dataset(
#         {
#              'mr_aws'      : (('station', 'time'), mr_ws.values),               # (station,)
#              't_aws'       : (('station', 'time'), t_ws.values),                # (station,)
#              'height_aws'  : (('station'), gridheightWS),
#              'shortcut_aws': (('station'), ws['shortcut'].values),
#              'distance_aws': (('station'), ws['distance_aws'].values),
#              },
#              coords={'station': ws['station'].values,
#                      'time'   : gridtime,
#             }
#     )
    
#     # Merge with other gidded data
#     data = xr.merge([ds_lidars, ds_aws])
    
    
#     # add metadata
#     data.attrs['description'] = (
#         'DA10, PPL and AWs water-vapor-mixing‐ratio and temperature data'
#         'on a uniform 10 m x 20 min grid'
#     )
#     data.mr_da10.attrs     .update(units='g/kg',   long_name='DIAL mixing ratio')
#     data.mr_unc_da10.attrs .update(units='g/kg',   long_name='DIAL mr uncertainty')
#     data.t_ppl.attrs       .update(units='°C',     long_name='PPL temperature')
#     data.mr_ppl.attrs      .update(units='g/kg',   long_name='PPL mixing ratio')
#     data.t_aws.attrs       .update(units='°C',     long_name='AWS temperature')
#     data.mr_aws.attrs      .update(units='g/kg',   long_name='AWS mixing ratio')
#     data.distance_aws.attrs.update(units='g/kg',   long_name='Distance of the stations to the lidars')

#     return data


# start, end = datetime(2024, 6, 18), datetime(2024, 10, 22)
# dates = [start + timedelta(days=x) for x in range((end - start).days + 1)] 
# dates = np.array(dates, dtype="datetime64[ns]")

# ds2 = fuse_dial_aws_with_raso(dates, rsondes, da10, awstations)
# outpath2 = os.path.join(os.path.dirname(os.getcwd()), 'data', 'joint_vertical_profiles_raso-dial-aws.nc')
# ds2.to_netcdf(outpath2)
# print(f'saved joint dataset as {outpath2}')

#%% DA10 & PPL (20 min) & AWS
# ds3 = fuse_dial_raman_aws(ppl2, da10, awstations, hmax, dh, dt)
# outpath3 = os.path.join(os.path.dirname(os.getcwd()), 'data', f'2d_dial-raman20min-aws_dt{dt}min_dh{dh}m_hmax{hmax}m.nc')
# ds3.to_netcdf(outpath3)
# print(f'saved joint dataset as {outpath3}')
    
# dstest = xr.open_dataset(outpath2)
# print(dstest)    
 
#%%

import pandas as pd

def time_to_height(ds, target_height):
    """Time difference in minutes from launch to reaching target_height."""
    results = {}
    for launch in ds.launch.values:
        da = ds.sel(launch=launch)
        h = da['height'].values
        t = da['time'].values
        
        # First valid time = launch time
        valid = ~np.isnan(h)
        if not valid.any():
            results[launch] = np.nan
            continue
        
        t_launch = t[valid][0]
        
        # First index where height >= target_height
        above = np.where(h >= target_height)[0]
        if len(above) == 0:
            results[launch] = np.nan
            continue
        
        t_target = t[above[0]]
        results[launch] = (t_target - t_launch) / np.timedelta64(1, 'm')
    
    return pd.Series(results, name=f'min_to_{target_height}m')


_, _, _, _, _, raso = adjust_data(da10_wvmr, da10_abs, awstations, ppl10s, ppl20m, rsondes)

i=76
rs = raso.sel(launch=i)
rs = rs.where(rs.time.notnull(), drop=True)

# Check if PPL measurement available and run respective function
print("Launch label:", i, 
      "| Date:", rs.date.values.astype('datetime64[D]'),
              "| When:", rs.day_night.values, '\n') 
launchtime  = rs.time.values.min().astype('datetime64[m]')
    # derive key times from radiosonde
time_rs0km  = rs.time.values.min() #.astype('M8[ms]').astype(datetime)
time_rs2km  = rs.time.where(rs.height > 2000,  drop=True).values[0]
time_rs4km  = rs.time.where(rs.height > 4000,  drop=True).values[0]
time_rs12km = rs.time.where(rs.height > 12000, drop=True).values[0]
        
dt_2km  = time_to_height(raso, 2000)
dt_4km  = time_to_height(raso, 4000)
dt_6km  = time_to_height(raso, 6000)
dt_12km = time_to_height(raso, 12000)

df = pd.DataFrame({'dt_2km_min': dt_2km, 'dt_4km_min': dt_4km, 'dt_6km_min': dt_6km, 'dt_12km_min': dt_12km})
print(df.describe())
