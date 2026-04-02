# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:02:07 2026

@author: alleh
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from metpy.units import units
import metpy.calc as mpcalc

def read_raso_files(files):
   
    all_data = [] # initialize list of dataframes for the chosen date
    df_old = pd.DataFrame({'time': [], 'p': [], 'h': [], 't': [], 'td': [], 
                           'rh': [], 'rv': [], 'dd': [], 'ff': [],
                           'theta': [], 'thetae': [], 'thetav': [] })
    rsfiles = []
    for csv_filepath in files:
       
        # Read the CSV file using pandas
        df = pd.read_csv(csv_filepath, delimiter=',', skipinitialspace=True) 
    
        # drop nans
        df = df.dropna(subset=[
            'pressure', 'geopotential height', 'temperature',
            'dew point temperature', 'relative humidity', 'mixing ratio',
            'wind direction', 'wind speed'])
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['pressure'])
        
        # Drop unnessasary columns
        df = df.drop(columns=['ice point temperature','humidity wrt ice'])
                    
        # Rename Columns to standard shortcut
        df.rename(columns={"longitude": "lon", 
                           "latitude": "lat",
                           "pressure": "p", # hPa
                           "geopotential height": "gph", # m ASL
                           "temperature": "t", # °C
                           "dew point temperature": "td", # °C
                           "relative humidity": "rh", # %
                           "mixing ratio": "mr",
                           "wind direction": "dd", # deg
                           "wind speed": "ff"}, # m/s?
                           inplace=True)
        
        #  ---- calculate some parameters (θ, θe and θv)
        p = df.p.to_numpy() * units.hPa
        t = df.t.to_numpy() * units.degC
        td = df.td.to_numpy() * units.degC
        mr = df.mr.to_numpy() * units('g/kg')
        gph = df.gph.to_numpy() 
        ff = df.ff.to_numpy() * units('m/s')
        dd = df.dd.to_numpy() * units.deg
        
        theta = mpcalc.potential_temperature(p, t)
        thetae = mpcalc.equivalent_potential_temperature(p, t , td)
        thetav = mpcalc.virtual_potential_temperature(p, t, mr)
        
        df["theta"] = theta.magnitude # K
        df["thetae"] = thetae.magnitude  # K
        df["thetav"] = thetav.magnitude  # K
        
        u, v = mpcalc.wind_components(ff, dd)
        df["u"] = u.magnitude # m/s
        df["v"] = v.magnitude # m/s
        
        # Calculate the apparent geometric height value above the earth's surface
        # to correctly model the effect of the Earth's curvature
        earth_radius = 6356766 # in m
        z = (earth_radius * gph) / (earth_radius - gph)
        df["z"] = z # m
        
        ds = xr.Dataset.from_dataframe(df)
        #ds = ds.set_coords("gph")
        #ds = ds.swap_dims({"index": "gph"})
        #ds = ds.groupby("gph").first()
        
        # ---- Append data to the list if new dataset
        if not df.equals(df_old):
            all_data.append(df)
            rsfiles.append(ds)
            
        df_old = df

    return rsfiles


def merge_raso():
    
    folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\raso"
    files = [str(p) for p in Path(folder).glob("*.csv")]
    nr_of_sondes = len(files)
    sondes = read_raso_files(files)
    
    # ---- xr array adjustments
    sondes = [ds.rename({"gph": "altitude"}) for ds in sondes]
    sondes = [ds.rename({"mr": "wvmr"}) for ds in sondes]
    sondes = [ds.rename({"t": "temp"}) for ds in sondes]
    sondes = [ds.drop_vars(['u', 'v', 'z', 'thetav', 'thetae']) for ds in sondes]
    
    # ---- add date, launch nr and day/night coord to filter later
    raso = []
    for i, ds in enumerate(sondes):
        launch_time = np.datetime64(ds.time.values[0])
        launch_date = launch_time.astype('datetime64[D]')
        hour = launch_time.astype(object).hour
        day_night = 'night' if hour < 6 or hour > 18 else 'day'
        ds = ds.assign_coords(launch=i, date=launch_date, day_night=day_night)
        raso.append(ds)

    # ---- Expand all indexes to same length
    max_len = max(ds.dims['index'] for ds in raso)
    common_index = np.arange(1, max_len + 1)
    raso = [ds.reindex(index=common_index) for ds in raso]

    # ---- Concatenate them along a new dimension 'launch' & save file
    combined = xr.concat(raso, dim='launch')

    # # --- Save to NetCDF and test
    # filename = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\radiosondes2.nc"
    # combined.to_netcdf(filename)
    
    return combined

#data = xr.open_dataset(r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\radiosondes.nc")

rs = merge_raso() 
filename = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\data\radiosondes2.nc"
rs.to_netcdf(filename) 
