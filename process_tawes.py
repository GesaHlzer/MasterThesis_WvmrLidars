# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:59:28 2026

@author: alleh
"""

# import numpy as np
import pandas as pd
# import netCDF4
import xarray as xr
import os
# from datetime import datetime, timezone, timedelta 
# from pathlib import Path
# import cftime
# import cfgrib 
# import xarray as xr


from metpy.units import units
import metpy.calc as mpcalc
# import matplotlib.dates as mdates

def read_tawes(date):
    tawesfile = os.path.join(os.path.dirname(os.getcwd()), 
                             'data', 
                             'TAWES', 
                             'data.csv'
                             )
    # Read the CSV file
    df = pd.read_csv(tawesfile, delimiter=';', skiprows=1)
    
    # Rename 'rawdate' to 'time' and convert to datetime
    df.rename(columns={"rawdate": "time"}, inplace=True)
     # Convert to datetime, handling invalid entries
    df['time'] = pd.to_datetime(df['time'], errors='coerce') 
    
    # Filter the DataFrame to include only rows between date_beg and date_end
    df = df[(df['time'] >= date) & 
            (df['time'] <= date + timedelta(days=1))]
    
    # If no rows remain after filtering, raise a warning
    if df.empty:
        print(f"No data available for {date}.")
        return None
    
    # ---- Calculate mixing ratio using MetPy
    pressure = df['p'].values * units.hPa           # Convert pressure to hectopascals
    temperature = (df['tl'].values) * units.degC    # Temperature in °C
    relative_humidity = (df['rf'].values / 100.0)   # Relative humidity as a fraction
    temperature2 = (df['tl2'].values) * units.degC  # Temperature in °C
    relative_humidity2 = (df['rf2'].values / 100.0) # Relative humidity as a fraction
    
    df['mr'] = mpcalc.mixing_ratio_from_relative_humidity(pressure, 
                                                          temperature, 
                                                          relative_humidity
                                                          )
    df['mr2'] = mpcalc.mixing_ratio_from_relative_humidity(pressure, 
                                                           temperature2, 
                                                           relative_humidity2
                                                           ) 
    # ---- Convert the DataFrame to an xarray.Dataset
    data_tawes = xr.Dataset.from_dataframe(df)
    data_tawes = data_tawes.swap_dims({"index": "time"})
    data_tawes['height']= 2
    data_tawes = data_tawes.set_coords("height")
    
    return data_tawes