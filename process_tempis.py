# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:00:34 2026

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


# from metpy.units import units
# import metpy.calc as mpcalc
# import matplotlib.dates as mdates
    
def read_tempis(date=None): 

    main_dir = os.path.dirname(os.getcwd())
    
    def get_tempis_metadata(): 
        tempis_stations = pd.read_csv(os.path.join(main_dir, 
                                      'data', 
                                      'Tempis', 
                                      'tempis_station_data.txt'),
                                      delimiter=" ", 
                                      header=None
                                      )
        
        # Convert `tempis_stations` first row into column headers
        tempis_stations.columns = tempis_stations.iloc[0]  # Set first row as column headers
        tempis_stations = tempis_stations[1:].reset_index(drop=True)  # Drop original first row
        
        # Create a lookup dictionary using the station 'Name' column.
        station_info_mapping = {}
        
        for _, row in tempis_stations.iterrows():
            station_info_mapping[row["Name"]] = row[['shortcut', 
                                                     'longitude', 
                                                     'latitude', 
                                                     'mASL']]
        
        return station_info_mapping
    
    def read_dat():       
        # Create an xarray Dataset for time-series data
        dataset_dict = {}
        folder_path = os.path.join(main_dir, 'data', 'Tempis', 'data') 
        
        for file in os.listdir(folder_path):
            
            if file.endswith(".dat"):
                file_path = os.path.join(folder_path, file)
        
                with open(file_path, "r") as f:
                    lines = f.readlines()
        
                # Extract station name from header
                station_name = None
                for line in lines:
                    if line.startswith("% Station:"):
                        station_name = line.split(":")[1].strip()
                        print(station_name)
                        break
        
                # Read data
                data = [line.strip().split() 
                        for line in lines 
                        if not line.startswith("%")]
                
                # Differentiate between the stations with and without wind data.
                if len(data[0]) == 3:             
                    df = pd.DataFrame(data, columns=["UTC-date", 
                                                     "temp (degC)", 
                                                     "humidity (%)"]
                                      )
                elif len(data[0]) == 5:             
                    df = pd.DataFrame(data, columns=["UTC-date", 
                                                     "temp (degC)", 
                                                     "humidity (%)", 
                                                     "wind direction (deg)", 
                                                     "wind speed (m/s)"]
                                      )
                else:
                    break
                
                # Convert date to pd datetime
                df["UTC-date"] = pd.to_datetime(df["UTC-date"], format="%Y%m%d%H%M")  
                df.rename(columns={"UTC-date": "time"}, inplace=True)
                
                if date is None:
                    # Filter data for June to October 2024
                    df_filtered = df[(df["time"].dt.year == 2024) 
                                     & (df["time"].dt.month.between(6, 10))]
                    
                else:
                    # Filter for the date range between date_beg & date_end
                    df_filtered = df[(df['time'] >= date) 
                                     & (df['time'] <= date+timedelta(days=1))
                                     ]
                
                dataset_dict[station_name] = df_filtered
                
        return dataset_dict
    
    def fuse_info():
        
        # get data
        datasets_dict = read_dat()
        station_info = get_tempis_metadata()
        
        # Add information from station info to the dataset dict
        datasets_updated = {}
        
        for station_name, df in datasets_dict.items():
            # Convert the date column to datetime and set it as the index.
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            
            if station_name in station_info:
                # Get the station info (as a pandas Series)
                info = station_info[station_name]
                # Add each attribute as a new column, constant over time for the given station.
                for attr in info.index:
                    df[attr] = info[attr]
            else:
                print(f"Warning: No station info found for {station_name}")
            
            datasets_updated[station_name] = df
            
        return datasets_updated
    
    def covert_to_xr():
        datasets_updated = fuse_info()
        
        # Convert each updated DataFrame to an xarray Dataset and add a station dimension.
        ds_list = []
        for station, df in datasets_updated.items():
            ds_station = df.to_xarray()
            # Expand dimensions to include a "station" coordinate.
            ds_station = ds_station.expand_dims(station=[station])
            ds_list.append(ds_station)
        
        # Concatenate along the "station" dimension to create one combined dataset.
        xs = xr.concat(ds_list, dim="station")
        
        # Extract station attributes that are constant for a station.
        
        # attrs = {}
        # for attr in ['shortcut', 'longitude', 'latitude', 'mASL']:
        #     attrs[attr] = [datasets_updated[station][attr].iloc[0] for station in combined_ds.station.values]
        # # Assign these attributes as coordinates indexed by station.
        # combined_ds = combined_ds.assign_coords(**attrs)
        
        station_attrs = {}
        for attr in ['shortcut', 'longitude', 'latitude', 'mASL']:
            station_attrs[attr] = (("station",), 
                                   [datasets_updated[station][attr].iloc[0] 
                                    for station in xs.station.values]
                                   )
        
        # Assign these as data variables (from a station)
        xs = xs.assign(**station_attrs)
        
        xs = xs.rename({'temp (degC)': 't'})
        xs = xs.rename({'humidity (%)': 'rh'})
        xs = xs.rename({'longitude': 'lon'})
        xs = xs.rename({'latitude': 'lat'})
        xs = xs.rename({'mASL': 'altitude'})
        xs = xs.rename({'wind direction (deg)': 'dd'})
        xs = xs.rename({'wind speed (m/s)': 'ff'})
        
        xs["t"].attrs["units"] = '°C'
        xs["t"].attrs["units"] = '°C'
        xs["rh"].attrs["units"] = '%'
        xs["dd"].attrs["units"] = '°'
        xs["ff"].attrs["units"] = 'm s-1'
        
        xs["t"].attrs["long_name"] = 'Temperature in deg Celcius'
        xs["altitude"].attrs["long_name"] = 'Altitude in m ASL'
        
        return xs # returns combined_ds
    
    return covert_to_xr()