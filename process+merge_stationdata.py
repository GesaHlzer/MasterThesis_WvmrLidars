# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:01:49 2025

@author: alleh
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime


#__________________________________________________
# Merge Lawis Data ('Seegrube' & 'Hafelekar')

# already done

def get_metadata(f):
     
    station_lookup = {
        '1': {'lon': 47.312079, 'lat': 11.383623, 'altitude': 2270, 'station': 'Hafelekar'},
        '2': {'lon': 47.306382, 'lat': 11.377934, 'altitude': 1921, 'station': 'Seegrube'}
    }
    parameter_lookup = {
        'LT': {'shortcut': 't',  'long_name': 'air temperature', 'unit': '°C'},
        'LF': {'shortcut': 'rh', 'long_name': 'relative humidity', 'unit': '%'},
        'TP': {'shortcut': 'td', 'long_name': 'dew point temperature', 'unit': '°C'},
        'WR': {'shortcut': 'dd', 'long_name': 'wind direction', 'unit': '°'},
        'WG': {'shortcut': 'ff', 'long_name': 'wind speed', 'unit': 'm/s'},
        }

    records = []

    #for i, f in enumerate(filenames):
    key = f[4]  # assuming f is a string (not Path object)
    par = f[6:8]
    station_info = station_lookup.get(key, {'lon': None, 'lat': None, 'altitude': None, 'station': 'Unknown'})
    parameter_info = parameter_lookup.get(par, {'shortcut': None, 'long_name': None, 'unit': None})
    records.append({
        #'index': i,
        'station': station_info['station'],
        'lon': station_info['lon'],
        'lat': station_info['lat'],
        'altitude': station_info['altitude'],
        'var_longname': parameter_info['long_name'],
        'var': parameter_info['shortcut'],
        'unit': parameter_info['unit'],
        'years': f[9:18],
        'filename': f
        })

    df = pd.DataFrame(records)
    return df

def get_dataset(key):
        
    filepath = os.path.join(os.path.dirname(os.getcwd()), 'data', 'Lawis')
    #key = 'ISEE2_LT'
    
    filenames = [f for f in os.listdir(filepath) if key in f and f.endswith('.csv')]
    data = [pd.read_csv(os.path.join(filepath, f),
                       delimiter=';', 
                       skiprows=15, 
                       encoding='latin1') 
           for f in filenames]
    
    f = filenames[0]
    metadata = get_metadata(f)

    df = pd.concat(data, axis=0, ignore_index=True) # fuse list
    
    df['Datum/Uhrzeit'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('Datum/Uhrzeit').reset_index(drop=True)
    df.columns = ['time', 'value']
    
    df = df[(df['time'] >= datetime(2024, 6, 18, 0, 0, 0)) & 
            (df['time'] <= datetime(2024, 10, 21, 12, 59, 59))]
    
    df['value'] = pd.to_numeric(df['value'].astype(str)
                                .str.replace(',', '.', regex=False),
                                errors='coerce')
    df = df.drop(df.index[15115])

    xs = xr.Dataset.from_dataframe(df)
    
    for i, meta_row in metadata.iterrows():
        station_name = meta_row['station']
        # df['station'] = station_name
        lon = meta_row['lon']
        lat = meta_row['lat']
        altitude = meta_row['altitude']
        unit = meta_row['unit']
        longname = meta_row['var_longname']
        var = meta_row['var']
        
    xs['lon'] = lon
    xs['lat'] = lat
    xs['altitude'] = altitude
    xs['station'] = station_name
    
    xs = xs.set_coords("time")
    xs = xs.swap_dims({"index": "time"})
    xs = xs.drop_vars('index')
    xs = xs.set_coords("station")
    xs = xs.expand_dims(dim='station')
    
    xs["value"].attrs["long_name"] = longname
    xs["value"].attrs["units"] = unit
    xs = xs.rename({'value': var})
    
    return xs

def merge_lawis_data():
        
    filepath = os.path.join(os.path.dirname(os.getcwd()), 'data', 'Lawis')
    keys = [f[:8] for f in os.listdir(filepath) if f.endswith('.csv')]
    
    t1 = get_dataset('ISEE1_LT')
    td1 = get_dataset('ISEE1_TP')
    rh1 = get_dataset('ISEE1_LF')
    dd1 = get_dataset('ISEE1_WR')
    ff1 = get_dataset('ISEE1_WG')
    
    t2 = get_dataset('ISEE2_LT')
    td2 = get_dataset('ISEE2_TP')
    rh2 = get_dataset('ISEE2_LF')
    
    # Fuse different datasets
    ds1 = xr.merge([t1, td1, rh1, dd1, ff1])
    ds2 = xr.merge([t2, td2, rh2])
    
    ds = xr.concat([ds1, ds2], dim='station')
    
    # save data
    filename = os.path.join(filepath, 'Lawis_data.nc')
    ds.to_netcdf(filename)
    
    # Verify stored dataset
    ds_check = xr.open_dataset(filename)
    print(ds_check)

#_________________________________________
# Merge Tempis Data ('HoettingerAlm', 'Rastlboden', 'Hungerburg',
# 'Alpenzoo',Hauptbahnhof' & 'OlympischesDorf')

# already done

def get_tempis_metadata(): 

    tempis_stations = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 
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
    folder_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'Tempis', 'data') 
    
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
            
            df_filtered = df[(df['time'] >= datetime(2024, 6, 18, 0, 0, 0)) & 
                             (df['time'] <= datetime(2024, 10, 31, 12, 59, 59))]
    
            
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
    station_attrs = {}
    for attr in ['shortcut', 'longitude', 'latitude', 'mASL']:
        station_attrs[attr] = (("station",), 
                               [datasets_updated[station][attr].iloc[0] 
                                for station in xs.station.values]
                               )
    xs = xs.assign(**station_attrs)
    
    xs = xs.rename({'temp (degC)': 't'})
    xs = xs.rename({'humidity (%)': 'rh'})
    xs = xs.rename({'longitude': 'lon'})
    xs = xs.rename({'latitude': 'lat'})
    xs = xs.rename({'mASL': 'altitude'})
    xs = xs.rename({'wind direction (deg)': 'dd'})
    xs = xs.rename({'wind speed (m/s)': 'ff'})
    
    xs["t"].attrs["units"] = '°C'
    xs["rh"].attrs["units"] = '%'
    xs["dd"].attrs["units"] = '°'
    xs["ff"].attrs["units"] = 'm s-1'
    
    xs["t"].attrs["long_name"] = 'Temperature in deg Celcius'
    xs["altitude"].attrs["long_name"] = 'Altitude in m ASL'
    
    return xs # returns combined_ds

def merge_tempis_data():
    
    ds = covert_to_xr()
    
    filename = os.path.join(os.path.dirname(os.getcwd()), 'data', 'Tempis', 'Tempis_data.nc')
    ds.to_netcdf(filename)
    
    # Verify stored dataset
    ds_check = xr.open_dataset(filename)
    print(ds_check)

#__________________________________________
# Read Tawes Data
def get_tawes():
    
    tawesfile = os.path.join(os.path.dirname(os.getcwd()), 'data', 'TAWES', 'data.csv')
    
    df = pd.read_csv(tawesfile, delimiter=';', skiprows=1)
    df.rename(columns={"rawdate": "time"}, inplace=True)

    df['time'] = pd.to_datetime(df['time'], errors='coerce') 
    
    # filter time for the measurement period
    df = df[(df['time'] >= datetime(2024, 6, 18, 0, 0, 0)) & 
            (df['time'] <= datetime(2024, 10, 31, 12, 59, 59))]
    df = df.drop_duplicates(subset='time', keep='first')
    
    # ---- Convert the DataFrame to an xarray.Dataset
    xs = xr.Dataset.from_dataframe(df)
    xs = xs.swap_dims({"index": "time"})
    # xs['height'] = 2
    # xs = xs.set_coords("height")
    xs = xs.drop_vars('index')
    
    # xs["value"].attrs["long_name"] = longname
    # xs["value"].attrs["units"] = unit
    xs = xs.rename({'tl': 't'})
    xs = xs.rename({'tp': 'td'})
    xs = xs.rename({'ffm': 'ff'})
    xs = xs.rename({'ddm': 'dd'})
    xs = xs.rename({'rf': 'rh'})
    
    xs['altitude'] = 579
    xs['lon'] = 11.3851659
    xs['lat'] = 47.2642936874
    data_tawes = xs[['t', 'td', 'rh', 'p', 'ff', 'dd', 'altitude', 'lon', 'lat']]
    
    data_tawes['station'] = 'Tawes'
    data_tawes = data_tawes.set_coords("station")
    data_tawes = data_tawes.expand_dims(dim='station')
    data_tawes['shortcut'] = 'TAWES'
    
    return data_tawes

#__________________________________________
def merge_all_stations():
    
    # --- Open Datasets and Combine
    
    tempis = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()), 'data', 'Tempis', 'Tempis_data.nc'))
    lawis = xr.open_dataset(os.path.join(os.path.dirname(os.getcwd()), 'data', 'Lawis', 'lawisdata.nc'))
    
    tawes = get_tawes()
    tawes = tawes.sel(time=tawes.time.isin(lawis.time))    

    ds = xr.concat([lawis, tempis, tawes], dim= "station")
    ds['shortcut'] = ds['shortcut'].where(ds['station'] != 'Hafelekar', other='HFK')
    ds['shortcut'] = ds['shortcut'].where(ds['station'] != 'Seegrube', other='SGR')
    
    # --- Physical constants
    
    g =  9.80665 #m/s²,
    M_d = 0.0289647 #kg/mol,
    R_d = 8.314462618 / 0.0289647  # ≈ 287.05 J/(kg·K)
    epsilon = 621.9800221013629 # g/kg
    e0 = 6.113 # hPa
    Lv = 2.5 * 10**6 #J/kg
    Rv = 461 # J/(K*kg)
    T0 = 273.15 # K
    
    # --- Use the barometric height formula to estimate pressure at each station
    
    # Reference pressure from Tawes 
    p_ref = ds['p'].sel(station='Tawes').astype(float)                     # shape: (time,)
    h_ref = ds['altitude'].sel(station='Tawes').item()       # scalar altitude
    t_ref = ds['t'].sel(station='Tawes').astype(float)  + 273.15  # °C → K, shape: (time,)
    
    # Get Target Altitudes & Temperatures
    stations = ds.station.values
    mask = stations != 'Tawes'

    h_other = ds['altitude'].sel(station=mask).astype(float)    # shape: (station,)
    t_other = ds['t'].sel(station=mask).astype(float) + 273.15  # °C → K, shape: (station, time)
    
    t_mean = ((t_other + t_ref) / 2)         #(station: 8, time: 19518)
    
    # Compute Pressure Using Broadcasting
    delta_h = (h_other - h_ref)#.expand_dims({'time': t_other.time}).T # (station: 8, time: 19518)
    p_ref_expanded = p_ref#.expand_dims({'station': t_other.station}) # (station: 8, time: 19518)
    
    exponent = (- (g * delta_h) / (R_d * t_mean))
    p_other = p_ref_expanded * xr.ufuncs.exp(exponent)  # shape: (station, time)
    
    
    # Add to combined dataset
    p_merged = xr.concat([p_other, p_ref], dim='station')
    ds['p_estimated'] = p_merged.T

    
    # ---- Calculate mixing ratio according to:
    #Atmospheric Thermodynamics book by C. Bohren and B. Albrecht (1998, Oxford Univ. Press, 402 pp).
    
    # Extract variables and convert types
    t = ds['t'].astype(float) + 273.15  # °C → K, shape: (station, time)
    rh_decimal = ds['rh'].astype(float) / 100.0  # in [0, 1], shape: (station, time)
    p = ds['p_estimated'].transpose('station', 'time')  # in hPa; shape: (station, time)
    
    # calculate wvmr 
    e = rh_decimal * (e0 * np.exp((Lv/Rv) * ((1 / T0) - ( 1 / t ))))                   
    wvmr = (e * epsilon) / (p - e)
    
    # Add mixing ratio to dataset
    ds['wvmr'] = wvmr
    
    # --- adjust dataset for saving
    ds['shortcut'] = ds['shortcut'].astype(str)
    ds['lon'] = (ds['lon']).astype(float)
    ds['lat'] = (ds['lat']).astype(float)
    ds['altitude'] = (ds['altitude']).astype(float)
    ds['t'] = ds['t'].astype(float)
    ds['rh'] = ds['rh'].astype(float)
   
    dd = ds['dd'].astype(str) 
    dd = xr.where(dd== '', np.nan, dd)   
    ds['dd'] = dd.astype(float)
    
    ff = ds['ff'].astype(str) 
    ff = xr.where(ff== '', np.nan, ff)   
    ds['ff'] = ff.astype(float)
    
    # mask values that have the error code -999
    mask_t = (ds.t == -999)
    ds['t'] = ds.t.where(~mask_t, other=np.nan)
    mask_rh = (ds.rh == -999)
    
    ds['rh'] = ds.rh.where(~mask_t, other=np.nan)
    
    # 2) mask wvmr if wvmr < 0 or wvmr > 60
    mask_wvmr = (ds.wvmr < 0) | (ds.wvmr > 60)
    ds['wvmr'] = ds.wvmr.where(~mask_wvmr, other=np.nan)

    ds = ds.rename({"t": "temp"}) #newer convention in my code

    
    return ds
 
data =  merge_all_stations()

filename = os.path.join(os.path.dirname(os.getcwd()), 'data', 'stationsdata.nc')
data.to_netcdf(filename)

# Verify stored dataset
ds_check = xr.open_dataset(filename)
print(ds_check)
wvmr=data['wvmr'].values.T
temp=data['temp'].values.T
