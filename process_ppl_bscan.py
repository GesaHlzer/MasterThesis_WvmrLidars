# -*- coding: utf-8 -*-

"""
Created on Fri May 09 08:52:12 2025
@author: gesa hoelzer using parts of federer-m 
"""
import numpy as np
import pandas as pd
import os
import xarray as xr
from datetime import datetime, timedelta

PS = 3230 # Profil size: Size of one Profil 
BS = 3.75 # Bin size = 3.75m
# W = 10 # 10 min. Window for floating average [min] (7.5 bevor and after)
var, avg_time = 'MR', '10s'

def get_filenames(var, avg_time):
    
    ppl_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'PPL', 'BScan') 
    filenames = [f for f in os.listdir(ppl_dir) if f.endswith('.bsc') and var in f and avg_time in f]
    # example 20240823_000010_to_20240823_060016_MRgl_10s_97m.bsc
    return filenames 
            
def p_bscan(filepath):
    '''
    Processing routine for statistical uncertainty calculations
    Input: filpath of the Bscan file
    Output: 
    '''
    Bscan_array = get_array(filepath)
    #header = get_header_YDM(Bscan_array)
    mr = Bscan_array.byteswap().view(Bscan_array.dtype.newbyteorder()) # byteswap().newbyteorder()
    mr = get_values(Bscan_array)
    #mr_avg = movavg_time(mr, avg_time) 
    mr = pd.DataFrame(mr)
    #height = mr.columns
    #height = pd.to_numeric(height)
    # mr = mr.transpose()
    # mr.insert(loc=0, column='height', value=height)
    # mr = mr.reset_index(drop=True)
    # mr = mr.transpose()
    return mr

def get_array(filepath):
    """
    Extract all Profiles of a Bscan to an array
        np.reshape(1, 2, 3) function is used to reshape an existing array
          1. array that you want to reshape
          2. resulting reshaped array will have int(x)(=number of profiles) rows 
          3. PS (profile size = 4030) columns 
          == One Row = One Profil 
          
    Example: T_0824 = get_array("20220824_145239_Tan_10s_97m.bsc")
    """
    res = np.fromfile(filepath, dtype=">f4")
    # res = res.view(res.dtype.newbyteorder('='))  # Convert to little-endian format
    x = res.shape[0] / PS
    array = np.reshape(res, (int(x), PS))
    
    return array

def get_header_YDM(array):

    """
    Extract all the Headers of a gluead Bscan to a Dataframe
    
    Example:
      array = read_bscan("20220908_145239_Tan_10s_97m.bsc")

      print(get_headers_glued(array))
    """

    headers = array[:, :30]
    dates = pd.DataFrame(headers[:,(13, 11, 12, 0, 1, 2)].astype(int).astype(str))
    dates.columns = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    df = pd.DataFrame(headers[:, [6, 19, 3, 21, 26, 28]])
    df.columns = [
        'latitude',      # [6]  = 47°N
        'longitude',     # [19] = 11°E
        'GPSaltitude',   # [3]  = 574m ASL
        'resRange',      # [21] = 3.75m
        'shots'        , # [26] variable
        'gates',         # [28] = 3230 = PS
    ]

    df['DateTime'] = pd.to_datetime(
        dates['year'] + '-' + dates['month'] + '-' + dates['day'] + ' ' +
        dates['hour'] + ':' + dates['minutes'] + ':' + dates['seconds']
    )
    return df

def get_values(array):
    
    """
    seperate Values --> drop Headers
    transform to a pandas dataframe
    add height (574 + Column indx * 3.75)
    
    Example:
    array = read_bscan("20220908_145239_Tan_10s_97m.bsc")
    values = get_values(array)
    
    """
    values = array[:, 30:] 
    values = pd.DataFrame(values)
    values.columns = [str(BS * (i + 1)) for i in range(values.shape[1])]
    
    return values

def sort_bscan_data(var, avg_time):
    filenames = get_filenames(var, avg_time)
    
    header_list = [] # MRgl 10s 97m, Tgl 10s 97m
    data_list = []
    
    for file in filenames:
        filepath = os.path.join(os.path.dirname(os.getcwd()), 'data', 'PPL', 'BSCAN', file)
    
        data0 = p_bscan(filepath) # mr, t, ...
        array = get_array(filepath)
        header0 = get_header_YDM(array) 
        
        header_list.append(header0)
        data_list.append(data0)
    
    header = pd.concat(header_list, axis=0, ignore_index=True)   
    data = pd.concat(data_list, axis=0, ignore_index=True)
    
    # height = data.columns
    # height = pd.to_numeric(height)
    # data = data.transpose()
    # data.insert(loc=0, column='height', value=height)
    # data = data.reset_index(drop=True)
    # data = data.transpose()

    return header, data

def create_netcdf(var, avg_time):
    
    # define the long name of the variable in use for saving
    if var == 'WV': var_name = 'wv'
    elif var == 'MR': var_name = 'wvmr'
    elif var == 'T': var_name = 'temp'
    elif var == 'BR': var_name = 'br'
    else: var_name = str(var)

    # Load and prepare data
    
    header, data = sort_bscan_data(var, avg_time)
    
    height = data.columns
    height = pd.to_numeric(height)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            str(var_name): (["time", "height"], data.values), #  Extract actual data (skip the first row)
            #"shots":       ("time", header["shots"].values.astype(np.float32)),
        },
        coords={
            "time": ("time", header["DateTime"].values),  # Store DateTime directly
            "height": ("height", height.values)  # First row contains height in meters 
        },
        attrs={
            "timestep": (avg_time),  # Adding metadata
            "resolution_range": (header["resRange"].values.astype(np.float32)[1]),
            "h_bins":  (header["gates"].values.astype(np.float32)[1]),
            "gliding_h_mean": ("Gliding mean with a window of dh = 97.5 m (26 bins)"),
            "latitude":    (52.208683),
            "longitude":   (14.122507),
            "GPS_altitude_asl": (header["GPSaltitude"].values.astype(np.float32)[1]),
            }
    )
    
    # Add time attributes (optional, improves NetCDF readability)
    ds["time"].attrs["long_name"] = "Date and time"
    
    ds["height"].attrs["long_name"] = "Distance from LIDAR, here height above ground level" 
    ds["height"].attrs["units"] = "m above GPS altitude of LIDAR" 
    
    # ds["latitude"].attrs["long_name"] = "'Platform latitude'"
    # ds["latitude"].attrs["units"] = "°"
    
    # ds["longitude"].attrs["long_name"] = "'Platform longitude'"
    # ds["longitude"].attrs["units"] = "°"
    
    # ds["GPSaltitude"].attrs["long_name"] = "GPS altitude of Lidar above mean sea level'"
    # ds["GPSaltitude"].attrs["units"] = "m ASL"
       
    # ds["resRange"].attrs["long_name"] = "Range resolution of data"
    # ds["resRange"].attrs["units"] = "m"
    
    # ds["shots"].attrs["long_name"] = "Number of laser shots used to derive this profile'"
    # ds["shots"].attrs["units"] = " "
        
    # ds["gates"].attrs["long_name"] = " "
    # ds["gates"].attrs["units"] = "Number of data bins in this profile (without header)"
       
    return ds


# perform for defined averaging time

# var = 'BR' # BR T MR
avg_time= '10s' #'1200s' # 10s'

# Generate the list of datetime days
date_beg = datetime(2024, 8, 23)
date_end = datetime(2024, 9, 8)
dates = [date_beg+timedelta(days=x) for x in range((date_end-date_beg).days+1)]

br   = create_netcdf('BR', avg_time)
temp = create_netcdf('T',  avg_time)
wvmr = create_netcdf('MR', avg_time)

ppl = wvmr.copy()
ppl['temp'] = temp['temp']
ppl['br']   = br['br']
ppl.attrs['title'] = f'Purple Pulse Lidar Data in {avg_time} steps'

ppl["temp"].attrs["long_name"] = "Temperature" 
ppl["temp"].attrs["units"]     = "K" 
ppl["wvmr"].attrs["long_name"] = "Water Vapor Mixing Ratio" 
ppl["wvmr"].attrs["units"]     = "g/kg" 
ppl["br"].attrs["long_name"] = "Backscatter Ratio" 
        

# Save dataset to NetCDF
    
# Define paths
target_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'PPL')
filename = os.path.join(target_dir, f'ppl_{avg_time}_gl97m.nc')
ppl.to_netcdf(filename)
print(f"Saved netCDF file at: {filename}. \n \n Open and verify saved file ... \n")

# Verify stored dataset
ds_check = xr.open_dataset(filename)

# #########################################################################

# PS = 4030 # Profil size: Size of one Profil 
# H  = 30   # Header: first 30 Index of a profile
# HL = 1500 # Height Lidar: Number of Bins until ca. 6 Km height
# BS = 3.75 # Bin size = 3.75m
# HR = 1300 # Height Radiosonde: Number of Bins until ca. 6 Km height
# LAL = 47.264 # Latitude position Lidar
# LOL = 11.385 # Longitude position Lidar
# W = 7.5 # 15 min. Window for floating average [min] (7.5 bevor and after)

# Constants for the standard atmosphere
# G0 = 9.80665  # Acceleration due to gravity, m/s^2
# R = 287.053  # Gas constant for dry air, J/kg*K
# M = 0.0289644  # Molar mass of dry air, kg/mol
# T0 = 288.15  # Temperature at sea level, K
# L = -0.0065  # Temperature lapse rate, K/m
# P0 = 101325  # Pressure at sea level, Pa

