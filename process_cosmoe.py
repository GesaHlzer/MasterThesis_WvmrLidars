# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:55:48 2026

@author: alleh
"""

# import numpy as np
# import pandas as pd
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


date = datetime(2024, 8, 22)
start = np.datetime64("2024-08-22")

def reas_cosmoe(date):
    date_str = date.strftime('%Y_%m_%d')
    folder = fr'C:/Users/alleh/Documents/+Uni_Innsbruck/+MasterThesis/data/COSMOE/{date_str}_03_icon-ch1-eps_uibk_acinn/'
    filenames = [f for f in os.listdir(folder) if f.endswith('.grb2')]
    # filepaths = [os.join(folder, filename) for filename in filenames]
    ds = [cfgrib.open_datasets(os.path.join(folder, filename)) for filename in filenames]
    
    ds0 = ds[0]
    
    param =['parameter','member','level','time','leadtime',
            'PAY','BRN','ALT', 'FRLUX','FRLYO','FRNIM','DLIDA','DLSTU','DLSTS',
            'DLKUE','DLMUE', 'DLKEM','OSWIE','OSINN','CRZAG','IYCAM','IYLIN','IYSAN',
            'SHA','GRE','SMA','GVE','BAS','MLS','NAP','SAE','NVI','OSWIS','OSINU',
            'HFL','00',,'01gk','20240908 03:00','000:00',
            '20904.2','20904.2','20904.2','20904.2','20904.2','20904.2','20904.2','20904.2',
            20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,
            20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,20904.2,
            20904.2,20904.2,20904.2
            ]