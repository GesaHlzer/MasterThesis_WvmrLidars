# -*- coding: utf-8 -*-
"""
Created on Wed March 12nd, 2025

@author: Gesa Hölzer

Downloads radiosonde files from Wyoming (for years from 2018)

For this you need to define

1) the date ("start_date") or a range of dates ("start_date" & "end_date")
2) the station id 

The program creates (if not exsisting) a folder in your current directory 
".\data\raso\" and saves the csv-files there. 

"""

#Packages
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')

# Define (Example)_____________________________________________________________

# Specify the directory where you want to save the CSV files (target_dir)
main_dir = os.path.dirname(os.getcwd())  # parent dir of current working dir
station_id = '11120' # Station ID (Adjust according to your needs)

#Def. Date to download
start_date = datetime(2024, 10, 1) # Analysis day

# or date range            # Start and end dates
end_date = datetime(2024, 10, 23)


# Functions____________________________________________________________________

def delete_file_duplicates(raso_dir, target_date, station_id):
    """
    Compare and remove duplicate CSV files for a given datetime date.

    Parameters:
    - directory: Path to the folder containing CSV files.
    - target_date: Datetime object for the target date (e.g., datetime(2024, 8, 12)).
    - station_id: Station ID to filter (e.g., "11120").
    """

    # Step 1: Filter filenames based on the target date
    target_date_str = target_date.strftime("%Y%m%d")
    csv_files = [f for f in os.listdir(raso_dir) if f.endswith('.csv') and station_id in f]
    filtered_files = [f for f in csv_files if target_date_str in f]
    
    # Step 2: Compare the contents of CSV files
    unique_dataframes = []
    for file in filtered_files:
        file_path = os.path.join(raso_dir, file)
        df = pd.read_csv(file_path)

        # Check if this DataFrame matches any in the unique_dataframes list
        is_duplicate = any(df.equals(existing_df) for existing_df in unique_dataframes)
        
        if not is_duplicate:
            unique_dataframes.append(df)
        else:
            # If it's a duplicate, delete the file
            os.remove(file_path)
            print(f"Deleted duplicate file: {file}")
    
def download_csv_wyomingfrom2018(raso_dir, date, time_str):
    
    # Ensure the directory exists, if not, create it
    os.makedirs(raso_dir, exist_ok=True)

    # Format the date for the URL with the provided time
    date_str = date.strftime("%Y-%m-%d") + f" {time_str}"
    
    # Base URL (adjust if needed)
    base_url = "https://weather.uwyo.edu/cgi-bin/bufrraob.py"
    
    # Construct the URL
    url = f"{base_url}?datetime={date_str}&id={station_id}&type=TEXT:CSV"

    try:
        # Make the request to get the CSV data, bypassing SSL verification
        response = requests.get(url, verify=False)

        # If the response is successful (status code 200) and there is content
        if response.status_code == 200 and response.content:
            # Generate a filename based on the date and time
            filename = os.path.join(raso_dir, f"{date.strftime('%Y%m%d')}{time_str[0:2]}-{station_id}.csv")
            
            # Save the CSV file
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return True  # Return True if download is successful
        
        else:
            print(f"No data for date: {date_str}")
            return False  # Return False if data is missing
        
    except Exception as e:
        # Handle exceptions such as network errors
        print(f"Failed to download for date: {date_str}, Error: {str(e)}")
        return False  # Return False on failure

def save_csv(main_dir, station_id, start_date, end_date=None):
    
    raso_dir = os.path.join(main_dir, 'data', 'raso')
    
    #Download Wyoming from 2018 Nighttime 2-4  & Midday 10-13  
    times_0_5   = ["00:00:00", "01:00:00", "02:00:00", "03:00:00", "04:00:00", "05:00:00"]
    times_6_11  = ["06:00:00", "07:00:00", "08:00:00", "09:00:00", "10:00:00", "11:00:00"]
    times_12_17 = ["12:00:00", "13:00:00", "14:00:00", "15:00:00", "16:00:00", "17:00:00"]
    times_18_23 = ["18:00:00", "19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00"]
    
    if end_date == None:
        end_date = start_date
    else:
        end_date = end_date
        
    current_date = start_date
    while current_date <= end_date:
        #print(current_date.strftime('%Y-%m-%d'))
    
        for time_str in times_0_5: # Download Wyoming from 2018 Nighttime   
            success = download_csv_wyomingfrom2018(raso_dir, current_date, time_str)
            if success:
                  break  # Stop trying other times if download is successful
                  
        for time_str in times_6_11: #Download Wyoming from 2018 Morning
            success = download_csv_wyomingfrom2018(raso_dir, current_date, time_str)
            if success:
                  break 
        for time_str in times_12_17: #Download Wyoming from 2018 Midday
            success = download_csv_wyomingfrom2018(raso_dir, current_date, time_str)
            if success:
                  break 
    
        for time_str in times_18_23: #Download Wyoming from 2018 Evening
            success = download_csv_wyomingfrom2018(raso_dir, current_date, time_str)
            if success:
                  break   
        
        delete_file_duplicates(raso_dir, current_date, station_id)
        
        # Move to the next day
        current_date += timedelta(days=1)
        
def rename_sonde_files():
    
    
    # 1. Gather all files
    raso_dir    = Path(os.path.dirname(os.getcwd())) / "data" / "raso"
    station_id  = "11120"      # adjust if needed
    time_col    = "time"       # CSV column holding the timestamp

    csv_paths = sorted(raso_dir.glob("*.csv"))
    
    # 3. Process each file

    for old_path in csv_paths:
        try:
            # 3a. Read only the first data row (after header)
            df0 = pd.read_csv(
                old_path,
                parse_dates=[time_col],
                usecols=[time_col],
                nrows=1
            )
            # 3b. Extract the Python datetime
            t0 = df0[time_col].iloc[0].to_pydatetime()
        except Exception as e:
            print(f"  ✗ Skipping {old_path.name}: cannot read first timestamp ({e})")
            continue
    
        # 4. Build the new filename
        new_name = f"{t0.strftime('%Y%m%d%H')}-{station_id}.csv"
        new_path = old_path.with_name(new_name)
    
        # 5. Skip if already correctly named
        if old_path.name == new_name:
            print(f"  ✔ {old_path.name} is already correct")
            continue
    
        # 6. Handle potential name conflict
        if new_path.exists():
            print(f"  ✗ Cannot rename {old_path.name} → {new_name}: target exists")
            continue
    
        # 7. Perform the rename
        try:
            old_path.rename(new_path)
            print(f"  → Renamed {old_path.name} → {new_name}")
        except Exception as e:
            print(f"  ✗ Failed to rename {old_path.name}: {e}")

# def download_raso():
if end_date == None:      

    save_csv(main_dir, station_id, start_date)

elif end_date == start_date:  
    save_csv(main_dir, station_id, start_date)
    
else:
    save_csv(main_dir, station_id, start_date, end_date)

rename_sonde_files()

# Co
    
    
