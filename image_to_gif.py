# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:45:47 2025

@author: alleh
"""
from PIL import Image, ImageDraw
import os

folder_paths = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\Lidar_Comparison\diff_timeseries_dt10s_rl1"
image_paths = [file for file in os.listdir(folder_paths) if file.endswith(".png")]

# Bilder laden
images = [Image.open(os.path.join(folder_paths, path)) for path in image_paths]
#pillow_imagedraw.gif
images[0].save(os.path.join(folder_paths,'pillow_imagedraw_from23.gif'),
               save_all = True, append_images = images[1:], 
               optimize = False, duration = 2000) # 1000 ms = 1 Sekunde
