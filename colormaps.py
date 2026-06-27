# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 00:26:55 2025

@author: Gesa Hölzer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def cmap_wvmr_diff():
    """Diverging colormap matching the wvmr colormap aesthetic:
       cyan (negative) ← white (zero) → yellow-orange (positive)"""
    cmap = np.array([
        [  0, 210, 210],   # cyan  (most negative)
        [ 80, 160, 200],   # blue-ish
        [255, 255, 255],   # white (zero)
        [255, 220,  80],   # yellow
        [230, 130,  30],   # orange (most positive)
    ], dtype=np.float32)
    cmap /= 255.0

    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    return LinearSegmentedColormap.from_list(
        "wvmr_diff", list(zip(positions, cmap))
    )

def cmap_wvmr2():
    cmap_colors = [
    (0.000,0.345,0.345,0.336),
    (0.059,0.529,0.521,0.470),
    (0.118,0.678,0.678,0.514),
    (0.176,0.800,0.774,0.506),
    (0.235,0.859,0.828,0.484),
    (0.294,0.886,0.843,0.425),
    (0.353,0.918,0.806,0.404),
    (0.412,0.937,0.730,0.411),
    (0.471,0.906,0.626,0.462),
    (0.529,0.863,0.522,0.500),
    (0.588,0.851,0.441,0.528),
    (0.647,0.796,0.399,0.576),
    (0.706,0.729,0.376,0.661),
    (0.765,0.650,0.412,0.706),
    (0.824,0.563,0.469,0.749),
    (0.882,0.463,0.557,0.808),
    #(0.941,0.379,0.663,0.871),
    #(0.000,0.282,0.679,0.890),
    (0.921,0.379,0.663,0.871),
    (0.980,0.282,0.679,0.890),
    # --- added white cap ---
    (0.990, 0.328, 0.779, 1.999),  # fade start (optional)
    (1.000, 1.000, 1.000, 1.000),  # pure white
    ]

    # rgb(87, 87, 85)
    # rgb(134, 132, 119)
    # rgb(172, 172, 131)
    # rgb(204, 197, 129)
    # rgb(219, 211, 123)
    # rgb(225, 214, 108)
    # rgb(234, 205, 103)
    # rgb(238, 186, 104)
    # rgb(231, 159, 117)
    # rgb(220, 133, 127)
    # rgb(217, 112, 134)
    # rgb(202, 101, 146)
    # rgb(185, 95, 168)
    # rgb(165, 105, 180)
    # rgb(143, 119, 190)
    # rgb(117, 142, 206)
    # rgb(96, 169, 222)
    # rgb(71, 173, 226)
    positions = [c[0] for c in cmap_colors]
    rgb_values = [(c[1], c[2], c[3]) for c in cmap_colors]
     
    # LinearSegmentedColormap aus den Stützstellen erstellen
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "From PPLS website deriveved wvmr colormap setup",
        list(zip(positions, rgb_values)),
        N=256
    )
    return cmap

def cmap_wvmr3():
    """Custom colormap: white → yellow → purple → blue → cyan → black"""
    cmap = np.array([
        [255, 255, 255],   # white
        [255, 255,   0],   # yellow
        [128,  51, 153],   # purple
        [ 26,  77, 204],   # blue (slightly less dark)
        [  0, 230, 230],   # cyan (a bit more)
        [  0,   0,   0],   # black (small top fraction)
    ], dtype=np.float32)
    cmap /= 255.0
 
    # Equal spacing except black stays compressed at top
    positions = [0.0, 0.20, 0.50, 0.75, 0.94, 1.0]
 
    return LinearSegmentedColormap.from_list(
        "custom", list(zip(positions, cmap))
    )

def cmap_wvmr():
    """Custom colormap: white → yellow → purple → blue → cyan → black"""
    cmap = np.array([
        [255, 255, 255],   # white
        [255, 255,   0],   # yellow
        [128,  51, 153],   # purple
        [ 26,  77, 204],   # blue
        [  0, 230, 230],   # cyan
        [  0,   0,   0],   # black (small top fraction)
    ], dtype=np.float32)
    cmap /= 255.0
 
    # Equally spaced positions
    positions = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
 
    return LinearSegmentedColormap.from_list(
        "custom", list(zip(positions, cmap))
    )

def cmap_wvmr1():
    """Custom colormap: white → yellow → purple → blue → cyan → black"""
    cmap = np.array([
        [255, 255, 255],   # white
        [255, 255,   0],   # yellow
        [128,  51, 153],   # purple
        [ 26,  77, 204],   # blue
        [  0, 230, 230],   # cyan
        [  0,   0,   0],   # black (small top fraction)
    ], dtype=np.float32)
    cmap /= 255.0
 
    # Equally spaced positions
    positions = [0.0, 0.15, 0.45, 0.7, 0.93, 1.0]
 
    return LinearSegmentedColormap.from_list(
        "custom", list(zip(positions, cmap))
    )

def cmap_abs():
    colors = [ 
    (0.20, 0.15, 0.19),   # muted twilight purple-brown
    (0.35, 0.28, 0.22),   # warm dusty beige
    (0.78, 0.65, 0.50),   # soft sand
    (0.92, 0.80, 0.75),   # pale sand
    (1.00, 0.98, 0.95)    # near-white, twilight-like highlight
    
    ]
    cmap = LinearSegmentedColormap.from_list("dust", colors)
    return cmap.reversed()

def cmap_backscatter():
    """ Custom colormap for Backscatter vlues"""
    cmap = np.array([[234, 250, 250],
                     #[215, 253, 242],
                     #[206, 252, 213],
                     #[214, 240, 174],
                     [235, 218, 135],
                     [255, 188, 118],
                     [251, 158, 148],
                     [218, 136, 166],
                     [169, 120, 158],
                     [116, 102, 126],
                     [78, 78, 78]], dtype=np.float32)

    cmap /= 255.0 # Normalize the colormap to [0, 1] range

    return LinearSegmentedColormap.from_list("backscatter", cmap)

def cmap_windspeed():
    """ Custom colormap: UIBK wind speed """
    
    cmap = np.array([[255, 255, 255],
                    [255, 252, 203],
                    [224, 243, 139],
                    [171, 231, 131],
                    [109, 220, 136],
                    [0, 208, 149],
                    [0, 197, 165],
                    [0, 185, 180],
                    [0, 173, 193],
                    [0, 159, 204],
                    [0, 144, 212],
                    [55, 127, 216],
                    [112, 108, 216],
                    [145, 89, 211],
                    [168, 69, 201],
                    [183, 50, 188],
                    [192, 35, 173],
                    [195, 48, 93],
                    [210, 103, 73],
                    [246, 139, 69],
                    [255, 204, 79]], dtype=np.float32)

    cmap /= 255.0 # Normalize the colormap to [0, 1] range

    return LinearSegmentedColormap.from_list("windspeed", cmap)

def cmap_purplebrown40():
    """Custom colormap: purple-brown (40 steps)"""
    
    cmap = np.array([[0.192, 0.165, 0.337],
                    [0.231, 0.204, 0.392],
                    [0.275, 0.247, 0.447],
                    [0.318, 0.286, 0.506],
                    [0.361, 0.325, 0.569],
                    [0.404, 0.369, 0.627],
                    [0.447, 0.408, 0.690],
                    [0.490, 0.451, 0.753],
                    [0.533, 0.490, 0.820],
                    [0.580, 0.537, 0.859],
                    [0.624, 0.584, 0.882],
                    [0.663, 0.631, 0.906],
                    [0.706, 0.678, 0.925],
                    [0.749, 0.722, 0.945],
                    [0.788, 0.765, 0.965],
                    [0.827, 0.812, 0.976],
                    [0.867, 0.851, 0.992],
                    [0.902, 0.890, 1.000],
                    [0.937, 0.929, 1.000],
                    [0.965, 0.965, 0.992],
                    [0.992, 0.961, 0.945],
                    [0.992, 0.925, 0.890],
                    [0.984, 0.882, 0.827],
                    [0.969, 0.843, 0.769],
                    [0.953, 0.796, 0.706],
                    [0.929, 0.753, 0.643],
                    [0.902, 0.706, 0.580],
                    [0.875, 0.659, 0.514],
                    [0.843, 0.612, 0.447],
                    [0.812, 0.565, 0.373],
                    [0.776, 0.518, 0.294],
                    [0.733, 0.475, 0.231],
                    [0.675, 0.435, 0.212],
                    [0.616, 0.396, 0.184],
                    [0.561, 0.357, 0.161],
                    [0.502, 0.314, 0.133],
                    [0.447, 0.275, 0.106],
                    [0.392, 0.235, 0.075],
                    [0.337, 0.196, 0.035],
                    [0.286, 0.161, 0.000]], dtype=np.float32)
    return LinearSegmentedColormap.from_list("purplebrown40", cmap)

def cmap_bluered40():
    """ Custom colormap: blue-red gradient (40 steps)"""
    
    cmap = np.array([[0.000, 0.184, 0.439],
                    [0.000, 0.227, 0.490],
                    [0.035, 0.275, 0.549],
                    [0.106, 0.318, 0.612],
                    [0.153, 0.365, 0.675],
                    [0.196, 0.408, 0.741],
                    [0.275, 0.455, 0.776],
                    [0.353, 0.498, 0.796],
                    [0.424, 0.545, 0.820],
                    [0.486, 0.588, 0.843],
                    [0.545, 0.635, 0.863],
                    [0.600, 0.678, 0.882],
                    [0.655, 0.718, 0.902],
                    [0.706, 0.761, 0.922],
                    [0.753, 0.800, 0.937],
                    [0.800, 0.835, 0.953],
                    [0.843, 0.875, 0.961],
                    [0.886, 0.906, 0.969],
                    [0.922, 0.933, 0.973],
                    [0.953, 0.957, 0.973],
                    [0.973, 0.953, 0.953],
                    [0.980, 0.922, 0.922],
                    [0.976, 0.886, 0.886],
                    [0.973, 0.843, 0.843],
                    [0.961, 0.800, 0.800],
                    [0.945, 0.757, 0.757],
                    [0.929, 0.706, 0.710],
                    [0.910, 0.659, 0.659],
                    [0.886, 0.608, 0.608],
                    [0.863, 0.557, 0.557],
                    [0.831, 0.502, 0.502],
                    [0.804, 0.447, 0.447],
                    [0.769, 0.392, 0.392],
                    [0.733, 0.333, 0.337],
                    [0.690, 0.282, 0.282],
                    [0.624, 0.243, 0.243],
                    [0.561, 0.204, 0.204],
                    [0.494, 0.165, 0.165],
                    [0.431, 0.122, 0.125],
                    [0.373, 0.078, 0.082]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("bluered40", cmap)

def cmap_bluered16():
    """ Custom colormap:  blue-red gradient (16 steps) """

    cmap = np.array([[0.000, 0.184, 0.439],
                    [0.082, 0.298, 0.588],
                    [0.204, 0.416, 0.753],
                    [0.408, 0.537, 0.816],
                    [0.565, 0.651, 0.871],
                    [0.706, 0.761, 0.922],
                    [0.827, 0.859, 0.957],
                    [0.929, 0.941, 0.973],
                    [0.980, 0.929, 0.929],
                    [0.969, 0.827, 0.827],
                    [0.929, 0.706, 0.710],
                    [0.871, 0.576, 0.576],
                    [0.796, 0.435, 0.439],
                    [0.702, 0.290, 0.290],
                    [0.533, 0.188, 0.188],
                    [0.373, 0.078, 0.082]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("bluered16", cmap)

def cmap_basic_seq_mhue_viridis():
    """ Custom colormap: basic viridis (sequential multi-hue) """

    cmap = np.array([[0.294, 0.000, 0.333],
                    [0.290, 0.039, 0.365],
                    [0.282, 0.110, 0.400],
                    [0.263, 0.161, 0.431],
                    [0.239, 0.208, 0.463],
                    [0.200, 0.251, 0.490],
                    [0.137, 0.294, 0.518],
                    [0.000, 0.333, 0.541],
                    [0.000, 0.376, 0.557],
                    [0.000, 0.412, 0.573],
                    [0.000, 0.451, 0.584],
                    [0.000, 0.490, 0.592],
                    [0.000, 0.525, 0.596],
                    [0.000, 0.561, 0.592],
                    [0.000, 0.592, 0.588],
                    [0.000, 0.624, 0.580],
                    [0.000, 0.655, 0.565],
                    [0.000, 0.686, 0.549],
                    [0.000, 0.714, 0.525],
                    [0.000, 0.737, 0.498],
                    [0.000, 0.765, 0.467],
                    [0.212, 0.784, 0.431],
                    [0.361, 0.808, 0.392],
                    [0.475, 0.827, 0.349],
                    [0.573, 0.843, 0.302],
                    [0.667, 0.859, 0.255],
                    [0.753, 0.871, 0.208],
                    [0.839, 0.882, 0.176],
                    [0.918, 0.890, 0.173],
                    [0.992, 0.890, 0.200]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("basic_seq_mhue_viridis", cmap)

def cmap_basic_seq_mhue_terrain2():
    """ Custom colormap: basic terrain2 (sequential multi-hue) """

    cmap = np.array([[0.008, 0.486, 0.118],
                    [0.161, 0.506, 0.129],
                    [0.235, 0.525, 0.145],
                    [0.298, 0.541, 0.165],
                    [0.349, 0.561, 0.184],
                    [0.400, 0.576, 0.204],
                    [0.443, 0.596, 0.231],
                    [0.486, 0.612, 0.255],
                    [0.529, 0.627, 0.282],
                    [0.569, 0.643, 0.310],
                    [0.604, 0.659, 0.337],
                    [0.639, 0.675, 0.365],
                    [0.675, 0.686, 0.392],
                    [0.706, 0.702, 0.424],
                    [0.737, 0.714, 0.455],
                    [0.765, 0.729, 0.482],
                    [0.792, 0.741, 0.514],
                    [0.820, 0.753, 0.545],
                    [0.843, 0.765, 0.573],
                    [0.867, 0.780, 0.604],
                    [0.886, 0.788, 0.635],
                    [0.902, 0.800, 0.663],
                    [0.918, 0.812, 0.690],
                    [0.933, 0.824, 0.718],
                    [0.941, 0.831, 0.745],
                    [0.949, 0.843, 0.773],
                    [0.949, 0.851, 0.800],
                    [0.949, 0.863, 0.824],
                    [0.937, 0.871, 0.847],
                    [0.886, 0.886, 0.886]], dtype=np.float32)
    
    return LinearSegmentedColormap.from_list("basic_seq_mhue_terrain2", cmap)

def cmap_basic_seq_mhue_plasma():
    """ Custom colormap: basic plasma (multi-hue sequential) """
    
    cmap = np.array([[0.000, 0.094, 0.537],
                    [0.110, 0.063, 0.537],
                    [0.243, 0.024, 0.537],
                    [0.329, 0.000, 0.545],
                    [0.396, 0.000, 0.549],
                    [0.455, 0.000, 0.553],
                    [0.506, 0.000, 0.553],
                    [0.557, 0.000, 0.553],
                    [0.600, 0.000, 0.549],
                    [0.643, 0.039, 0.541],
                    [0.682, 0.098, 0.529],
                    [0.718, 0.149, 0.518],
                    [0.753, 0.192, 0.502],
                    [0.780, 0.239, 0.482],
                    [0.808, 0.282, 0.455],
                    [0.835, 0.325, 0.427],
                    [0.859, 0.373, 0.392],
                    [0.878, 0.416, 0.353],
                    [0.894, 0.463, 0.306],
                    [0.906, 0.506, 0.251],
                    [0.918, 0.553, 0.176],
                    [0.925, 0.600, 0.047],
                    [0.929, 0.647, 0.000],
                    [0.929, 0.694, 0.000],
                    [0.925, 0.745, 0.000],
                    [0.918, 0.792, 0.000],
                    [0.906, 0.843, 0.000],
                    [0.890, 0.894, 0.000],
                    [0.871, 0.945, 0.118],
                    [0.855, 1.000, 0.278]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("basic_seq_mhue_plasma", cmap)

def cmap_adv_seq_mhue_inferno23plus1():
    """ Custom colormap: advanced inferno 23+1 (multi-hue sequential) """
    
    cmap = np.array([[0.000, 0.000, 0.000],
                    [0.000, 0.078, 0.098],
                    [0.000, 0.118, 0.165],
                    [0.000, 0.145, 0.227],
                    [0.000, 0.169, 0.290],
                    [0.110, 0.184, 0.353],
                    [0.227, 0.196, 0.416],
                    [0.333, 0.204, 0.467],
                    [0.431, 0.204, 0.506],
                    [0.529, 0.208, 0.533],
                    [0.620, 0.220, 0.549],
                    [0.698, 0.243, 0.545],
                    [0.773, 0.282, 0.529],
                    [0.835, 0.333, 0.494],
                    [0.886, 0.396, 0.435],
                    [0.925, 0.467, 0.349],
                    [0.941, 0.545, 0.247],
                    [0.937, 0.631, 0.196],
                    [0.925, 0.714, 0.208],
                    [0.910, 0.792, 0.286],
                    [0.890, 0.871, 0.400],
                    [0.875, 0.949, 0.522],
                    [0.867, 1.000, 0.659],
                    [0.950, 1.000, 0.850]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("adv_seq_mhue_inferno23plus1", cmap)

def cmap_adv_seq_mhue_inferno20():
    """ Custom colormap: advanced inferno 20 (multi-hue sequential) """
    
    cmap = np.array([[0.016, 0.016, 0.016],
                    [0.071, 0.063, 0.145],
                    [0.137, 0.082, 0.208],
                    [0.208, 0.102, 0.263],
                    [0.282, 0.118, 0.314],
                    [0.361, 0.133, 0.357],
                    [0.443, 0.153, 0.396],
                    [0.522, 0.176, 0.420],
                    [0.600, 0.208, 0.435],
                    [0.675, 0.247, 0.439],
                    [0.745, 0.298, 0.431],
                    [0.808, 0.353, 0.408],
                    [0.867, 0.416, 0.361],
                    [0.918, 0.486, 0.290],
                    [0.945, 0.569, 0.235],
                    [0.953, 0.651, 0.251],
                    [0.961, 0.737, 0.310],
                    [0.973, 0.820, 0.396],
                    [0.980, 0.906, 0.502],
                    [1.000, 0.996, 0.620]], dtype=np.float32)
    inv = cmap[::-1]

    return LinearSegmentedColormap.from_list("adv_seq_mhue_inferno20", inv)

def cmap_adv_seq_mhue_inferno():
    """ Custom colormap: advanced inferno (multi-hue sequential)  """
    
    cmap = np.array([[0.016, 0.016, 0.016],
                    [0.051, 0.047, 0.114],
                    [0.090, 0.071, 0.165],
                    [0.133, 0.082, 0.208],
                    [0.180, 0.094, 0.243],
                    [0.227, 0.106, 0.278],
                    [0.278, 0.118, 0.310],
                    [0.329, 0.125, 0.341],
                    [0.380, 0.137, 0.369],
                    [0.435, 0.149, 0.392],
                    [0.486, 0.165, 0.412],
                    [0.537, 0.180, 0.424],
                    [0.588, 0.204, 0.435],
                    [0.639, 0.227, 0.439],
                    [0.686, 0.255, 0.439],
                    [0.733, 0.290, 0.431],
                    [0.776, 0.325, 0.420],
                    [0.816, 0.361, 0.400],
                    [0.855, 0.404, 0.373],
                    [0.890, 0.447, 0.333],
                    [0.922, 0.494, 0.278],
                    [0.941, 0.545, 0.239],
                    [0.949, 0.604, 0.235],
                    [0.953, 0.659, 0.255],
                    [0.961, 0.714, 0.290],
                    [0.965, 0.769, 0.341],
                    [0.973, 0.824, 0.400],
                    [0.976, 0.878, 0.467],
                    [0.988, 0.933, 0.541],
                    [1.000, 0.996, 0.620]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("adv_seq_mhue_inferno", cmap)

def cmap_adv_seq_mhue_greens():
    """ Custom colormap: advanced greens (multi-hue sequential) """
    
    cmap = np.array([[0.000, 0.275, 0.086],
                    [0.000, 0.314, 0.110],
                    [0.020, 0.349, 0.133],
                    [0.063, 0.384, 0.153],
                    [0.098, 0.420, 0.173],
                    [0.125, 0.459, 0.192],
                    [0.157, 0.494, 0.212],
                    [0.184, 0.529, 0.227],
                    [0.212, 0.565, 0.243],
                    [0.235, 0.600, 0.255],
                    [0.263, 0.635, 0.271],
                    [0.322, 0.663, 0.314],
                    [0.380, 0.690, 0.365],
                    [0.431, 0.718, 0.412],
                    [0.482, 0.741, 0.455],
                    [0.529, 0.769, 0.498],
                    [0.573, 0.792, 0.541],
                    [0.616, 0.816, 0.580],
                    [0.655, 0.835, 0.620],
                    [0.694, 0.859, 0.659],
                    [0.729, 0.878, 0.694],
                    [0.765, 0.894, 0.729],
                    [0.800, 0.914, 0.765],
                    [0.831, 0.929, 0.796],
                    [0.859, 0.945, 0.831],
                    [0.886, 0.957, 0.859],
                    [0.910, 0.969, 0.886],
                    [0.933, 0.976, 0.914],
                    [0.953, 0.980, 0.937],
                    [0.965, 0.984, 0.957]], dtype=np.float32)
    


    return LinearSegmentedColormap.from_list("adv_seq_mhue_greens", cmap)

def cmap_adv_div_green_brown():
    """ Custom colormap: advanced green-brown (diverging) """
    
    cmap = np.array([[0.000, 0.278, 0.004],
                    [0.020, 0.345, 0.082],
                    [0.094, 0.412, 0.141],
                    [0.149, 0.478, 0.188],
                    [0.196, 0.545, 0.235],
                    [0.247, 0.608, 0.286],
                    [0.349, 0.663, 0.380],
                    [0.439, 0.718, 0.463],
                    [0.525, 0.765, 0.541],
                    [0.604, 0.812, 0.620],
                    [0.678, 0.855, 0.690],
                    [0.749, 0.894, 0.761],
                    [0.816, 0.925, 0.824],
                    [0.878, 0.953, 0.882],
                    [0.937, 0.969, 0.937],
                    [0.976, 0.957, 0.933],
                    [0.976, 0.925, 0.871],
                    [0.961, 0.886, 0.804],
                    [0.937, 0.843, 0.733],
                    [0.906, 0.792, 0.655],
                    [0.871, 0.741, 0.576],
                    [0.831, 0.686, 0.490],
                    [0.784, 0.627, 0.400],
                    [0.737, 0.569, 0.294],
                    [0.686, 0.510, 0.161],
                    [0.616, 0.451, 0.090],
                    [0.541, 0.392, 0.024],
                    [0.467, 0.333, 0.000],
                    [0.396, 0.271, 0.000],
                    [0.325, 0.212, 0.000]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("adv_div_green_brown", cmap)

def cmap_adv_div_brown_green():
    """ Custom colormap: advanced green-brown (diverging) """
     
    cmap = np.array([[0.325, 0.212, 0.000],
                    [0.396, 0.271, 0.000],
                    [0.467, 0.333, 0.000],  
                    [0.541, 0.392, 0.024],
                    [0.616, 0.451, 0.090],
                    [0.686, 0.510, 0.161],
                    [0.737, 0.569, 0.294],
                    [0.784, 0.627, 0.400],
                    [0.831, 0.686, 0.490],
                    [0.871, 0.741, 0.576],
                    [0.906, 0.792, 0.655],
                    [0.937, 0.843, 0.733],
                    [0.961, 0.886, 0.804],
                    [0.976, 0.925, 0.871],
                    [0.976, 0.957, 0.933],
                    [0.937, 0.969, 0.937],
                    [0.878, 0.953, 0.882],
                    [0.816, 0.925, 0.824],
                    [0.749, 0.894, 0.761],
                    [0.678, 0.855, 0.690],
                    [0.604, 0.812, 0.620],
                    [0.525, 0.765, 0.541],
                    [0.439, 0.718, 0.463],
                    [0.349, 0.663, 0.380],
                    [0.247, 0.608, 0.286],
                    [0.196, 0.545, 0.235],
                    [0.149, 0.478, 0.188],
                    [0.094, 0.412, 0.141],
                    [0.020, 0.345, 0.082],
                    [0.000, 0.278, 0.004]], dtype=np.float32)

    return LinearSegmentedColormap.from_list("adv_div_brown_green", cmap)

if __name__ == "__main__":
    # # Example usage
    # cmap = cmap_windspeed()
    # cmap = cmap_purplebrown40()
    # cmap = cmap_bluered40()
    # cmap = cmap_bluered16()
    # cmap = cmap_basic_seq_mhue_viridis()
    # cmap = cmap_basic_seq_mhue_terrain2()
    # cmap = cmap_basic_seq_mhue_plasma()
    # cmap = cmap_adv_seq_mhue_inferno23plus1()
    # cmap = cmap_adv_seq_mhue_inferno20()
    # cmap = cmap_adv_seq_mhue_inferno()
    # cmap = cmap_adv_seq_mhue_greens()
    # cmap = cmap_adv_div_green_brown()
    # cmap = cmap_backscatter()
    # cmap =  cmap_adv_div_brown_green()
    # cmap = cmap_ppls_wvmr()
    cmap = cmap_wvmr()
    
    
    # --- Beispielplot mit Colorbar ---
    fig, ax = plt.subplots(figsize=(8, 5))
     
    x = np.linspace(0, 2 * np.pi, 300)
    y = np.linspace(0, 2 * np.pi, 300)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
     
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wert")
    ax.set_title("Beispiel mit custom colormap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
     
    plt.tight_layout()
    plt.show()