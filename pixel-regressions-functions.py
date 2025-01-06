# %% 1.0

import random
import rasterio as rio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd

from scipy import stats
import matplotlib.pyplot as plt

random.seed(20)

# %% 2.0 

def match_img_bands(ls_path: str, s2_path: str, band_name: str):
    """
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands
    """
    if band_name == 'Green':
        s2_band_idx = 1 and ls_band_idx = 1
    elif band_name == 'NIR':
        s2_band_idx = 1 and ls_band_idx = 2
    else:
        print('Make a better data structure for band names')

    with rio.open(ls_path) as src_ls, rio.open(s2_path) as src_s2:
        ls_meta = src_ls.meta
        s2_meta = src_s2.meta
        ls_array = src_ls.read()
        s2_array = src_s2.read()

    return ls_array, s2_array

def apply_river_mask(dataset)

    

