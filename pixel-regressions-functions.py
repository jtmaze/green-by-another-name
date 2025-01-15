# %% 1.0

import random
from typing import Optional
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling
import numpy as np
import pandas as pd

from scipy import stats
import matplotlib.pyplot as plt

random.seed(20)

# %% 

###############################################
### Helper Functions
###############################################
def read_band_by_description(raster_path: str, description: str, image_window_params: Optional[dict]):
    """
    Returns an array from a specific band description (not index) in a raster file. 
    Optional argument to crop the band to a window.
    """
    
    data = None
    with rio.open(raster_path) as src:
        # Get a description list
        desc_list = src.descriptions
        # Read the data
        for idx, desc in enumerate(desc_list, start=1):
            if desc == description and image_window_params is None:
                data = src.read(idx)
                return data
            elif desc == description and image_window_params is not None:
                window = from_bounds(*image_window_params['bounds'], 
                                     image_window_params['transform']
                        )
                data = src.read(idx,
                                window=window, 
                                out_shape=image_window_params['shape'],
                                resampling=Resampling.nearest)
                return data
     
        if data is None:
            print("Error, could not find a band description to match your target band")
            return None

###############################################
### 1.0 Functions to read data
###############################################

def rio_get_data_arrays(ls_path: str, s2_path: str, band_name: str):
    """
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands
    """

    ls_data = read_band_by_description(ls_path, band_name, image_window_params=None)
    s2_data = read_band_by_description(s2_path, band_name, image_window_params=None)

    # Get the data's (LandSat's) bounds and transform as a window for the PLD mask
    with rio.open(ls_path) as src: 
        ls_bounds = src.bounds
        ls_transform = src.transform
        ls_shape = src.shape

    image_window_params = {
        'bounds': ls_bounds,
        'transform': ls_transform,
        'shape': ls_shape
    }
    
    # The data should have the same shapes to broadcast together.
    if ls_data.shape != s2_data.shape:
        print('The Sentinel-2 and Landsat8 data shapes are different, cannot compare')

    return ls_data, s2_data, image_window_params
    

def make_measure_mask(pld_path: str, 
                     image_window_params: dict, 
                     zone: str,
                     buffer_delim: int,
                     buffer_delim_outer: Optional[int]):
    """
    Returns a binary mask denoting which pixels to compare across images.
    Determined from PLD attributes to define lakes, land, and shoreline at different buffer values.    
    """
    desc = f'buffered_{buffer_delim}m'
    pld_mask = read_band_by_description(pld_path, desc, image_window_params)

    measure_mask = np.zeros_like(pld_mask)
    if zone == 'lake':
        measure_mask = np.where(pld_mask == 1, 1, measure_mask)
    elif zone == 'land':
        measure_mask = np.where(pld_mask == 1, measure_mask, 1)
    elif zone == 'shoreline':
        outer_desc = f'buffered_{buffer_delim}m'
        outer_mask = read_band_by_description(pld_path, outer_desc, image_window_params)
        measure_mask = np.where(pld_mask == 0, 1, measure_mask)
        measure_mask = np.where(outer_mask == 1, measure_mask, 0)

    return measure_mask

# %% 

test_fp_s2 = './data/sr_images/Sentinel2-sr_date_2021-07-01_roi_YKF_sub1_resampled_bilinear30.tif'
test_fp_ls8 = './data/sr_images/Sentinel2-sr_date_2021-07-01_roi_YKF_sub1_resampled_bilinear30.tif'
pld_path = './data/pld_rasterized/YKF_sub1_lake_masks.tif'

ls_data, s2_data, image_window_params = rio_get_data_arrays(test_fp_ls8, test_fp_s2, band_name='Green')
measure_mask = make_measure_mask(pld_path, image_window_params, zone='shoreline', buffer_delim=0, buffer_delim_outer=60)

# %%
