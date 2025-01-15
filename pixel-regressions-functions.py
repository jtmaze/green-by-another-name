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

###############################################
### Helper Functions
###############################################

def read_band_by_description(raster_path: str, description: str):
    with rio.open(raster_path) as src:
        desc_list = src.descriptions

    for idx, desc in enumerate(desc_list, start=1):
        if desc == description:
            return src.read(idx)


def rio_get_data_arrays(ls_path: str, s2_path: str, band_name: str):
    """
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands
    """

    with rio.open(ls_path) as src_ls, rio.open(s2_path) as src_s2:
        ls_meta = src_ls.meta
        s2_meta = src_s2.meta
        ls_array = src_ls.read()
        s2_array = src_s2.read()

    return ls_array, s2_array

def apply_river_mask(dataset: np.array, roi_name: str):
    """
    Masks the rivers out of a dataset before performing an analysis
    """

    

