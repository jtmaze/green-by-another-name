# %% 1.0 Libraries and dirctories
import glob
import pandas as pd
import pprint as pp

from img_data_fetching_functions import extract_unique
from img_water_area_calc_functions import make_area_thresholding_summaries


# Specify the level and resample method
level = 'toa'
levels = [level]
resample_method = 'bilinear30'
rois = ['YKF_sub1']
image_dates = ['2024-06-21']

# %% 2.0 Dictionaries to hold image information

image_info = {
    'level': None, # levels will be iterated though, keep at toa or sr for smaller output data chunks. 
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None, # Bands will be specified
    'resample_method': None, #
}
# %% 3.0 Run area calculations

image_info['resample_method'] = resample_method

out_df = make_area_thresholding_summaries(
    image_info, 
    levels, 
    rois, 
    image_dates, 
    hist_return=False, 
    write_rasters=True
)