# %% 1.0 Libraries and dirctories
import glob
import pandas as pd
import pprint as pp

from img_data_fetching_functions import extract_unique
from img_water_area_calc_functions import make_area_thresholding_summaries

toa_files = glob.glob('./data/toa_images/**/*tif')
sr_files = glob.glob('./data/sr_images/**/*.tif')
full_files = toa_files + sr_files
date_pattern = r'_date_(.*?)_roi'
roi_pattern = r'_roi_(.*?).tif'
resample_pattern = r'/reprojected_(.*?)_'

image_dates = extract_unique(full_files, date_pattern)
rois = extract_unique(full_files, roi_pattern)
resample_methods = extract_unique(full_files, resample_pattern)

# Specify the level and resample method
level = 'toa'
levels = [level]
resample_method = 'lanczos60'

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
    write_rasters=False
)

print("Area summaries finished")
# %% 4.0 Write the output to csv
out_df = out_df[out_df['ls_otsu_threshold'].notna()]
# NOTE: change the batch number if you add more ROIs to the study!
out_df.to_csv(f'./data/lake_area_results/{level}_resampled_{resample_method}_area_summaries_batch2.csv', index=False)
# %%