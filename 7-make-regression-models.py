# %% 1.0 Get the unique dates and rois from the image files

import glob
import pandas as pd

from image_analysis_functions import extract_unique
from image_analysis_functions import make_reflectance_summaries
from image_analysis_functions import make_otsu_area_summaries


toa_files = glob.glob('./data/toa_images/*tif')
sr_files = glob.glob('./data/sr_images/*.tif')
full_files = toa_files + sr_files
date_pattern = r'_date_(.*?)_roi'
roi_pattern = r'_roi_(.*?)_resampled'

image_dates = extract_unique(full_files, date_pattern)
rois = extract_unique(full_files, roi_pattern)
levels = ['sr', 'toa']

regression_summaries = []

# %% 2.0 Make regressions for PLD 60 meter buffered

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': 60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 50_000,
    'outlier_frac': 0.0005,
}

# %% 2.1 Green Band PLD 60 meter buffered
image_info['band_name'] = 'Green'

dates = image_dates[0:3]

df = make_reflectance_summaries(image_info=image_info, 
                                mask_params=mask_params, 
                                regression_params=regression_params,
                                levels=levels, 
                                rois=rois, 
                                dates=dates,
                                hist_return=True)

# %% 2.2 NIR Band PLD 60 meter buffered

image_info['band_name'] = 'NIR'

# %% 2.3 NDWI Band PLD 60 meter buffered

image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0 # Don't omit any ndwi outliers, because they are not way off scale.
regression_params['outlier_frac'] = 0.0005


# %% 2.4 Save the 60m Lake regression summaries to a csv

out_df = pd.DataFrame(regression_summaries)
out_df.head(25)

# %%
out_df = df[df['slope'] != 'No Image Data']
# Create mask for valid model outputs
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv('./data/test_60m_lake.csv', index=False)

# %%
out_df.to_csv('./data/regression_summaries_60m_lake.csv', index=False)

# %% 3.0 Make regression summaries for land (PLD 120 meter buffered)

# Update mask params
mask_params['zone'] = 'land'
mask_params['buffer_delim'] = 120


# %% 3.1 Green Band PLD 120 meter buffered

# %% 3.2 NIR Band PLD 120 meter buffered

# %% 3.3 NDWI Band PLD 120 meter buffered
out_df = make_otsu_area_summaries(image_info, levels, rois, image_dates, hist_return=True)
out_df = out_df[out_df['ls_s2_percent_diff'] != 'No Image Data']
out_df.to_csv('./data/area_data_v1.csv', index=False)

