# %% 1.0 Get the unique dates and rois from the image files

import glob
import pandas as pd

from image_analysis_functions import extract_unique
from image_analysis_functions import regress_image_pairs
from image_analysis_functions import otsu_image_wtr_area


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
    'outlier_frac': 0.02,
}

# %% 2.1 Green Band PLD 60 meter buffered

image_info['band_name'] = 'Green'

for level in levels:
    image_info['level'] = level
    for roi in rois:
        image_info['roi'] = roi
        for date in image_dates:
            image_info['date'] = date

            regression_summary = regress_image_pairs(image_info, mask_params, regression_params, hist_return=True)
            regression_summaries.append(regression_summary)

# %% 2.2 NIR Band PLD 60 meter buffered

image_info['band_name'] = 'NIR'

for level in levels:
    image_info['level'] = level
    for roi in rois:
        image_info['roi'] = roi
        for date in image_dates:
            image_info['date'] = date

            regression_summary = regress_image_pairs(image_info, mask_params, regression_params, hist_return=True)
            regression_summaries.append(regression_summary)

# %% 2.3 NDWI Band PLD 60 meter buffered

image_info['band_name'] = 'NDWI'

for level in levels:
    image_info['level'] = level
    for roi in rois:
        image_info['roi'] = roi
        for date in image_dates:
            image_info['date'] = date

            regression_summary = regress_image_pairs(image_info, mask_params, regression_params, hist_return=False)
            regression_summaries.append(regression_summary)


# %%

out_df = pd.DataFrame(regression_summaries)
out_df = out_df[out_df['model_domain'] == 'No Image Data']
out_df.to_csv('./data/regression_summaries.csv', index=False)

# %%

area_summaries = []

for level in levels:
    image_info['level'] = level
    for roi in rois:
        image_info['roi'] = roi
        for date in image_dates:
            image_info['date'] = date

            otsu_items = otsu_image_wtr_area(image_info, write_mask=False, hist_return=True)
            if otsu_items['ls_threshold'] == 'No Image Data' or otsu_items['ls_threshold'] == 'Poor Quality Image':
                ls_s2_percent_diff = 'No Image Data'
            else:
                ls_s2_percent_diff = (
                    (otsu_items['ls_water_frac'] - otsu_items['s2_water_frac']) 
                    / otsu_items['ls_water_frac']
                ) * 100

            summary = {
                'date': date,
                'roi': roi,
                'level': level,
                'otsu_items': otsu_items,
                'ls_s2_percent_diff': ls_s2_percent_diff
            }

            area_summaries.append(summary)

# %%

out_df = pd.DataFrame(area_summaries)
out_df = out_df[out_df['ls_s2_percent_diff'] != 'No Image Data']
