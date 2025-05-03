# %% 1.0 Get the unique dates and rois from the image files
# 1.0 Get the unique dates and rois from the image files
import glob
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.img_data_fetching_functions import extract_unique
from functions.ac_pixel_regression_functions import make_ac_reflectance_summaries

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

toa_files = glob.glob('./data/toa_images/**/*tif')
sr_files = glob.glob('./data/sr_images/**/*.tif')
full_files = toa_files + sr_files
date_pattern = r'_date_(.*?)_roi'
roi_pattern = r'_roi_(.*?).tif'
resample_pattern = r'/reprojected_(.*?)_'

image_dates = extract_unique(full_files, date_pattern)
rois = extract_unique(full_files, roi_pattern)
resample_methods = extract_unique(full_files, resample_pattern)
resample_method = 'bilinear30'
satellites = ['Landsat8', 'Sentinel2']

# SECTION 2.0: Make regressions for PLD 60 meter buffered lake zone

image_info = {
    'level': None,
    'date': None,  # Dates will be iterated through
    'roi': None,   # ROIs will be iterated through
    'band_name': None,  # Bands will be specified
    'resample_method': None,
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': 60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0,
}

# SECTION 2.1: Green Band - Lake Zone (60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 2.2: NIR Band - Lake Zone (60m buffer)
image_info['band_name'] = 'NIR'
image_info['resample_method'] = resample_method

nir_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 2.3: NDWI Band - Lake Zone (60m buffer)
image_info['band_name'] = 'NDWI'
image_info['resample_method'] = resample_method
regression_params['outlier_frac'] = 0  # Don't omit any ndwi outliers, because they are not way off scale.

ndwi_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 2.4: Save the 60m Lake regression summaries to a CSV
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))
out_df.to_csv(f'./data/regression_summaries/AC_regression_summaries_60m_lake_{resample_method}_batch3.csv', index=False)

# SECTION 3.0: Green, NIR, NDWI over lakes with 0 meter buffer

image_info = {
    'level': None,
    'date': None,
    'roi': None,
    'band_name': None
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': 0,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0,
}

# SECTION 3.1: Green Band - Lake Zone (0m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 3.2: NIR Band - Lake Zone (0m buffer)
image_info['band_name'] = 'NIR'

nir_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 3.3: NDWI Band - Lake Zone (0m buffer)
image_info['band_name'] = 'NDWI'
image_info['resample_method'] = resample_method
regression_params['outlier_frac'] = 0

ndwi_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 3.4: Save the 0m Lake regression summaries to a CSV
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))
out_df.to_csv(f'./data/regression_summaries/AC_regression_summaries_0m_lake_{resample_method}_batch3.csv', index=False)

# SECTION 4.0: Green, NIR, NDWI over shoreline with -60 meter and +60 meter buffer

image_info = {
    'level': None,
    'date': None,
    'roi': None,
    'band_name': None
}

mask_params = {
    'zone': 'shoreline',
    'buffer_delim': -60,
    'buffer_delim_outer': 60,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0,
}

# SECTION 4.1: Green Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 4.2: NIR Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'NIR'

nir_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 4.3: NDWI Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'NDWI'
image_info['resample_method'] = resample_method
regression_params['outlier_frac'] = 0

ndwi_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 4.4: Save the -60m to +60m Shoreline regression summaries to a CSV
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))
out_df.to_csv(f'./data/regression_summaries/AC_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv', index=False)

# SECTION 5.0: Green, NIR, NDWI over land with a 60m buffer

image_info = {
    'level': None,
    'date': None,
    'roi': None,
    'band_name': None
}

mask_params = {
    'zone': 'land',
    'buffer_delim': 60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0,
}

# SECTION 5.1: Green Band - Land Zone (60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 5.2: NIR Band - Land Zone (60m buffer)
image_info['band_name'] = 'NIR'
nir_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 5.3: NDWI Band - Land Zone (60m buffer)
image_info['band_name'] = 'NDWI'
image_info['resample_method'] = resample_method
regression_params['outlier_frac'] = 0

ndwi_df = make_ac_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    satellites=satellites,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# SECTION 5.4: Save the 60m Land regression summaries to a CSV
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))
out_df.to_csv(f'./data/regression_summmaries/AC_regression_summaries_60m_land_{resample_method}.csv', index=False)

# %%
