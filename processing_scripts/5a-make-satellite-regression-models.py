# %% 1.0 Get the unique dates and rois from the image files
import glob
import sys
import os
import pandas as pd
import pprint as pp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.img_data_fetching_functions import extract_unique
from functions.satellite_pixel_regression_functions import make_satellite_reflectance_summaries

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
resample_method = 'bilinear30'  # 'nearest', 'bilinear30', 'cubic30'
levels = ['sr', 'toa']

image_info = {
    'level': None,
    'date': None,  # Dates will be itterated through
    'roi': None,   # ROIs will be itterated through
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
# %%
# Green Band - Lake Zone (60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
print("Done")

# %%

# NIR Band - Lake Zone (60m buffer)
image_info['band_name'] = 'NIR'

nir_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False,
)
print("Done")
# %% 

# NDWI Band - Lake Zone (60m buffer)
image_info['band_name'] = 'NDWI'
image_info['resample_method'] = resample_method
regression_params['outlier_frac'] = 0  # Don't omit any ndwi outliers

ndwi_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
print("Done")

# %%

# Save the 60m Lake regression summaries to a CSV
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))

# out_df.to_csv(
#     f'./data/regression_summaries/sat_regression_summaries_60m_lake_{resample_method}_batch3.csv',
#     index=False
# )

image_info = {
    'level': None,
    'date': None,  # Dates will be itterated through
    'roi': None,   # ROIs will be itterated through
    'band_name': None  # Bands will be specified
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

# Green Band - Lake Zone (0m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NIR Band - Lake Zone (0m buffer)
image_info['band_name'] = 'NIR'
nir_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NDWI Band - Lake Zone (0m buffer)
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0
ndwi_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
# out_df.to_csv(
#     f'./data/regression_summaries/sat_regression_summaries_0m_lake_{resample_method}_batch3.csv',
#     index=False
# )
# %%

image_info = {
    'level': None,
    'date': None,  # Dates will be itterated through
    'roi': None,   # ROIs will be itterated through
    'band_name': None  # Bands will be specified
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
# %%
# Green Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
# %%
# NIR Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'NIR'
nir_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NDWI Band - Shoreline Zone (-60m to +60m buffer)
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0
ndwi_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# %%
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
# out_df.to_csv(
#     f'./data/regression_summaries/sat_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv',
#     index=False
# )

"""
Land outside of the 60m buffer
"""

image_info = {
    'level': None,
    'date': None,  # Dates will be itterated through
    'roi': None,   # ROIs will be itterated through
    'band_name': None  # Bands will be specified
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

# Green Band - Land Zone (60m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NIR Band - Land Zone (60m buffer)
image_info['resample_method'] = resample_method
image_info['band_name'] = 'NIR'
nir_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NDWI Band - Land Zone (60m buffer)
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0
ndwi_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
# out_df.to_csv(
#     f'./data/regression_summaries/sat_regression_summaries_60m_land_{resample_method}_batch3.csv',
#     index=False
# )

# %%

"""
Shoreline at finer scale inside the 30m-30m buffer
"""

image_info = {
    'level': None,
    'date': None,  # Dates will be itterated through
    'roi': None,   # ROIs will be itterated through
    'band_name': None  # Bands will be specified
}

mask_params = {
    'zone': 'shoreline',
    'buffer_delim': -30,
    'buffer_delim_outer': 30,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0,
}

# Green Band - Shoreline Zone (-30m to +30m buffer)
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method
green_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NIR Band - Shoreline Zone (-30m to +30m buffer)
image_info['band_name'] = 'NIR'
nir_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# NDWI Band - Shoreline Zone (-30m to +30m buffer)
image_info['band_name'] = 'NDWI'
ndwi_df = make_satellite_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
# out_df.to_csv(
#     f'./data/regression_summaries/sat_regression_summaries_shoreline_neg30-30_{resample_method}_batch3.csv',
#     index=False
# )

# %%
