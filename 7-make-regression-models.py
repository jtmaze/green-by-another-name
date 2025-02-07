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

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
print("Done")
# 2.2 NIR Band PLD 60 meter buffered

image_info['band_name'] = 'NIR'

nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
print("Done")
# 2.3 NDWI Band PLD 60 meter buffered

image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0 # Don't omit any ndwi outliers, because they are not way off scale.

ndwi_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
print("Done")

# %% 2.4 Save the 60m Lake regression summaries to a csv

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
print(len(out_df))
# %%
out_df.to_csv('./data/regression_summaries_60m_lake.csv', index=False)


# %% 3.0 Make regression summaries for land (PLD 120 meter buffered)

image_info = {
    'level': None, # Levels will be itterated through
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'land',
    'buffer_delim': 120,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 50_000,
    'outlier_frac': 0.0005,
}

# 3.1 Green Band PLD 120 meter buffered

image_info['band_name'] = 'Green'
green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)

# 3.2 NIR Band PLD 120 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)

print("Done")

# %% 3.3 Write the land Green and NIR bands to a csv
out_df = pd.concat([green_df, nir_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv('./data/regression_summaries_120m_land.csv', index=False)


# %% 4.0 Green, NIR, NDWI on conservative lakes -60 buffer
image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': -60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 15_000,
    'outlier_frac': 0.0005,
}

# 4.1 Green band PLD -60 meter buffered

image_info['band_name'] = 'Green'
green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 4.2 NIR band PLD -60 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 4.3 NDWI band PLD -60 meter buffered
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0 # Don't omit any ndwi outliers, because they are not way off scale.
ndwi_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

print("Done")
# %% 4.4 Write the conservative -60m lake Green, NIR, NDWI bands to a csv
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv('./data/regression_summaries_neg60m_lake.csv', index=False)

# %% 5.0 Green, NIR, NDWI on shorelines -30m to 0m
image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'shoreline',
    'buffer_delim': -30,
    'buffer_delim_outer': 0,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0.0005,
}
# 5.1 Green band PLD -30 to 0 meter buffered
image_info['band_name'] = 'Green'
green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 5.2 NIR band PLD -30 to 0 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 5.3 NDWI band PLD -30 to 0 meter buffered
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0 # Don't omit any ndwi outliers, because they are not way off scale.
ndwi_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
print('Done')
# %% 5.1 Write the shoreline 0-neg30m Green, NIR, NDWI bands to a csv
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv('./data/regression_summaries_shoreline_neg30-0.csv', index=False)

# %% 6.0 Green, NIR, NDWI on shorelines 0 to +30m buffer

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'shoreline',
    'buffer_delim': 0,
    'buffer_delim_outer': 30,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0.0005,
}
# 6.1 Green band PLD 0 to 30 meter buffered
image_info['band_name'] = 'Green'
green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 6.2 NIR band PLD 0 to 30 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=True
)
# 6.3 NDWI band PLD 0 to 30 meter buffered
image_info['band_name'] = 'NDWI'
regression_params['outlier_frac'] = 0 # Don't omit any ndwi outliers, because they are not way off scale.
ndwi_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# %% 6.4 Write the shoreline 0-30m Green, NIR, NDWI bands to a csv
out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv('./data/regression_summaries_shoreline_0-plus30.csv', index=False)

# %% 4.0 Calculate the Area by Otsu thresholding images

out_df = make_otsu_area_summaries(image_info, levels, rois, image_dates, hist_return=True)
out_df = out_df[out_df['ls_s2_percent_diff'] != 'No Image Data']
out_df.to_csv('./data/area_data_v1.csv', index=False)

