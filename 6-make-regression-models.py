# %% 1.0 Get the unique dates and rois from the image files

import glob
import pandas as pd
import pprint as pp

from img_data_fetching_functions import extract_unique
from img_pixel_regression_functions import make_reflectance_summaries

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
levels = ['sr', 'toa']
regression_summaries = []

# %% 2.0 Make regressions for PLD 60 meter buffered

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None, # Bands will be specified
    'resample_method': None, #
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': 60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 5_000,
    'outlier_frac': 0.0005,
}

# %% 2.1 Green Band PLD 60 meter buffered
image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False,
)
print("Done")

# %% 2.2 NIR Band PLD 60 meter buffered

image_info['band_name'] = 'NIR'

nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False,
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
out_df.to_csv(f'./data/regression_summaries_60m_lake_{resample_method}.csv', index=False)


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
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
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
    hist_return=False
)

image_info['band_name'] = 'NDWI'
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

# %% 3.3 Write PLD 120m buffered regressions to a csv
out_df = pd.concat([green_df, nir_df, ndwi_df])
print(len(out_df))
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv(f'./data/regression_summaries_120m_land_{resample_method}.csv', index=False)


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
image_info['resample_method'] = resample_method
green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
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
    hist_return=False
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
print(len(out_df))
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv(f'./data/regression_summaries_neg60m_lake_{resample_method}.csv', index=False)

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
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
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
    hist_return=False
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
out_df.to_csv(f'./data/regression_summaries_shoreline_neg30-0_{resample_method}.csv', index=False)

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
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
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
    hist_return=False
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
out_df.to_csv(f'./data/regression_summaries_shoreline_0-plus30_{resample_method}.csv', index=False)

# %% 7.0 Green, NIR, NDWI over lakes with 0 meter buffer

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'lake',
    'buffer_delim': 0,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0.0005,
}
# 7.1 Green band PLD 0 meter buffered

image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
# 7.2 NIR band PLD 0 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
# 7.3 NDWI band PLD 0 meter buffered
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

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv(f'./data/regression_summaries_0m_lake_{resample_method}.csv', index=False)

# %% 8.0 Green, NIR, NDWI over shoreline with -60 meter and +60 meter

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'shoreline',
    'buffer_delim': -60,
    'buffer_delim_outer': 60,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0.0005,
}
# 8.1 Green band PLD -60 to 60 meter buffered

image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
# 8.2 NIR band PLD -60 to 60 meter buffered
image_info['band_name'] = 'NIR'
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# 8.3 NDWI band PLD -60 to 60 meter buffered
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

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv(f'./data/regression_summaries_shoreline_neg60-60_{resample_method}.csv', index=False)

# %% 9.0 Green, NIR, NDWI over land with a 60 buffer

image_info = {
    'level': None,
    'date': None, # Dates will be itterated through
    'roi': None, # ROIs will be itterated through
    'band_name': None # Bands will be specified
}

mask_params = {
    'zone': 'land',
    'buffer_delim': 60,
    'buffer_delim_outer': None,
}

regression_params = {
    'sample_size': 10_000,
    'outlier_frac': 0.0005,
}
# %% 8.1 Green band land 60 meter buffered

image_info['band_name'] = 'Green'
image_info['resample_method'] = resample_method

green_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)
# %% 8.2 NIR band land 60 meter buffered
image_info['resample_method'] = resample_method
image_info['band_name'] = 'NIR'
levels = ['toa']
nir_df = make_reflectance_summaries(
    image_info=image_info,
    mask_params=mask_params,
    regression_params=regression_params,
    levels=levels,
    rois=rois,
    dates=image_dates,
    hist_return=False
)

# %% 8.3 NDWI band land 60 meter buffered
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

out_df = pd.concat([green_df, nir_df, ndwi_df])
out_df = out_df[out_df['slope'] != 'No Image Data']
out_df = out_df[out_df['slope'] != 'Poor Quality Image Data']
out_df.to_csv(f'./data/regression_summaries_60m_land_{resample_method}.csv', index=False)