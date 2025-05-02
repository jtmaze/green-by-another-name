# %% 1.0 Libraries and dirctories
import os
import sys

import pandas as pd
import pprint as pp
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.img_data_fetching_functions import extract_unique
from functions.img_water_area_calc_functions import make_area_thresholding_summaries

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')
random.seed(42)

area_dir = './data/lake_area_results'

# %% 2.0 Determine 10 rois/dates with the highest LS8 and Sentinel-2 TOA-SR difference
resample_method = 'bilinear30'

toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch3.csv')
toa_data = toa_data[
    toa_data['pld_plus_valid_frac'] >= 70
]
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch3.csv')
sr_data = sr_data[
    sr_data['pld_plus_valid_frac'] >= 70
]
print(len(toa_data))
print(len(sr_data))
# %%

cols_to_keep =['date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive',
               'buff_lake_s2_water_frac_adaptive']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_water_frac',
        'roi': 'roi_name'
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_water_frac',
        'roi': 'roi_name'
    }
).copy()

combined = pd.concat([toa_data, sr_data])

combined['abs_sat_diff'] = combined['ls_water_frac'] - combined['s2_water_frac']
combined['rel_sat_diff'] = combined['abs_sat_diff'] / combined['ls_water_frac'] * 100


# %% 3.0 Choose the 5 highest satellite discrepancies for toa and sr

highest_toa_sat_diff = combined[
    combined['level'] == 'toa'
].sort_values(by='abs_sat_diff', ascending=True).head(5).copy()

highest_sr_sat_diff = combined[
    combined['level'] == 'sr'
].sort_values(by='abs_sat_diff', ascending=True).head(5).copy()

toa_image_info_dicts = []

for idx, row in highest_toa_sat_diff.iterrows():
    image_info = {
        'level': None,
        'date': row['date'],
        'roi': row['roi_name'],
        'band_name': None,
        'resample_method': 'bilinear30'
    }
    toa_image_info_dicts.append(image_info)

sr_image_info_dicts = []
for idx, row in highest_sr_sat_diff.iterrows():
    image_info = {
        'level': None,
        'date': row['date'],
        'roi': row['roi_name'],
        'band_name': None,
        'resample_method': 'bilinear30'
    }
    sr_image_info_dicts.append(image_info)

toa_date_roi_set = {(d['date'], d['roi']) for d in toa_image_info_dicts}
sr_date_roi_set = {(d['date'], d['roi']) for d in sr_image_info_dicts}

# Find the intersection (common elements)
common_date_roi = toa_date_roi_set.intersection(sr_date_roi_set)

# Count the number of common elements
num_common = len(common_date_roi)

print(f"Number of high sat discrepancies with common date and ROI: {num_common}")

# If you want to see which combinations are common
print("\nCommon date-ROI combinations:")
for date, roi in common_date_roi:
    print(f"Date: {date}, ROI: {roi}")

# %% Randomly select 5 rows from the combined dataframe
random_samples = combined.sample(n=5, random_state=42)
random_image_info_dicts = []

for idx, row in random_samples.iterrows():
    image_info = {
        'level': None,
        'date': row['date'],
        'roi': row['roi_name'],
        'band_name': None,
        'resample_method': 'bilinear30'
    }
    random_image_info_dicts.append(image_info)

print(f"\nRandomly selected 5 roi/date pairs:")
for d in random_image_info_dicts:
    print(f"Date: {d['date']}, ROI: {d['roi']}")

combined_image_info = toa_image_info_dicts + sr_image_info_dicts + random_image_info_dicts


# %% 3.0 Run area calculations

for d in combined_image_info:
    print(d)
    rois = [d.get('roi')]
    image_dates = [d.get('date')]
    levels = ['sr', 'toa']
    _ = make_area_thresholding_summaries(
        d, 
        levels, 
        rois, 
        image_dates, 
        hist_return=False, 
        write_rasters=True
    )
    print('***********************')
    print('***********************')
    print('***********************')
# %%
