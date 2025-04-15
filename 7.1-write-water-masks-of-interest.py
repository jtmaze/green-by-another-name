# %% 1.0 Libraries and dirctories
import glob
import pandas as pd
import pprint as pp

from img_data_fetching_functions import extract_unique
from img_water_area_calc_functions import make_area_thresholding_summaries

area_dir = './data/lake_area_results'

# %% 2.0 Determine 10 rois/dates with the highest LS8 and Sentinel-2 TOA-SR difference
resample_method = 'bilinear30'

resample_method = 'bilinear30'
toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch2.csv')
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch2.csv')

cols_to_keep =['date', 'roi', 'buff_lake_ls_water_frac_adaptive',
               'buff_lake_s2_water_frac_adaptive']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'toa_ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 'toa_s2_water_frac',
        'roi': 'roi_name'
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'sr_ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 'sr_s2_water_frac',
        'roi': 'roi_name'
    }
).copy()

combined = pd.merge(left=toa_data, right=sr_data, on=['date', 'roi_name'], how='inner')
combined['abs_ls_ac_diff'] = combined['toa_ls_water_frac'] - combined['sr_ls_water_frac']
combined['rel_ls_ac_diff'] = combined['abs_ls_ac_diff'] / combined['toa_ls_water_frac'] * 100
combined['abs_s2_ac_diff'] = combined['toa_s2_water_frac'] - combined['sr_s2_water_frac']
combined['rel_s2_ac_diff'] = combined['abs_s2_ac_diff'] / combined['toa_s2_water_frac'] * 100

# %% 3.0 

highest_ls_ac_diff = combined.sort_values(by='abs_ls_ac_diff', ascending=True).head(5).copy()
highest_s2_ac_diff = combined.sort_values(by='abs_s2_ac_diff', ascending=True).head(5).copy()

ls_image_info_dicts = []

for idx, row in highest_ls_ac_diff.iterrows():
    image_info = {
        'level': None,
        'date': row['date'],
        'roi': row['roi_name'],
        'band_name': None,
        'resample_method': 'bilinear30'
    }
    ls_image_info_dicts.append(image_info)

s2_image_info_dicts = []
for idx, row in highest_s2_ac_diff.iterrows():
    image_info = {
        'level': None,
        'date': row['date'],
        'roi': row['roi_name'],
        'band_name': None,
        'resample_method': 'bilinear30'
    }
    s2_image_info_dicts.append(image_info)

ls_date_roi_set = {(d['date'], d['roi']) for d in ls_image_info_dicts}
s2_date_roi_set = {(d['date'], d['roi']) for d in s2_image_info_dicts}

# Find the intersection (common elements)
common_date_roi = ls_date_roi_set.intersection(s2_date_roi_set)

# Count the number of common elements
num_common = len(common_date_roi)

print(f"Number of dictionaries with common date and ROI: {num_common}")

# If you want to see which combinations are common
print("\nCommon date-ROI combinations:")
for date, roi in common_date_roi:
    print(f"Date: {date}, ROI: {roi}")

# %%

combined_image_info = ls_image_info_dicts + s2_image_info_dicts

# %% Write some masks for images that agree decently well. Picked 3 randomly

dict1 = {
    'level': None,
    'date': '2019-05-24',
    'roi': 'AND_sub1',
    'band_name': None,
    'resample_method': 'bilinear30'
}

dict2 = {
    'level': None,
    'date': '2021-06-21',
    'roi': 'MRD_sub2',
    'band_name': None,
    'resample_method': 'bilinear30'
}

dict3 = {
    'level': None,
    'date': '2024-06-17',
    'roi': 'TUK_sub2',
    'band_name': None,
    'resample_method': 'bilinear30'
}

agree_img_info_dicts = [dict1, dict2, dict3]

# %% 3.0 Run area calculations

for d in agree_img_info_dicts:
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