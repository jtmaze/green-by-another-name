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


# %% 3.0 

highest_toa_sat_diff = combined.sort_values(by='abs_ls_ac_diff', ascending=False).head(5).copy()
highest_sr_sat_diff = combined.sort_values(by='abs_s2_ac_diff', ascending=False).head(5).copy()

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

# %%

combined_image_info = toa_image_info_dicts + sr_image_info_dicts

# %% Write some masks for images that agree decently well. Picked 3 randomly


# %% 3.0 Run area calculations

for d in toa_image_info_dicts:
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