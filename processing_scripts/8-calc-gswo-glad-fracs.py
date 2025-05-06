# %% 

import os 
import re
import glob
import pandas as pd
import numpy as np
import rasterio as rio

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')
gswo_glad_dir = './data/gswo_glad_pld/'
glad_june_files = glob.glob(f'{gswo_glad_dir}*dataset_GLAD_month_june*.tif')
print(len(glad_june_files))

# %%

results = []

for f in glad_june_files:

    # Find the ROI
    roi_match = re.search(r"_roi_(.*?)\.tif", f)
    if roi_match:
        roi_name = roi_match.group(1)

    glad_aug_path = f'{gswo_glad_dir}/dataset_GLAD_month_aug_roi_{roi_name}.tif'
    gswo_june_path = f'{gswo_glad_dir}/dataset_GSWO_month_june_roi_{roi_name}.tif'
    gswo_aug_path = f'{gswo_glad_dir}/dataset_GSWO_month_aug_roi_{roi_name}.tif'
    roi_fp = f'./data/roi_shapes/rois/rasterized_{roi_name}_shape_res30.tif'

    with rio.open(f) as glad_june, \
            rio.open(glad_aug_path) as glad_aug, \
            rio.open(gswo_june_path) as gswo_june, \
            rio.open(gswo_aug_path) as gswo_aug, \
            rio.open(roi_fp) as ref:
        
        roi_shape = ref.read(1)
        total_roi_pix = np.sum(roi_shape == 1)

        glad_june_data = glad_june.read(1)
        glad_aug_data = glad_aug.read(1)
        gswo_june_data = gswo_june.read(1)
        gswo_aug_data = gswo_aug.read(1)


        total_lake_pix = np.sum(glad_june_data != -1)

        glad_june_wtr = glad_june_data > 80 
        glad_june_wtr_count = np.sum(glad_june_wtr)
        glad_aug_wtr = glad_aug_data > 80
        glad_aug_wtr_count = np.sum(glad_aug_wtr)
        gswo_june_wtr = gswo_june_data > 80
        gswo_june_wtr_count = np.sum(gswo_june_wtr)
        gswo_aug_wtr = gswo_aug_data > 80
        gswo_aug_wtr_count = np.sum(gswo_aug_wtr)

        result = {
            'roi': roi_name,
            'total_roi_pix': total_roi_pix,
            'glad_june_wtr_count': glad_june_wtr_count,
            'glad_aug_wtr_count': glad_aug_wtr_count,
            'gswo_june_wtr_count': gswo_june_wtr_count,
            'gswo_aug_wtr_count': gswo_aug_wtr_count,
            'total_lake_pix': total_lake_pix
        }

        results.append(result)

        # glad_seasonal =  glad_june_wtr & ~glad_aug_wtr 
        # gswo_seasonal = gswo_june_wtr & ~gswo_aug_wtr

# %% Results to dataframe

out_df = pd.DataFrame(results)
out_df.to_csv('./data/gswo_glad_areas.csv', index=False)

# %%
