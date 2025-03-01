# %% 1.0 Libaries and directories

import re
import pandas as pd
import rasterio as rio
from rasterio.warp import reproject
import numpy as np

level = 'toa'
image_pairs = pd.read_csv(f'./data/{level}_img_exports_to_use.csv')

"""
Steps:

1. Read the best Sentinel-2 and Landsat8 image dates
2. Reproject them to the ROI mask
4. Find the valid overlap % for both images. 
- Don't write if valid overlap is too low
3. Apply the band description for export
"""

band_desc = {
    1: 'Blue',
    2: 'Green',
    3: 'Red',
    4: 'NIR'
}
in_dir = f'./data/{level}_image_downloads'
out_dir = f'./data/{level}_images/'
# %% 2.0

def extract_roi(file_path):
    """finds the roi name in a file path"""
    roi_pattern = r'roi_(.*?)_resampled'
    match = re.search(roi_pattern, file_path)
    if match:
        return match.group(1)
    else:
        return None

def extract_res(file_path):
    """finds the resolution from the file path"""
    res_pattern = r'(\d{2})_idx'
    match = re.search(res_pattern, file_path)
    if match:
        return match.group(1)
    else:
        return None


def read_imgs_ls8_ref(
        s2_path: str, 
        ls8_path: str,
        ref_path: str,
):
    
    with rio.open(s2_path) as s2_src, rio.open(ls8_path) as ls8_scr:
        s2_trans = s2_src.transform
    


    

# %%
mis_match = 0

for idx, row in image_pairs.iterrows():

    s2_fp = row['s2_fp']
    roi = extract_roi(s2_fp)
    res = extract_res(s2_fp)
    ls8_fp = row['ls8_fp']
    print(res)
    print(roi)

    ref_path = f'./data/roi_shapes/rois/rasterized{roi}_shape_res{res}.tif'

    result = read_imgs(s2_path=s2_fp, ls8_path=ls8_fp, ref_path=ref_path)
    if result == 'fml':
        mis_match += 1



# %%
print(mis_match)
# %%
