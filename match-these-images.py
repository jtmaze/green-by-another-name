# %% 1.0 Libaries and directories

import re
import os
import pandas as pd
import rasterio as rio

from rasterio.warp import reproject, calculate_default_transform, Resampling
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
    
def calc_union_bounds(
        src1: rio.DatasetReader,
        src2: rio.DatasetReader
):
    """
    Finds the minimum set of bounds from two raster images
    NOTE: CRS should be UTM
    """
    left = min(src1.bounds.left, src2.bounds.left)
    bottom = min(src1.bounds.bottom, src2.bounds.bottom)
    right = max(src1.bounds.right, src2.bounds.right)
    top = max(src1.bounds.top, src2.bounds.top)

    if (left >= right) or (bottom >= top):
        raise ValueError("No overlap on the rasters")
    
    inter_bounds = {
        "left": left,
        "bottom": bottom,
        "right": right, 
        "top": top
    }

    return inter_bounds


def reproj_to_ref(
    in_path: str,
    out_path: str,
    ref_meta: dict
):
    with rio.open(in_path) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': ref_meta['crs'],
            'transform': ref_meta['transform'],
            'width': ref_meta['width'],
            'height': ref_meta['height']
        })

        with rio.open(out_path, 'w', **out_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i), 
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_meta['transform'],
                    dst_crs=ref_meta['crs'],
                    resampling=Resampling.bilinear
                )

    print(f"Reprojected to {out_path}")
    
def match_imgs_to_ref(
    s2_path: str, 
    ls8_path: str,
    ref_path: str
):
    
    with rio.open(ref_path) as ref:
        ref_meta = {
            'crs': ref.crs,
            'transform': ref.transform,
            'width': ref.width,
            'height': ref.height
        }

    s2_basename = os.path.basename(s2_path)
    s2_out_path = os.path.join('./data/temp/', s2_basename)
    reproj_to_ref(s2_path, s2_out_path, ref_meta)

    ls8_basename = os.path.basename(ls8_path)
    ls8_out_path = os.path.join('./data/temp/', ls8_basename)
    reproj_to_ref(ls8_path, ls8_out_path, ref_meta)

    

    

# %%

image_pairs_test = image_pairs.iloc[0:5]

# %% 

for idx, row in image_pairs_test.iterrows():

    s2_fp = row['s2_fp']
    roi = extract_roi(s2_fp)
    res = extract_res(s2_fp)
    ls8_fp = row['ls8_fp']

    ref_path = f'./data/roi_shapes/rois/rasterized_{roi}_shape_res{res}.tif'

    match_imgs_to_ref(
        s2_path=s2_fp,
        ls8_path=ls8_fp,
        ref_path=ref_path
    )







# %%
print(mis_match)
# %%
