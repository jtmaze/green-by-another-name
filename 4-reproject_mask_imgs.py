# %% 1.0 Libraries and file paths

import os
import glob
import re
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

from image_analysis_functions import extract_unique

level = 'toa' # must be 'sr' or 'toa'
res = 30 # 30 or 60 meters
download_dir = f'./data/{level}_image_downloads/'
roi_dir = './data/roi_shapes/rois/'

all_s2_files = glob.glob(f'{download_dir}/Sentinel2*.tif')
print(len(all_s2_files))
all_ls8_files = glob.glob(f'{download_dir}/Landsat8*.tif')
print(len(all_ls8_files))

roi_pattern = r'roi_(.*?).tif'
date_pattern = r'date_(.*?)_roi'
unique_rois = extract_unique(all_ls8_files, roi_pattern)
unique_dates = extract_unique(all_s2_files, date_pattern)

pairs = list(product(unique_rois, unique_dates))


# %% 2.0 

def check_for_pair_fp(
    pair_info: tuple,
    level: str
):
    roi = pair_info[0]
    date = pair_info[1]
    download_dir = f'./data/{level}_image_downloads/'

    ls8_fp = f'{download_dir}Landsat8_{level}_date_{date}_roi_{roi}.tif'
    s2_fp = f'{download_dir}Sentinel2_{level}_date_{date}_roi_{roi}.tif'

    ls8_exist = os.path.exists(ls8_fp)
    s2_exist = os.path.exists(s2_fp)

    if ls8_exist and s2_exist:
        return (s2_fp, ls8_fp)
    
def find_bounds_intersection(
    bounds1: dict,
    bounds2: dict,
):
    
    left = max(bounds1.left, bounds2.left)
    right = min(bounds1.right, bounds2.left)
    top = min(bounds1.top, bounds2.top)
    bottom = max(bounds1.bottom, bounds2.bottom)

    pass
    
def reproject_to_ref(
    img_fp: str,
    ref_fp: str,
    resample_method: str
):
    """
    Reads the images and or common masks and reprojects them to match the reference raster.
    Writes the data to the "temp" folder
    
    Parameters:
    -----------
    img_fp : str
        Path to the source image to reproject
    ref_fp : str
        Path to the reference image that defines the target projection
    resample_method : str
        Resampling method to use. Options: 'nearest', 'bilinear', 'cubic', 'average'
    """

    resamp_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }
    resamp = resamp_methods.get(resample_method)

    with rio.open(ref_fp) as ref, rio.open(img_fp) as src:
        
        out_meta = src.meta.copy()
        ref_meta = ref.meta
        out_data = np.zeros((src.count, ref.height, ref.width), dtype=src.dtypes[0])
        src_data = src.read()

        out_meta.update({
            'crs': ref_meta['crs'],
            'transform': ref_meta['transform'],
            'width': ref_meta['width'],
            'height': ref_meta['height']
        })

        reproject(
            source=src_data,
            destination=out_data,
            src_transform=src.transform, 
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=resamp
        )

        temp_dir = './data/temp/'
        basename = os.path.basename(img_fp)
        ref_res = round(ref.transform[0])
        out_fp = f'{temp_dir}/reprojected_{resample_method}{ref_res}_{basename}'

        with rio.open(out_fp, 'w', **out_meta) as dst:
            dst.write(out_data)

        print(f"Reprojected {out_fp}")

        return out_fp


# %% Functions to mask the images

def apply_cloud_river_masks(
    s2_temp_fp: str,
    ls8_temp_fp: str,
    cloud_mask_fp: str,
    level: str, 
):
    
    with rio.open(s2_temp_fp) as s2, rio.open(ls8_temp_fp) as ls8, rio.open(cloud_mask_fp) as mask:

        s2_data = s2.read()
        s2_meta = s2.meta
        ls8_data = ls8.read()
        ls8_meta = ls8.meta
        cloud_mask_data = mask.read(1)

        s2_valid = np.any(s2_data != 0, axis=0) # Checks for any bands not equal to zero
        ls8_valid = np.any(ls8_data != 0, axis=0)
        # Valid pixels for images and cloud mask
        valid_pixels_mask = s2_valid & ls8_valid & (cloud_mask_data == 0)

        s2_masked = s2_data.copy()
        for i in range(s2.count):
            s2_masked[i, ~valid_pixels_mask] = 0

        ls8_masked = ls8_data.copy()
        for i in range(ls8.count):
            ls8_masked[i, ~valid_pixels_mask] = 0


        out_dir = f'./data/{level}_images/'

        s2_basename = os.path.basename(s2_temp_fp)
        ls8_basename = os.path.basename(ls8_temp_fp)
        s2_out_fp = f'{out_dir}{s2_basename}'
        ls8_out_fp = f'{out_dir}{ls8_basename}'

        s2_meta.update({'nodata': 0})
        ls8_meta.update({'nodata': 0})

        with rio.open(s2_out_fp, 'w', **s2_meta) as dst_s2:
            dst_s2.write(s2_masked)

        with rio.open(ls8_out_fp, 'w', **ls8_meta) as dst_ls8:
            dst_ls8.write(ls8_masked)

    print("Masked some images")
    # os.remove(s2_temp_fp)
    # os.remove(ls8_temp_fp)
    # os.remove(cloud_mask_fp)







# %%

for p in pairs[0:10]:

    pair_paths = check_for_pair_fp(p, level)
    if pair_paths is not None:
        roi = extract_unique(pair_paths, roi_pattern)[0]
        date = extract_unique(pair_paths, date_pattern)[0]
        s2_fp = pair_paths[0]
        ls8_fp = pair_paths[1]
        ref_fp = f'./data/roi_shapes/rois/rasterized_{roi}_shape_res{res}.tif'
        mask_fp = f'./data/{level}_masks/CommonMask_date_{date}_roi_{roi}.tif'

        # Reproject the data to common reference grid write to temp folder
        s2_temp_fp = reproject_to_ref(
            s2_fp,
            ref_fp,
            'nearest'
        )

        ls8_temp_fp = reproject_to_ref(
            ls8_fp,
            ref_fp,
            'nearest'
        ) 

        cloud_temp_fp = reproject_to_ref(
            mask_fp,
            ref_fp, 
            'nearest'
        )

        # Apply the cloud mask to the images
        apply_cloud_river_masks(
            s2_temp_fp,
            ls8_temp_fp,
            cloud_temp_fp,
            level,
        )


    

# %%
