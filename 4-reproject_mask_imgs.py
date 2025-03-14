# %% 1.0 Libraries and file paths

import os
import glob
import re
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

from image_analysis_functions import extract_unique

level = 'sr' # must be 'sr' or 'toa'
res = 30 # 30 or 60 meters
resample_method = 'lanczos'

band_desc = {
    1: 'Blue',
    2: 'Green',
    3: 'Red', 
    4: 'NIR'
}

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

# %% 2.0 Functions

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
        'cubic': Resampling.cubic,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average
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

        #print(f"Reprojected {out_fp}")

        return out_fp

def apply_cloud_river_masks(
    s2_temp_fp: str,
    ls8_temp_fp: str,
    cloud_mask_fp: str,
    rivers_fp: str,
    band_dict: dict,
    level: str, 
    roi: str,
    resample_method: str,
    res: int
):
    """
    Applies the appropriate rivers and cloud masks to a given image. 
    """
    with (
        rio.open(s2_temp_fp) as s2, 
        rio.open(ls8_temp_fp) as ls8, 
        rio.open(cloud_mask_fp) as mask, 
        rio.open(rivers_fp) as rivers
    ):

        s2_data = s2.read()
        s2_meta = s2.meta
        ls8_data = ls8.read()
        ls8_meta = ls8.meta
        cloud_mask_data = mask.read(1)
        river_mask_data = rivers.read(1)

        s2_valid = np.any(s2_data != 0, axis=0) # Checks for any bands not equal to zero
        ls8_valid = np.any(ls8_data != 0, axis=0)
        # Valid pixels for images and cloud mask
        valid_pixels_mask = s2_valid & ls8_valid & (cloud_mask_data == 0) & (river_mask_data == 0)

        s2_masked = s2_data.copy()
        for i in range(s2.count):
            s2_masked[i, ~valid_pixels_mask] = 0

        ls8_masked = ls8_data.copy()
        for i in range(ls8.count):
            ls8_masked[i, ~valid_pixels_mask] = 0


        out_dir = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}{res}/'
        os.makedirs(out_dir, exist_ok=True)

        s2_basename = os.path.basename(s2_temp_fp)
        ls8_basename = os.path.basename(ls8_temp_fp)
        s2_out_fp = f'{out_dir}{s2_basename}'
        ls8_out_fp = f'{out_dir}{ls8_basename}'

        s2_meta.update({'nodata': 0})
        ls8_meta.update({'nodata': 0})

        with rio.open(s2_out_fp, 'w', **s2_meta) as dst_s2:
            dst_s2.write(s2_masked)
            for idx, band_name in band_dict.items():
                dst_s2.set_band_description(idx, band_name)

        with rio.open(ls8_out_fp, 'w', **ls8_meta) as dst_ls8:
            dst_ls8.write(ls8_masked)
            for idx, band_name in band_dict.items():
                dst_ls8.set_band_description(idx, band_name)


def reproj_mask_img_pairs(
    pair: tuple,
    level: str,
    resample_method: str,
    res: str,
    band_desc: dict
):

    pair_paths = check_for_pair_fp(pair, level)
    if pair_paths is not None:
        print("------------------------")
        roi = extract_unique(pair_paths, roi_pattern)[0]
        date = extract_unique(pair_paths, date_pattern)[0]
        s2_fp = pair_paths[0]
        ls8_fp = pair_paths[1]
        ref_fp = f'./data/roi_shapes/rois/rasterized_{roi}_shape_res{res}.tif'
        mask_fp = f'./data/{level}_masks/CommonMask_date_{date}_roi_{roi}.tif'
        river_mask_fp = f'./data/river_files/{roi}_binary_rivers_dilated180_res{res}.tif'

        # Reproject the data to common reference grid write to temp folder
        s2_temp_fp = reproject_to_ref(
            s2_fp,
            ref_fp,
            resample_method
        )

        ls8_temp_fp = reproject_to_ref(
            ls8_fp,
            ref_fp,
            resample_method
        ) 
        print("*****")

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
            river_mask_fp,
            band_desc,
            level,
            roi,
            resample_method,
            res
        )
        print(roi, date)
        os.remove(s2_temp_fp)
        os.remove(ls8_temp_fp)
        os.remove(cloud_temp_fp)

# %%

for p in pairs:

    reproj_mask_img_pairs(
        pair=p,
        level=level,
        resample_method=resample_method,
        res=res,
        band_desc=band_desc
    )



    

# %%
