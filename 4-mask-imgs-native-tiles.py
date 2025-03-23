# %% Libraries and file paths

import os
import glob
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

from image_analysis_functions import extract_unique

level = 'sr'

download_dir = f'./data/{level}_image_downloads/'
roi_dir = f'./data/roi_shapes/rois/'

band_desc = {
    1: 'Blue',
    2: 'Green',
    3: 'Red', 
    4: 'NIR'
}

all_s2_files = glob.glob(f'{download_dir}/Sentinel2*.tif')
print(len(all_s2_files))
all_ls8_files = glob.glob(f'{download_dir}/Landsat8*.tif')
print(len(all_ls8_files))

roi_pattern = r'roi_(.*?).tif'
date_pattern = r'date_(.*?)_roi'
unique_rois = extract_unique(all_ls8_files, roi_pattern)
unique_dates = extract_unique(all_s2_files, date_pattern)

pairs = list(product(unique_rois, unique_dates))
pairs = pairs[0:20]

print(pairs)

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
    
def reproject_cloud_rivers_masks(
    cloud_mask_fp: str,
    river_mask_fp: str,
    target_fp: str,
):
    """
    Reprojects the common cloud mask and river mask
    Uses transform and resolution of the target image's (S2 or LS8 tile's)
    Writes the reprojected data to a temporary folder. 
    """

    with rio.open(target_fp) as tgt:
        target_meta = tgt.meta.copy()

    with rio.open(cloud_mask_fp) as src_cloud, rio.open(river_mask_fp) as src_river:
        out_cloud_meta = src_cloud.meta.copy()
        out_river_meta = src_river.meta.copy()
        out_cloud_data = np.zeros((src_cloud.count, tgt.height, tgt.width), dtype=src_cloud.dtypes[0])
        out_river_data = np.zeros((src_river.count, tgt.height, tgt.width), dtype=src_river.dtypes[0])
        src_cloud_data = src_cloud.read()
        src_river_data = src_river.read()

        out_cloud_meta.update({
            'crs': target_meta['crs'],
            'transform': target_meta['transform'], 
            'width': target_meta['width'], 
            'height': target_meta['height']
        })

        out_river_meta.update({
            'crs': target_meta['crs'],
            'transform': target_meta['transform'], 
            'width': target_meta['width'], 
            'height': target_meta['height']
        })

        # Cloud mask
        reproject(
            source=src_cloud_data, 
            destination=out_cloud_data,
            src_transform=src_cloud.transform,
            src_crs=src_cloud.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )
        # River mask 
        reproject(
            source=src_river_data,
            destination=out_river_data,
            src_transform=src_river.transform, 
            src_crs=src_cloud.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )

        temp_dir = './data/temp/'
        tgt_basename = os.path.basename(target_fp)
        river_basename = os.path.basename(river_mask_fp)
        cloud_basename = os.path.basename(cloud_mask_fp)
        satellite = tgt_basename.split('_')[0]

        out_river_fp = f'{temp_dir}{satellite}_{river_basename}'
        out_cloud_fp = f'{temp_dir}{satellite}_{cloud_basename}'

        print(out_river_fp, out_cloud_fp)

        with rio.open(out_river_fp, 'w', **out_river_meta) as dst_river:
            dst_river.write(out_river_data)

        with rio.open(out_cloud_fp, 'w', **out_cloud_meta) as dst_cloud:
            dst_cloud.write(out_cloud_data)

def make_img_valid_footprint_mask(
    primary_img_path: str,
    secondary_img_path: str
):
    """
    Due to different tile coverages accross the roi, 
    the LS8 and S2 images often have different valid footprints.
    Furthermore, Atmospheric Correction can introduce negative values, 
    which should be removed (identically) from both images in the set
    """

    with rio.open(primary_img_path) as primary, rio.open(secondary_img_path) as secondary:

        primary_meta = primary.meta
        primary_data = primary.read()
        secondary_meta = secondary.meta
        secondary_data = secondary.read()
        secondary_reproj = np.zeros(
            (primary.count, primary.height, primary.width), 
            dtype=primary.dtypes
        )
        secondary_meta.update({
            'crs': primary_meta['crs'],
            'transform': primary_meta['transform'],
            'width': primary_meta['width'],
            'height': primary_meta['height']
        })

        reproject(
            source=secondary_data,
            destination=secondary_reproj,
            src_transform=secondary.transform,
            src_crs=secondary.crs,
            dst_transform=primary.transform,
            dst_crs=primary.crs,
            resampling=Resampling.nearest
        )

        primary_valid_mask = np.any(primary_data > 0, axis=0)
        print(primary_valid_mask.shape)
        secondary_valid_mask = np.any(secondary_reproj > 0, axis=0)
        print(secondary_valid_mask.shape)

def mask_tiles_at_native_trans(
    pair: tuple,
    level: str, 
    band_desc: dict
):
    """
    Brings all the masks into the Sentinel-2 or Landsat8 tile's transformation
    This enables subsequent water frac calcs without any reprojection
    """
    pair_paths = check_for_pair_fp(pair, level)
    if pair_paths is not None:
        print("-----------------------")
        roi = extract_unique(pair_paths, roi_pattern)[0]
        date = extract_unique(pair_paths, date_pattern)[0]
        s2_fp = pair_paths[0]
        ls_fp = pair_paths[1]
        cloud_mask_fp = f'./data/{level}_masks/CommonMask_date_{date}_roi_{roi}.tif'
        # The res does not matter for river files, because we reproject the river file
        river_mask_fp = f'./data/river_files/{roi}_binary_rivers_dilated180_res30.tif' 

        # Reproject masks into LS8
        reproject_cloud_rivers_masks(
            cloud_mask_fp=cloud_mask_fp,
            river_mask_fp=river_mask_fp,
            target_fp=ls_fp
        )

        # Reproject masks into S2
        reproject_cloud_rivers_masks(
            cloud_mask_fp=cloud_mask_fp,
            river_mask_fp=river_mask_fp, 
            target_fp=s2_fp
        )

        # Reproject the Sentinel-2 valid footprint to LS8 Resolution
        make_img_valid_footprint_mask(
            primary_img_path=s2_fp,
            secondary_img_path=ls_fp
        )
        # Reproject the Landsat8 valid footprint to S2 Resolution
        make_img_valid_footprint_mask(
            primary_img_path=ls_fp,
            secondary_img_path=s2_fp
        )




# %% 

for p in pairs:
    mask_tiles_at_native_trans(
        pair=p,
        level=level,
        band_desc=band_desc
    )
# %%
