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
        satellite = os.path.basename(target_fp).split('_')[0]
        river_basename = os.path.basename(river_mask_fp)
        cloud_basename = os.path.basename(cloud_mask_fp)

        temp_river_fp = f'{temp_dir}{satellite}_{river_basename}'
        temp_cloud_fp = f'{temp_dir}{satellite}_{cloud_basename}'

        with rio.open(temp_river_fp, 'w', **out_river_meta) as dst_river:
            dst_river.write(out_river_data)

        with rio.open(temp_cloud_fp, 'w', **out_cloud_meta) as dst_cloud:
            dst_cloud.write(out_cloud_data)

    return temp_cloud_fp, temp_river_fp

def make_img_valid_footprint_mask(
    primary_img_path: str,
    secondary_img_path: str
):
    """
    Due to different tile coverages across the regions of interest, 
    the LS8 and S2 images often have different valid footprints and even utm zones.
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
            dtype=primary.dtypes[0]
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
        secondary_valid_mask = np.any(secondary_reproj > 0, axis=0)
        
        valid_pixels_mask = primary_valid_mask & secondary_valid_mask

        temp_dir = './data/temp/'
        primary_basename = os.path.basename(primary_img_path)
        primary_satellite = primary_basename.split('_')[0]
        temp_parts = primary_basename.split('_')[1:]
        temp_path = f"{temp_dir}{primary_satellite}_CommonInvalids_{'_'.join(temp_parts)}"

        primary_meta.update({
            'count': 1,
            'dtype': 'uint8'
        })

        with rio.open(temp_path, 'w', **primary_meta) as dst_valid:
            dst_valid.write(valid_pixels_mask.astype('uint8')[np.newaxis, :, :])
        
    return temp_path
    
def apply_masks_to_image_write_out(
    img_fp: str,
    cloud_mask_fp: str,
    river_mask_fp: str,
    valid_mask_fp: str,
    level: str,
    roi: str,
    band_dict: dict
):
    with (
        rio.open(img_fp) as src,
        rio.open(cloud_mask_fp) as cld,
        rio.open(river_mask_fp) as riv,
        rio.open(valid_mask_fp) as val
    ):
        # Read the image and mask data
        img_data = src.read()
        img_meta = src.meta
        cloud_mask = cld.read()
        river_mask = riv.read()
        valid_mask = val.read()

        # Make the comprehensive mask to apply to the image
        mask = (cloud_mask == 0) & (river_mask == 0) & (valid_mask == 1)
        mask = mask.squeeze() # Need to remove dimension from mask to make 2-D array
        print(mask.shape)
        # Apply the mask to the image
        img_masked = img_data.copy()
        print(f"img_masked shape: {img_masked.shape}")
        for i in range(src.count):
            img_masked[i, ~mask] = 0

        # Write the masked image
        out_dir = f'./data/{level}_images/roi_{roi}_noresample/'
        os.makedirs(out_dir, exist_ok=True)

        img_basename = os.path.basename(img_fp)
        img_out_fp = f'{out_dir}{img_basename}'

        out_meta = img_meta.copy()
        with rio.open(img_out_fp, 'w', **out_meta) as dst:
            dst.write(img_masked)
            for idx, band_name in band_dict.items():
                dst.set_band_description(idx, band_name)
        print(f'Image Processed to {img_out_fp}')


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
        cloud_mask_ls8_fp, river_mask_ls8_fp = reproject_cloud_rivers_masks(
            cloud_mask_fp=cloud_mask_fp,
            river_mask_fp=river_mask_fp,
            target_fp=ls_fp
        )

        # Reproject masks into S2
        cloud_mask_s2_fp, river_mask_s2_fp = reproject_cloud_rivers_masks(
            cloud_mask_fp=cloud_mask_fp,
            river_mask_fp=river_mask_fp, 
            target_fp=s2_fp
        )

        # Reproject the common valid footprint to LS8 grid
        common_valid_ls8_fp = make_img_valid_footprint_mask(
            primary_img_path=ls_fp,
            secondary_img_path=s2_fp
        )

        # Reproject the common valid footprint to S2 grid
        common_valid_s2_fp = make_img_valid_footprint_mask(
            primary_img_path=s2_fp,
            secondary_img_path=ls_fp
        )

        # Write the masked Landsat8
        apply_masks_to_image_write_out(
            img_fp=ls_fp,
            cloud_mask_fp=cloud_mask_ls8_fp,
            river_mask_fp=river_mask_ls8_fp,
            valid_mask_fp=common_valid_ls8_fp,
            level=level,
            roi=roi,
            band_dict=band_desc
        )
        # Write the masked Sentinel-2
        apply_masks_to_image_write_out(
            img_fp=s2_fp,
            cloud_mask_fp=cloud_mask_s2_fp,
            river_mask_fp=river_mask_s2_fp,
            valid_mask_fp=common_valid_s2_fp,
            level=level,
            roi=roi,
            band_dict=band_desc
        )

        # Clean the temp folder
        os.remove(cloud_mask_ls8_fp)
        os.remove(cloud_mask_s2_fp)
        os.remove(river_mask_ls8_fp)
        os.remove(river_mask_s2_fp)
        os.remove(common_valid_ls8_fp)
        os.remove(common_valid_s2_fp)

        print('-----------------------------')




# %% 

for p in pairs:
    mask_tiles_at_native_trans(
        pair=p,
        level=level,
        band_desc=band_desc
    )
# %%
