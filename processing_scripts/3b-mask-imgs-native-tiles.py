"""
"""
# %% Libraries and file paths

import os
import glob
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

from functions.img_data_fetching_functions import extract_unique

level = 'toa'

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
    secondary_img_path: str,
):
    """
    Due to different tile coverages across the regions of interest, 
    the LS8 and S2 images often have different valid footprints and even utm zones.
    This function ensures the valid footprints are the same for both images. 

    Defintionally:
    - The primary image is the one where the footprint is not being reprojected
    - The secondary image has its footprint reprojected to match the primary image.
    """

    with (
        rio.open(primary_img_path) as primary, 
        rio.open(secondary_img_path) as secondary, 
    ):
        out_meta = primary.meta.copy()
        primary_data = primary.read()

        # Create a simple binary mask for the primary image (1 for valid data, 0 for invalid)
        primary_nan_mask = np.isnan(primary_data)
        primary_valid = ~np.any(primary_nan_mask, axis=0)  # True where ALL bands have valid data
        
        # Create a simple binary mask for the secondary image
        secondary_data = secondary.read()
        secondary_nan_mask = np.isnan(secondary_data)
        secondary_valid = ~np.any(secondary_nan_mask, axis=0)  # True where ALL bands have valid data
        
        # Create a destination array for the reprojected secondary mask
        secondary_reproj = np.zeros(
            (primary.height, primary.width),
            dtype=np.uint8
        )

        reproject(
            source=secondary_valid.astype(np.uint8),
            destination=secondary_reproj,
            src_transform=secondary.transform,
            src_crs=secondary.crs,
            dst_transform=primary.transform,
            dst_crs=primary.crs,
            resampling=Resampling.nearest,
        )
        
        # Valid pixel mask for both images primary and secondar (resampled)
        valid_pixels_mask = (secondary_reproj == 1) & primary_valid

        temp_dir = './data/temp/'
        primary_basename = os.path.basename(primary_img_path)
        primary_satellite = primary_basename.split('_')[0]
        temp_parts = primary_basename.split('_')[1:]
        temp_path = f"{temp_dir}{primary_satellite}_CommonInvalids_{'_'.join(temp_parts)}"

        out_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0
        })

        with rio.open(temp_path, 'w', **out_meta) as dst_valid:
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
    """
    Applies cloud, river, and valid footprint masks to an image and writes the result to disk.
    
    Parameters:
    -----------
    img_fp: Path to the source image
    cloud_mask_fp: Path to the cloud mask (0=clear, 1=cloud)
    river_mask_fp: Path to the river mask (0=land, 1=river)
    valid_mask_fp: Path to the valid pixel mask (1=valid, 0=invalid)
    level: Processing level ('toa' or 'sr')
    roi: Region of interest identifier
    band_dict: Dictionary mapping band indices to band names
    """
    with (
        rio.open(img_fp) as src,
        rio.open(cloud_mask_fp) as cld,
        rio.open(river_mask_fp) as riv,
        rio.open(valid_mask_fp) as val
    ):
        # Read the image and mask data
        img_data = src.read()
        img_meta = src.meta.copy()

        # Read the masks
        cloud_mask = cld.read()
        river_mask = riv.read()
        valid_mask = val.read()

        # Make the comprehensive mask to apply to the image
        img_nans = np.isnan(img_data)
        img_nan_pixels = np.any(img_nans, axis=0)  # True where ANY band has NaN values

        # Combine all the mask conditions
        combined_mask = (cloud_mask == 0) & (river_mask == 0) & (valid_mask == 1) & (~img_nan_pixels)
        # Need to remove dimension from mask to make 2-D array
        if combined_mask.ndim > 2:
            combined_mask = combined_mask.squeeze() 

        # Note: GEE exports don't have explicit nodata values in the metadata.
        # We need populate these with -1 values. 

        # Apply the mask to the image
        img_masked = img_data.copy()
        for i in range(src.count):
            img_masked[i, ~combined_mask] = -1

        # Write the masked image
        out_dir = f'./data/{level}_images/roi_{roi}_noresample/'
        os.makedirs(out_dir, exist_ok=True)

        img_basename = os.path.basename(img_fp)
        img_out_fp = f'{out_dir}{img_basename}'

        # Update the metadata for the output image
        img_meta.update({
            'nodata': -1
        })

        # Write the masked image to disk
        with rio.open(img_out_fp, 'w', **img_meta) as dst:
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
        cloud_mask_fp = f'./data/common_masks/CommonMask_date_{date}_roi_{roi}.tif'

        if not os.path.exists(cloud_mask_fp):
            print(f'Missing cloud mask for {roi} on {date}. Skipping...')
            return
        
        
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

        # Reproject the common valid footprint to the LS8 grid
        common_valid_ls8_fp = make_img_valid_footprint_mask(
            primary_img_path=ls_fp,
            secondary_img_path=s2_fp,
        )

        # Reproject the common valid footprint to the S2 grid
        common_valid_s2_fp = make_img_valid_footprint_mask(
            primary_img_path=s2_fp,
            secondary_img_path=ls_fp,
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
"""
Old code for the negative values. Ingore this will probably delete later
"""

# def make_img_valid_footprint_mask_neg_vals(
#     primary_img_path: str,
#     primary_alt_lvl_path: str,
#     secondary_img_path: str,
#     secondary_alt_lvl_path: str
# ):
#     """
#     Due to different tile coverages across the regions of interest, 
#     the LS8 and S2 images often have different valid footprints and even utm zones.
#     This function ensures the valid footprints are the same for both images. 

#     Defintionally:
#     - The primary image is the one where the footprint is not being reprojected
#     - The secondary image has its footprint reprojected to match the primary image.

#     NOTE: We are no longer omitting the negative values. I left this code commented out
#     for reference, will delete later later. In this commented out code, the alt_lvl is the 
#     alternative level (SR/TOA or TOA/SR). This ensures a common negative pixel mask for different 
#     levels of Atmospheric correction
#     """

#     with (
#         rio.open(primary_img_path) as primary, 
#         rio.open(primary_alt_lvl_path) as primary_alt_lvl,
#         rio.open(secondary_img_path) as secondary, 
#         rio.open(secondary_alt_lvl_path) as secondary_alt_lvl
#     ):
#         primary_meta = primary.meta.copy()
#         primary_data = primary.read()
#         primary_alt_lvl_data = primary_alt_lvl.read()
#         secondary_data = secondary.read()
#         secondary_alt_lvl_data = secondary_alt_lvl.read()

#         # BUG: Interestingly, the TOA and SR data can have different shapes for the same satellite.
#         #     They might be off by one pixel. I reproject to get around this...
#         # NOTE: I use average resampling and float32 data types for a more conservative invalid pixel masking.
#         #      This ensures any contributing invalid pixels durring resampling are flagged. 
#         #      I do this to not have 'partial' coverage and edge effects. 
#         secondary_mask = np.all(secondary_data > 0, axis=0)
#         secondary_reproj = np.zeros(
#             (primary.height, primary.width), 
#             dtype='float32'
#         )
#         secondary_mask_alt_lvl = np.all(secondary_alt_lvl_data > 0, axis = 0)
#         secondary_alt_lvl_reproj = np.zeros(
#             (primary.height, primary.width),
#             dtype='float32'
#         )

#         # Reproject both of the TOA and SR invalid pixel masks for the secondary image
#         reproject(
#             source=secondary_mask.astype('float32'),
#             destination=secondary_reproj,
#             src_transform=secondary.transform,
#             src_crs=secondary.crs,
#             dst_transform=primary.transform,
#             dst_crs=primary.crs,
#             resampling=Resampling.average
#         )
#         reproject(
#             source=secondary_mask_alt_lvl.astype('float32'),
#             destination=secondary_alt_lvl_reproj,
#             src_transform=secondary.transform,
#             src_crs=secondary.crs,
#             dst_transform=primary.transform,
#             dst_crs=primary.crs,
#             resampling=Resampling.average
#         )

#         # Reproject the alternate atmospheric correction for the primary image. 
#         primary_mask_alt_lvl = np.all(primary_alt_lvl_data > 0, axis=0)
#         primary_alt_lvl_reproj = np.zeros((primary.height, primary.width), dtype='float32')
#         reproject(
#             source=primary_mask_alt_lvl.astype('float32'),
#             destination=primary_alt_lvl_reproj,
#             src_transform=primary_alt_lvl.transform,
#             src_crs=primary_alt_lvl.crs,
#             dst_transform=primary.transform,
#             dst_crs=primary.crs,
#             resampling=Resampling.average
#         )

#         # Convert to binary arrays with most conservative masking of invalid pixels
#         secondary_reproj_binary = (secondary_reproj == 1.0).astype('uint8')
#         secondary_alt_lvl_reproj_binary = (secondary_alt_lvl_reproj == 1.0).astype('uint8')
#         primary_alt_lvl_reproj_binary = (primary_alt_lvl_reproj == 1.0).astype('uint8')
#         primary_mask = np.all(primary_data > 0, axis=0).astype('uint8')
        
#         valid_pixels_mask = (
#             primary_mask & primary_alt_lvl_reproj_binary & secondary_reproj_binary & secondary_alt_lvl_reproj_binary
#         )

#         temp_dir = './data/temp/'
#         primary_basename = os.path.basename(primary_img_path)
#         primary_satellite = primary_basename.split('_')[0]
#         temp_parts = primary_basename.split('_')[1:]
#         temp_path = f"{temp_dir}{primary_satellite}_CommonInvalids_{'_'.join(temp_parts)}"

#         primary_meta.update({
#             'count': 1,
#             'dtype': 'uint8'
#         })

#         with rio.open(temp_path, 'w', **primary_meta) as dst_valid:
#             dst_valid.write(valid_pixels_mask.astype('uint8')[np.newaxis, :, :])
        
#     return temp_path