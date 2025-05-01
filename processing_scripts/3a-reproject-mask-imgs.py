"""
This script reprojects Sentinel-2 and Landsat-8 images to a common reference grid, applies cloud and river masks,
""" 
# %% 1.0 Libraries and file paths

import os
import glob
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

from functions.img_data_fetching_functions import extract_unique

level = 'toa' # must be 'sr' or 'toa'
res = 60 # 30 or 60 meters
resample_method = 'lanczos' # 'nearest', 'bilinear', 'cubic', 'average', 'lanczos'

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

# Get all the observation dates and image targe areas in the directory
pairs = list(product(unique_rois, unique_dates))


# %% 2.0 Functions

def check_for_pair_fp(
    pair_info: tuple,
    level: str
):
    """
    This checks to see if the combination of date and roi from pair_info exists in image downloads
    """
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
    Reads the images or common masks and reprojects them to match the reference raster.
    Writes the data to the "temp" folder
    
    Parameters:
    -----------
    img_fp : str
        Path to the source image to reproject
    ref_fp : str
        Path to the reference image that defines the target projection gererated by (1-rasterize_roi_shapes.py)
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
        #print(src.nodata)
        # For masks specifically (when processing cloud_temp_fp)
        out_data = np.full(
            (src.count, ref.height, ref.width), 
            fill_value=-1, 
            dtype=np.float32
        )

        src_data = src.read()
        # NOTE: Because Earth Engine does not define nodata metadata, we need to check for NaN values
        nan_mask = np.isnan(src_data)
        if nan_mask.any():
            print(f"Found {np.sum(nan_mask)} NaN values in source data")
            # Replace NaNs with a temporary value that will be mapped to nodata
            src_data_masked = src_data.copy()
            src_data_masked[nan_mask] = -9999  # Temporary value
        else:
            src_data_masked = src_data


        out_meta.update({
            'crs': ref_meta['crs'],
            'transform': ref_meta['transform'],
            'width': ref_meta['width'],
            'height': ref_meta['height'],
            'nodata': -1,
            'dtype': 'float32'
        })

        reproject(
            source=src_data_masked,
            destination=out_data,
            src_transform=src.transform, 
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=resamp, 
            src_nodata=-9999 if nan_mask.any() else None,
            dst_nodata=-1
        )

        # Any values that came from NaNs should be set to nodata
        if nan_mask.any():
            # This is needed because resampling might introduce intermediate values
            # when interpolating between NaN and valid pixels
            out_data[np.isnan(out_data)] = -1

        temp_dir = './data/temp/'
        basename = os.path.basename(img_fp)
        ref_res = round(ref.transform[0])
        out_fp = f'{temp_dir}/reprojected_{resample_method}{ref_res}_{basename}'

        with rio.open(out_fp, 'w', **out_meta) as dst:
            dst.write(out_data)

        #print(f"Reprojected {out_fp}")

        return out_fp
    
# NOTE: This function is no longer used, becuase we are keeping negative relfectance values

# def make_invalid_mask_for_other_level(
#     other_level_s2_fp: str,
#     other_level_ls8_fp: str,
#     ref_fp: str,
#     resample_method: str
# ):
#     """
#     Creates a mask for the other level (sr or toa) images' negative values. 
#     This removes the same negative values across AC levels.
#     """
#     resamp_methods = {
#         'nearest': Resampling.nearest,
#         'bilinear': Resampling.bilinear,
#         'cubic': Resampling.cubic,
#         'lanczos': Resampling.lanczos,
#         'average': Resampling.average
#     }
#     resamp = resamp_methods.get(resample_method)

#     if not os.path.exists(other_level_s2_fp) or not os.path.exists(other_level_ls8_fp):
#         return None
    
#     else:
#         with (
#             rio.open(other_level_s2_fp) as lvl_s2, 
#             rio.open(other_level_ls8_fp) as lvl_ls8,
#             rio.open(ref_fp) as ref
#         ):
            
#             other_level_s2_data = lvl_s2.read()
#             other_level_ls8_data = lvl_ls8.read()

#             out_s2_other_lvl = np.zeros((lvl_s2.count, ref.height, ref.width), dtype=other_level_s2_data.dtype)
#             out_ls8_other_lvl = np.zeros((lvl_ls8.count, ref.height, ref.width), dtype=other_level_ls8_data.dtype)
            
#             # Repoject the Sentinel-2 data from the alternate level
#             reproject(
#                 source=other_level_s2_data,
#                 destination=out_s2_other_lvl,
#                 src_transform=lvl_s2.transform,
#                 src_crs=lvl_s2.crs,
#                 dst_transform=ref.transform,
#                 dst_crs=ref.crs,
#                 resampling=resamp
#             )
#             # Reproject the Landsat-8 data from the alternate level
#             reproject(
#                 source=other_level_ls8_data,
#                 destination=out_ls8_other_lvl,
#                 src_transform=lvl_ls8.transform,
#                 src_crs=lvl_ls8.crs,
#                 dst_transform=ref.transform,
#                 dst_crs=ref.crs,
#                 resampling=resamp
#             )

#             # Create a mask for valid pixels in the other level data
#             other_level_valid = np.all(out_s2_other_lvl > 0, axis=0) & np.all(out_ls8_other_lvl > 0, axis=0)
#             other_level_valid = other_level_valid.astype('uint8') # Convert to uint8 for writing to temp
#             other_level_valid = other_level_valid[np.newaxis, :, :] # Add a new axis to match the number of bands

#             temp_dir = './data/temp/'
#             basename = os.path.basename(other_level_s2_fp)
#             basename = basename.replace('Sentinel2', 'OtherLevel_Invalids')
#             ref_res = round(ref.transform[0])
#             out_fp = f'{temp_dir}/{basename}_reprojected_{resample_method}{ref_res}.tif'

#             out_meta = lvl_s2.meta.copy()
#             out_meta.update({
#                 'crs': ref.crs,
#                 'transform': ref.transform,
#                 'width': ref.width,
#                 'height': ref.height,
#                 'count': 1,
#                 'dtype': 'uint8'
#             })
#             with rio.open(out_fp, 'w', **out_meta) as dst:
#                 dst.write(other_level_valid)

#             return out_fp

def apply_cloud_river_valid_masks(
    s2_temp_fp: str,
    ls8_temp_fp: str,
    cloud_mask_fp: str,
    rivers_fp: str,
    other_level_valid_fp: str,
    band_dict: dict,
    level: str, 
    roi: str,
    resample_method: str,
    res: int
):
    """
    Applies the appropriate rivers and cloud masks to a given image. 
    NOTE: We are not masking out negative values. We used to do this, but I'm keeping 
    them (commented out) as a reference for now. 
    """
    with (
        rio.open(s2_temp_fp) as s2, 
        rio.open(ls8_temp_fp) as ls8, 
        rio.open(cloud_mask_fp) as mask, 
        rio.open(rivers_fp) as rivers,
        #rio.open(other_level_valid_fp) as other_level_valid
    ):

        s2_data = s2.read()
        s2_meta = s2.meta
        ls8_data = ls8.read()
        ls8_meta = ls8.meta
        cloud_mask_data = mask.read(1)
        river_mask_data = rivers.read(1)
        # other_level_valid_data = other_level_valid.read(1)

        s2_coverage = ~np.all(s2_data == -1, axis=0) # Checks for any bands values less than zero
        ls8_coverage = ~np.all(ls8_data == -1, axis=0)
        coverage_mask = s2_coverage & ls8_coverage

        quality_mask = (cloud_mask_data == 0) & (river_mask_data == 0) #& (other_level_valid_data == 1)
        
        # Valid pixels for images and cloud mask
        valid_pixels_mask = coverage_mask & quality_mask

        s2_masked = s2_data.copy()
        for i in range(s2.count): # Applies mask to each band
            s2_masked[i, ~valid_pixels_mask] = -1

        ls8_masked = ls8_data.copy()
        for i in range(ls8.count):
            ls8_masked[i, ~valid_pixels_mask] = -1


        out_dir = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}{res}/'
        os.makedirs(out_dir, exist_ok=True)

        s2_basename = os.path.basename(s2_temp_fp)
        ls8_basename = os.path.basename(ls8_temp_fp)
        s2_out_fp = f'{out_dir}{s2_basename}'
        ls8_out_fp = f'{out_dir}{ls8_basename}'

        s2_meta.update({'nodata': -1})
        ls8_meta.update({'nodata': -1})

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
        mask_fp = f'./data/common_masks/CommonMask_date_{date}_roi_{roi}.tif'
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

        if not os.path.exists(mask_fp):
            print(f"Missing common mask for {roi} on {date}. Skipping...")
            os.remove(s2_temp_fp)
            os.remove(ls8_temp_fp)
            return

        cloud_temp_fp = reproject_to_ref(
            mask_fp,
            ref_fp, 
            'nearest'
        )

        # Paths to corresponding images with alternate atmospheric correction
        """
        NOTE: Because we keep negative values, this code is no longer used.
        I left it commented out for reference.
        """
        # if level == 'sr':
        #     other_level = 'toa'
        #     other_level_s2_fp = f'./data/toa_image_downloads/Sentinel2_toa_date_{date}_roi_{roi}.tif'
        #     other_level_ls8_fp = f'./data/toa_image_downloads/Landsat8_toa_date_{date}_roi_{roi}.tif'
        # elif level == 'toa':
        #     other_level = 'sr'
        #     other_level_s2_fp = f'./data/sr_image_downloads/Sentinel2_sr_date_{date}_roi_{roi}.tif'
        #     other_level_ls8_fp = f'./data/sr_image_downloads/Landsat8_sr_date_{date}_roi_{roi}.tif'
        # else:
        #     raise ValueError("Level must be 'sr' or 'toa'")
        
        # other_level_valid_mask_fp = make_invalid_mask_for_other_level(
        #     other_level_s2_fp=other_level_s2_fp,
        #     other_level_ls8_fp=other_level_ls8_fp,
        #     ref_fp=ref_fp,
        #     resample_method=resample_method
        # )

        # if other_level_valid_mask_fp is None:
        #     print(f'Missing {other_level} images for {roi} on {date}. Skipping...')
        #     os.remove(s2_temp_fp)
        #     os.remove(ls8_temp_fp)
        #     os.remove(cloud_temp_fp)

        #else:
        # Apply the cloud mask to the images
        other_level_valid_mask_fp = None
        apply_cloud_river_valid_masks(
            s2_temp_fp,
            ls8_temp_fp,
            cloud_temp_fp,
            river_mask_fp,
            other_level_valid_mask_fp,
            band_desc,
            level,
            roi,
            resample_method,
            res
        )

        print(roi, date, level)
        os.remove(s2_temp_fp)
        os.remove(ls8_temp_fp)
        os.remove(cloud_temp_fp)
        #os.remove(other_level_valid_mask_fp)

# %% Process the pairs

for p in pairs:

    reproj_mask_img_pairs(
        pair=p,
        level=level,
        resample_method=resample_method,
        res=res,
        band_desc=band_desc
    )



# %%
