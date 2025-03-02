# %% 1.0 

import glob
import re
import numpy as np
import pandas as pd
from itertools import product

import rasterio as rio
from rasterio.warp import reproject, Resampling

from image_analysis_functions import extract_unique

import pprint as pp

level = 'sr' #level should be 'sr' or 'toa'
raw_dir = f'./data/{level}_image_downloads/'
#roi_prefix = roi_name.split('_')[0]
roi_dir = './data/roi_shapes/rois/'

all_files = glob.glob(f'{raw_dir}/*.tif')
all_s2_files = glob.glob(f'{raw_dir}/Sentinel2*.tif')
all_ls8_files = glob.glob(f'{raw_dir}/Landsat8*.tif')

roi_pattern = r'roi_(.*?)_resampled'
date_pattern = r'date_(.*?)_roi'
resamp_pattern = r'resampled_(.*?)_'
unique_rois = extract_unique(all_files, roi_pattern)
unique_dates = extract_unique(all_files, date_pattern)
unique_resamps = extract_unique(all_files, resamp_pattern)

batches = list(product(unique_dates, unique_rois, unique_resamps))


# %% 2.0 Functions

def extract_date(export_name):
    """
    Finds the date pattern in export names.
    """
    date_pattern = r"date_(\d{4}-\d{2}-\d{2})"
    match = re.search(date_pattern, export_name)
    if match:
        return match.group(1)
    return None

def get_img_paths(batch_info: tuple, files: list):
    """ 
    Gets all the image paths for a given batch (i.e., same date and conditions)
    The batch is defined by the date, roi, and resampled value.
    """

    date = batch_info[0]
    roi = batch_info[1]
    resamp = batch_info[2]
    img_paths = [f for f in files if re.search(fr'date_{date}_roi_{roi}_resampled_{resamp}', f)]
    if len(img_paths) < 0:
        print(f"No images found for batch")
        return None
    else:
        return img_paths

def get_batch_attrs(
        level: str,
        batch_info: tuple
):
    """
    Finds the image attributes for given batch (with date, roi, level, and resampling method)
    """
    date = batch_info[0]
    roi = batch_info[1]
    resamp = batch_info[2]
    attrs_dir = './data/img_mask_solar_stats/all_exports/'
    ls8_attrs_path = f'{attrs_dir}LandSat8_img_attrs_for{roi}_{level}_resampled_{resamp}.csv'
    mask_attrs_path = f'{attrs_dir}mask_stats_for_{roi}_{level}_resampled_{resamp}.csv'
    s2_attrs_path = f'{attrs_dir}Sentinel2_img_attrs_for{roi}_{level}_resampled_{resamp}.csv'

    ls8_attrs = pd.read_csv(ls8_attrs_path)
    ls8_attrs['date'] = ls8_attrs['export_name'].apply(extract_date)
    ls8_attrs = ls8_attrs[ls8_attrs['date'] == str(date)]

    mask_attrs = pd.read_csv(mask_attrs_path)
    mask_attrs['date'] = mask_attrs['date'].astype(str)
    mask_attrs = mask_attrs[mask_attrs['date'] == date]

    s2_attrs = pd.read_csv(s2_attrs_path)
    s2_attrs['date'] = s2_attrs['export_name'].apply(extract_date)
    s2_attrs = s2_attrs[s2_attrs['date'] == str(date)]

    return (s2_attrs, ls8_attrs, mask_attrs)
    
    
def get_img_nan_frac(
    img_path: str,
    ref_raster: rio.DatasetReader
):
    """
    Reprojects the image to the reference raster (the roi mask)
    Then, the function calculates the amount of NaN & zero values inside the roi mask
    Returns a tuple with 
    """
    with rio.open(img_path) as src:
        # Create destination array with the shape (# bands, height, width)
        src_data = src.read()
        dst_data = np.empty((src.count, ref_raster.height, ref_raster.width), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_raster.transform,
            dst_crs=ref_raster.crs,
            resampling=Resampling.nearest 
        )
        roi_mask = ref_raster.read(1)

        invalid_pix = np.any(np.isnan(dst_data), axis=0)
        img_nan_pix = np.sum(invalid_pix & (roi_mask == 1)) # dst_data has nan values and roi_mask is 1
        roi_tot_pix = np.sum(roi_mask == 1)
        nan_frac = img_nan_pix / roi_tot_pix

        return (img_path, nan_frac)
    
def find_best_image(
    img_paths: list,
    ref_path: str,
):
    """
    Reprojects a list of images to the reference raster
    """

    ref_raster = rio.open(ref_path)
    #print(ref_raster.meta)
    nan_frac_list = []
    for path in img_paths:
        nan_frac = get_img_nan_frac(path, ref_raster)
        nan_frac_list.append(nan_frac)

    fracs = [f[1] for f in nan_frac_list]
    lowest_frac = min(fracs)
    best_img = [img[0] for img in nan_frac_list if img[1] == lowest_frac]
    # Edge case where both images have same nan frac
    if len(best_img) > 1:
        best_img = [best_img[0]]
    
    return best_img, lowest_frac

def select_attrs(best_img, all_attrs):
    """
    Selects attributes from a DataFrame based on the best image export name.
    """
    # Extract export name from file path
    export_name = best_img[0][0].split('downloads/')[1]
    export_name = export_name.replace(".tif", "")

    # Select attributes matching the export name
    selected_attrs = all_attrs[all_attrs['export_name'] == export_name]

    return selected_attrs

def find_best_images(
    batches: list,
    level: str,
    roi_dir: str
):
    img_pairs_list = []
    mask_attrs_list = []
    s2_attrs_list = []
    ls8_attrs_list = []

    for batch in batches:
        s2_imgs = get_img_paths(batch, all_s2_files)
        ls8_imgs = get_img_paths(batch, all_ls8_files)
    
        if not s2_imgs or not ls8_imgs:
            continue
    
        roi_name = batch[1]
        res = int(re.findall(r'\d+', batch[2])[0])
        ref_path = f'{roi_dir}rasterized_{roi_name}_shape_res{res}.tif'

        s2_attrs, ls8_attrs, mask_attrs = get_batch_attrs(level=level, batch_info=batch)
        best_s2 = find_best_image(s2_imgs, ref_path)
        print(f'Best S2 has {(best_s2[1] * 100):.2f}% of ROI masked')
        best_ls8 = find_best_image(ls8_imgs, ref_path)
        print(f'Best LS8 has {(best_ls8[1] * 100):.2f}% of ROI masked')
        s2_attrs = select_attrs(best_s2, s2_attrs)
        ls8_attrs = select_attrs(best_ls8, ls8_attrs)
        
        best_pair = {
            's2_fp': best_s2[0], 
            's2_nan_frac': best_s2[1],
            'ls8_fp': best_ls8[0],
            'ls8_nan_frac': best_ls8[1]
        }

        best_pair = pd.DataFrame(best_pair)


        img_pairs_list.append(best_pair)
        mask_attrs_list.append(mask_attrs)
        s2_attrs_list.append(s2_attrs)
        ls8_attrs_list.append(ls8_attrs)

    mask_attrs_out = pd.concat(mask_attrs_list)
    s2_attrs_out = pd.concat(s2_attrs_list)
    ls8_attrs_out = pd.concat(ls8_attrs_list)
    img_pairs_out = pd.concat(img_pairs_list)

    return mask_attrs_out, s2_attrs_out, ls8_attrs_out, img_pairs_out

# %% Get the best image pairs and filtered attributes

output = find_best_images(batches, level, roi_dir)
mask_attrs, s2_attrs, ls8_attrs, img_pairs = output

# %% 

img_pairs.to_csv(f'./data/{level}_img_exports_to_use.csv', index=False)
s2_attrs.to_csv('./data/img_mask_solar_stats/Sentinel2_attrs_best_imgs_v1.csv', index=False)
ls8_attrs.to_csv('./data/img_mask_solar_stats/LandSat8_attrs_best_imgs_v1.csv', index=False)
mask_attrs.to_csv('./data/img_mask_solar_stats/mask_attrs_best_imgs_v1.csv', index=False)



# 
# %%
