"""
Data Fetching Functions:
These obtain image data and masks for the image analysis functions, in the 
regress_image_functions.py and water_area_thresholding_functions.py files
"""
import os
import re
import random
import glob
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling

random.seed(20)

def extract_unique(
    files: list, 
    pattern: re.Pattern[str]
):
    unique_items = set()
    for f in files:
        match = re.search(pattern, f)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

def check_match_imgs(
    fp1: str, 
    fp2: str, 
):

    """
    Checks to ensure both images exist in directory
    If both images do not exist, then return false
    Otherwise, specifies if one file is missing and return false. 
    """
    if os.path.exists(fp2) and os.path.exists(fp1):
        return True
    elif not os.path.exists(fp2) and not os.path.exists(fp1):
        #print(f'No data for {image_info}')
        return False
    elif not os.path.exists(fp2) and os.path.exists(fp1):
        print(f'Missing file {fp2}')
        return False
    else:
        print(f'Missing file {fp1}')
        return False
    
def read_band_by_description(
    raster_path: str,
    description: str, 
    image_window_params: Optional[dict]
):
    """
    Returns an array from the band description (string, not index) in a raster file. 
    Optional argument image_window_params to crop the band to a window. 
    NOTE: This image_window_params is only used for cropping the PLD masks to
        the Sentinel-2 and Landsat8 imagery.
    """
    
    data = None
    with rio.open(raster_path) as src:
        # Get a description list
        desc_list = src.descriptions
        # Read the data for a specific band
        for idx, desc in enumerate(desc_list, start=1):
            if desc == description and image_window_params is None:
                data = src.read(idx)
                return data
            # Crops the data using image window params
            elif desc == description and image_window_params is not None:
                
                window = from_bounds(*image_window_params['bounds'], 
                                     transform=src.transform
                        )
                data = src.read(
                    idx,
                    window=window, 
                    out_shape=image_window_params['shape'],
                    boundless=True, # Assigns no data to pixels outside the raster's extent.
                    resampling=Resampling.nearest
                )
                
                return data
            
def apply_measure_mask(
    image_data: np.array,
    measure_mask: np.array
):
    """
    Uses the binary measure_mask to select pixels based on lake, land, or shoreline.
    the arrays must have the same dimensions
    """
    masked_data = image_data.copy()
    masked_data = np.where(measure_mask == 1, image_data, np.nan)

    return masked_data

def rio_get_data_arrays_with_common_trans(
    ls_path: str, 
    s2_path: str, 
    band_name: str
):
    """
    NOTE: This function is only intended for images resampled to a common reference grid. 
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands.
    Converts any values <= 0 to np.nan for both datasets
    """

    ls_data = read_band_by_description(ls_path, band_name, image_window_params=None) #Keep image window params None
    s2_data = read_band_by_description(s2_path, band_name, image_window_params=None) #Keep image window params None

    ls_data = ls_data.copy()
    s2_data = s2_data.copy()

    # Zero's values should already be nan, but just in case
    ls_data = np.where(ls_data >= 0, ls_data, np.nan)
    s2_data = np.where(s2_data >= 0, s2_data, np.nan)

    if ls_data.shape != s2_data.shape:
        raise ValueError(
        f"Incompatible array shapes: Landsat8 shape {ls_data.shape} != "
        f"Sentinel-2 shape {s2_data.shape}. Images must be resampled to identical dimensions "
        f"For images in native grid, use rio_get_data_array_native_trans()"
    )

    # Get the data's bounds and transform (LS and S2 grid will already be the same) 
    # The purpose here is window for the PLD mask already on their grids. 
    with rio.open(ls_path) as src: 
        meta = src.meta
        ls_bounds = src.bounds
        ls_transform = src.transform
        ls_shape = src.shape

    image_window_params = {
        'bounds': ls_bounds,
        'transform': ls_transform,
        'shape': ls_shape
    }

    return ls_data, s2_data, image_window_params


def downsample_image_arrays(
    arr1_pixels: np.array,
    arr2_pixels: np.array,
    sample_size: int
):
    """
    Downsampling makes the pixel regressions more efficient.
    Inputs: 2D images with identical masked pixels = np.nan
    Returns: 1D arrays with randomly downsampled to the sample_size if above pixel count.
    """

    if arr1_pixels.shape != arr2_pixels.shape:
        raise ValueError(
        f"Incompatible array shapes: array1 shape {arr1_pixels.shape} != "
        f"array2 shape {arr2_pixels.shape}. Images must be resampled to identical dimensions "
    )
    # Create a common mask so that we drop the same pixels in both arrays
    # (exclude NaNs or zeros in either array).
    common_mask = (
        ~np.isnan(arr1_pixels) & (arr1_pixels != 0) &
        ~np.isnan(arr2_pixels) & (arr2_pixels != 0)
    )

    # Flatten both arrays using the same mask
    arr1_flat = arr1_pixels[common_mask].flatten()
    arr2_flat = arr2_pixels[common_mask].flatten()

    # Double-check that both arrays have the same length after masking
    if arr1_flat.size != arr2_flat.size:
        raise ValueError(
            "After applying the common mask, the image arrays "
            "do not have the same valid pixel count."
        )

    valid_pix_cnt = arr1_flat.size # ls_flat.size and s2_flat.size will be the same
    if valid_pix_cnt < sample_size:
        print(f"Not downsampling the number of measured pixels {valid_pix_cnt} < {sample_size}")
        return arr1_flat, arr2_flat, valid_pix_cnt
    else:
        sample_idx = np.random.choice(arr1_flat.size, sample_size, replace=False)
        # Applies sample_idx to pixels_flat
        arr1_sampled = arr1_flat[sample_idx]
        arr2_sampled = arr2_flat[sample_idx]

        return arr1_sampled, arr2_sampled, valid_pix_cnt

def rio_get_data_arrays_native_trans(
    ls_fp,
    s2_fp, 
    band_name
):
    """
    Returns data arrays from the Sentinel-2 and Landsat8 images in their native tile resolution
    NOTE: This function is only intended for images with no resampling.
        This means the arrays will have different shapes. 
    Function was built for lake area classifacation in native resolution. Do not use for pixel regressions. 
    """

    ls_data = read_band_by_description(ls_fp, band_name, image_window_params=None) #Keep image window params None
    s2_data = read_band_by_description(s2_fp, band_name, image_window_params=None) #Keep image window params None

    # Zero's values should already be nan, but just in case
    ls_data_out = np.where(ls_data >= 0, ls_data, np.nan)
    s2_data_out = np.where(s2_data >= 0, s2_data, np.nan)

    return ls_data_out, s2_data_out


def make_measure_mask(
    pld_path: str, 
    image_window_params: dict, 
    zone: str, # 1) lake, 2) land, or 3) shoreline
    buffer_delim: int,
    buffer_delim_outer: Optional[int]
):
    
    """
    Returns a binary mask denoting which pixels to compare across images (0=ingore, 1=compare)
    Determined from PLD attributes to define lakes, land, and shoreline at different buffer values.
    """
    desc = f'buffered_{buffer_delim}m'
    pld_mask = read_band_by_description(pld_path, desc, image_window_params)

    measure_mask = np.zeros_like(pld_mask)
    # Measures inside the lake polygon
    if zone == 'lake':
        measure_mask = np.where(pld_mask == 1, 1, measure_mask)
    # Measures outside the lake polygon
    elif zone == 'land':
        measure_mask = np.where(pld_mask == 1, measure_mask, 1)
    # Measures outside the lake polygon but within the buffer
    elif zone == 'shoreline':
        if buffer_delim_outer is None:
            raise ValueError("buffer_delim_outer param required for shoreline zone")
        outer_desc = f'buffered_{buffer_delim_outer}m'
        # Read the outer buffer mask
        outer_mask = read_band_by_description(pld_path, outer_desc, image_window_params)
        measure_mask = np.where((pld_mask == 0) & (outer_mask == 1),
                                1,
                                measure_mask)
    else:
        raise ValueError("Zone must be 'lake', 'land' or 'shoreline'")

    return measure_mask

def find_measured_pixels(
    arr1_data: np.array,
    arr2_data: np.array,
    measure_mask: np.array
):
    """
    Returns two numpy arrays masked to measure specific regions (land, lake, or shoreline)
    Opperations:
    1) Selects shoreline, lake, or land pixels within measure mask
    2) Ensures both images have common set of nans
    """

    arr1_masked = apply_measure_mask(arr1_data, measure_mask)
    arr2_masked = apply_measure_mask(arr2_data, measure_mask)
    valid_arr1_mask = ~np.isnan(arr1_masked)
    valid_arr2_mask = ~np.isnan(arr2_masked)

    # Make the same nan values from filtering common to each dataset
    arr1_data_out = np.where(valid_arr2_mask, arr1_masked, np.nan)
    arr2_data_out = np.where(valid_arr1_mask, arr2_masked, np.nan)

    return arr1_data_out, arr2_data_out

def calc_ndwi(
        green_array: np.array, 
        nir_array: np.array
    ):
    """
    Given the Green and NIR data arrays, the function calculates the NDWI array
    """
    # Should be no negative values, but just in case
    green_array_calc = np.where(green_array <= 0, np.nan, green_array)
    nir_array_calc = np.where(nir_array <= 0, np.nan, nir_array)
    
    # # Extract valid Green values for plotting
    # valid_green = green_array_calc[~np.isnan(green_array_calc)]
    # plt.hist(valid_green, bins=100, edgecolor='black')
    # plt.xlabel('Green values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Valid Green Values')
    # plt.show()
    
    # # Extract valid NIR values for plotting
    # valid_nir = nir_array_calc[~np.isnan(nir_array_calc)]
    # plt.hist(valid_nir, bins=100, edgecolor='black')
    # plt.xlabel('NIR values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Valid NIR Values')
    # plt.show()

    # Calculate NDWI
    ndwi_array = np.divide(
        (green_array_calc - nir_array_calc),
        (green_array_calc + nir_array_calc),
        out=np.full_like(green_array_calc, np.nan, dtype=float), 
        where=(green_array_calc + nir_array_calc) != 0  # Boolean mask for pixels to perform division
    )

    return ndwi_array

def ndwi_images_vis(
    arr1_ndwi: np.array,
    arr2_ndwi: np.array,
    arr1_ndwi_title: str,
    arr2_ndwi_title: str
):
    # Find global min and max values across both arrays
    vmin = min(np.nanmin(arr1_ndwi), np.nanmin(arr2_ndwi))
    vmax = max(np.nanmax(arr1_ndwi), np.nanmax(arr2_ndwi))
    
    # Plot the NDWI images with the same color scale
    green_white_blue = LinearSegmentedColormap.from_list("GreenWhiteBlue", ["green", "white", "blue"])
    
    fig, ax = plt.subplots()
    ax.set_facecolor('darkgrey')
    im1 = ax.imshow(arr1_ndwi, cmap=green_white_blue, vmin=vmin, vmax=vmax)
    ax.set_title(f'{arr1_ndwi_title} NDWI')
    plt.colorbar(im1, ax=ax)
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_facecolor('darkgrey')
    im2 = ax.imshow(arr2_ndwi, cmap=green_white_blue, vmin=vmin, vmax=vmax)
    ax.set_title(f'{arr2_ndwi_title} NDWI')
    plt.colorbar(im2, ax=ax)
    plt.show()

def make_sat_ndwi_images(
    image_info: dict
):
    """
    Takes the file paths for two coincident images from Sentinel-2 and Landsat8
    Returns two NDWI images as numpy arrays. 
    Plots the NDWI images for visual inspection.
    NOTE: For not resampled images, the process is dramatically different. 
    """
    
    level, date, roi, resample_method = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['resample_method']
    )

    if resample_method != 'noresample':
        s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
        ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'

        ls_green, s2_green, image_window_params = rio_get_data_arrays_with_common_trans(
            ls8_fp, s2_fp, 'Green'
        )
        ls_nir, s2_nir, image_window_params = rio_get_data_arrays_with_common_trans(
            ls8_fp, s2_fp, 'NIR'
        )
        # Calculate NDWI
        ls_ndwi = calc_ndwi(ls_green, ls_nir)
        s2_ndwi = calc_ndwi(s2_green, s2_nir)
        # Redundant check to ensure image have the exact same common nan mask
        valid_ls_mask = ~np.isnan(ls_ndwi)
        valid_s2_mask = ~np.isnan(s2_ndwi)
        ls_ndwi_out = np.where(valid_s2_mask, ls_ndwi, np.nan)
        s2_ndwi_out = np.where(valid_ls_mask, s2_ndwi, np.nan)

    elif resample_method == 'noresample': 
        image_window_params = None # Don't need image window params, becuase PLD mask needs to be reprojected to image's native tile
        s2_fp = f'./data/{level}_images/roi_{roi}_noresample/Sentinel2_{level}_date_{date}_roi_{roi}.tif'
        ls8_fp = f'./data/{level}_images/roi_{roi}_noresample/Landsat8_{level}_date_{date}_roi_{roi}.tif'

        ls_green, s2_green = rio_get_data_arrays_native_trans(
            ls_fp=ls8_fp, s2_fp=s2_fp, band_name='Green'
        )
        ls_nir, s2_nir = rio_get_data_arrays_native_trans(
            ls_fp=ls8_fp, s2_fp=s2_fp, band_name="NIR"
        )

        # Calculate NDWI
        ls_ndwi_out = calc_ndwi(ls_green, ls_nir)
        s2_ndwi_out = calc_ndwi(s2_green, s2_nir)
    else:
        raise ValueError('ERROR: need to specify proper resampling method')

    # Quick plot of NDWI images
    #ndwi_images_vis(s2_ndwi_out, ls_ndwi_out, f'Sentinel-2 {level}', f'Landsat8 {level}')
    
    return ls_ndwi_out, s2_ndwi_out, image_window_params

def make_ac_ndwi_images(
    image_info: dict
):
    sat, roi, date, resample_method = (
        image_info['satellite'],
        image_info['roi'],
        image_info['date'],
        image_info['resample_method']
    )

    toa_fp = f'./data/toa_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_{sat}_toa_date_{date}_roi_{roi}.tif'
    sr_fp = f'./data/sr_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_{sat}_sr_date_{date}_roi_{roi}.tif'

    toa_green, sr_green, image_window_params = rio_get_ac_arrays(toa_fp, sr_fp, band_name='Green')
    toa_nir, sr_nir, _ = rio_get_ac_arrays(toa_fp, sr_fp, band_name='NIR')

    toa_ndwi = calc_ndwi(toa_green, toa_nir)
    sr_ndwi = calc_ndwi(sr_green, sr_nir)

    valid_toa_mask = ~np.isnan(toa_ndwi)
    valid_sr_mask = ~np.isnan(sr_ndwi)
    toa_ndwi_out = np.where(valid_sr_mask, toa_ndwi, np.nan)
    sr_ndwi_out = np.where(valid_toa_mask, sr_ndwi, np.nan)

    # Quick plot of NDWI images
    #ndwi_images_vis(toa_ndwi_out, sr_ndwi_out, f'TOA {sat}', f'SR {sat}')

    return toa_ndwi_out, sr_ndwi_out, image_window_params

def rio_get_ac_arrays(
    toa_path: str,
    sr_path: str,
    band_name: str
):

    toa_data = read_band_by_description(toa_path, band_name, image_window_params=None)
    sr_data = read_band_by_description(sr_path, band_name, image_window_params=None)

    toa_data = toa_data.copy()
    sr_data = sr_data.copy()

    # Zero values should already be nan, but just incase
    toa_data = np.where(toa_data >= 0, toa_data, np.nan)
    sr_data = np.where(sr_data >= 0, sr_data, np.nan)

    if toa_data.shape != sr_data.shape:
        raise ValueError(
        f"Incompatible array shapes: TOA shape {toa_data.shape} != "
        f"SR shape {sr_data.shape}."
    )

    with rio.open(toa_path) as src:
        meta = src.meta
        bounds = src.bounds
        transform = src.transform
        shape = src.shape

    image_window_params = {
        'bounds': bounds,
        'transform': transform,
        'shape': shape
    }

    return toa_data, sr_data, image_window_params