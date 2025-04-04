"""
Water Area Calculation Functions:
These calculate the water fractions for different zones (total landscape, lakes and shorelines) according to the PLD mask
Uses to related histogram-based methods 1) Otsu thresholding 2) Adaptive thresholding
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.warp import reproject, Resampling

# ------ Custom Functions ------
from img_data_fetching_functions import (
    check_match_imgs,
    make_measure_mask,
    make_ndwi_images
)

# Small helper function
def numpy_to_list(data):
    """
    Converts numpy arrays to lists for storage in a dataframe
    Useful for storing image NDWI histograms
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

"""
-----------------------------------------
Functions to collect data
-----------------------------------------
"""

def mask_ndwi_images_on_common_grid(
    ndwi: np.array,
    image_window_params: dict,
    roi: str,
    res: int
):
    """
    This masks NDWI images to lakes with a 60m buffer.
    The histogram based methods perform best when thresholds are derived over lake mask with some land.
    This function is used when calculating the Otsu thresholds for water area.
    """

    pld_path = f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif'
    pld_plus = make_measure_mask(pld_path, 
                                 image_window_params, 
                                 zone='lake', 
                                 buffer_delim=60, 
                                 buffer_delim_outer=None
    )

    ndwi_masked = ndwi[pld_plus == 1]

    return ndwi_masked

def mask_ndwi_images_native_grid(
    ndwi: np.array,
    image_path: str,
    roi: str,
):
    """
    Generates an NDWI image in the native Landsat8 or Sentinel-2 tile grid. 
    The image is masked to PLD + 60 meters for thresholding algorithms. 
    PLD +60 raster is reprojected onto the image's grid. 
    """
    pld_path = f'./data/pld_rasterized/{roi}_lake_masks_res30.tif' 

    with rio.open(image_path) as tgt, rio.open(pld_path) as src:
        src_data = src.read(6) # NOTE: Hard-coded the band index for lakes buffered by 60 meters.
        pld_reproj = np.zeros((tgt.height, tgt.width), dtype=src.dtypes[0])

        reproject(
            source=src_data,
            destination=pld_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )

    ndwi_masked = ndwi[pld_reproj == 1]

    return ndwi_masked
    
def clean_ndwi_data(
    ndwi_data: np.array
):
    """
    Removes NaN values from NDWI before computing Otsu Thresholds or Histograms
    """
    flat_ndwi = ndwi_data.flatten()
    valid_mask = ~np.isnan(flat_ndwi)
    valid_data = flat_ndwi[valid_mask]
    # Extra check to ensure NDWI values are valid (between -1 and 1)
    valid_data = np.clip(valid_data, -1, 1)
    
    return valid_data

"""
-----------------------------------------
Functions to determine land/water thresholds 
with otsu and adaptive methods
-----------------------------------------
"""

def find_otsu_threshold(
    ndwi: np.array, 
    show_hist: bool,
):
    """
    Input: NDWI array (-1,1)
    Output: Otsu Threshold for a given NDWI image
    """
    
    valid_data = clean_ndwi_data(ndwi) # NDWI data is now flat

    n_bins = 500
    hist, bin_edges = np.histogram(valid_data, bins=n_bins, range=(-1, 1))
    total_pixels = hist.sum()
    pdf = hist / total_pixels
    cumulative_prob = np.cumsum(pdf)               
    cumulative_intensity = np.cumsum(pdf * np.arange(n_bins))
    best_threshold = 0
    best_variance = 0

    for t in range(n_bins):
        # Probability of class 0
        w0 = cumulative_prob[t]
        # Probability of class 1
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            # This means all data is on one side of the threshold => skip
            continue
        # The mean intensity of the two classes
        m0 = cumulative_intensity[t] / w0
        m1 = (cumulative_intensity[-1] - cumulative_intensity[t]) / w1

        # Between-class variance
        var_between = w0 * w1 * (m0 - m1) ** 2

        if var_between > best_variance:
            best_variance = var_between
            best_threshold = t

    # Converts bin edges to NDWI value (in the middle)
    threshold = 0.5 * (bin_edges[best_threshold] + bin_edges[best_threshold + 1])

    if show_hist == True:
        plt.hist(valid_data, bins=50, edgecolor='black')
        plt.axvline(x=threshold, color='red', label=f'Threshold = {threshold}')
        plt.xlabel('NDWI values')
        plt.legend()
        plt.show()
    
    return threshold, (hist, bin_edges)

def find_adaptive_thresholds(
    hist: np.histogram, # Generated from np.histogram
    otsu_threshold: float,
    show_hist: bool
):
    """
    Uses the Otsu threshold and histograms to calculate the adpative threshold
    """
    
    # STEP 1: Divide the NDWI histogram into Land and Water classes based on Otsu threshold
    counts, bin_edges = hist
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    idx_otsu = np.digitize(otsu_threshold, bin_edges)
    # Boolean mask on bin centers
    water_class = bin_centers > otsu_threshold
    land_class = bin_centers < otsu_threshold

    # Bins and counts for each class
    water_counts = counts[water_class]
    water_bins = bin_centers[water_class]
    land_counts = counts[land_class]
    land_bins = bin_centers[land_class]
    
    # STEP 2: Find the 1)Peaks and 2)Prominence of the water and land histograms
    otsu_height = counts[idx_otsu]
    water_peak_height = np.max(water_counts)
    idx_water_peak = np.argmax(water_counts) # NOTE: argmax returns the first if two equal peaks
    water_peak_val = water_bins[idx_water_peak]
    water_prominence = water_peak_height - otsu_height

    land_peak_height = np.max(land_counts)
    idx_land_peak = np.argmax(land_counts) # NOTE: argmax returns the first if two equal peaks
    land_peak_val = land_bins[idx_land_peak]
    land_prominence = land_peak_height - otsu_height

    # STEP 3: Find NDWI (Land/Lower) and (Water/Upper)
    land_h_val = land_peak_height - (0.9 * land_prominence)
    land_ndwi = land_peak_val
    # Travel right through land_bins from land_peak_val 
    # until the count (histogram height) is less than land_h_val
    for i in range(idx_land_peak, len(land_bins)):
        if i >= len(land_counts) or land_counts[i] < land_h_val:
            land_ndwi = land_bins[i-1] if i > 0 else land_bins[0]
            break
    water_h_val = water_peak_height - (0.9 * water_prominence)
     # Travel left through water_bins from water_peak_val 
     # until the count (histogram height) is less than water_h_val
    water_ndwi = water_peak_val
    for i in range(idx_water_peak, -1, -1): # Go backwards (left)
        if i < 0 or water_counts[i] < water_h_val:
            water_ndwi = water_bins[i+1] if i < len(water_bins) - 1 else water_bins[-1]
            break
    
    if show_hist:
        plt.figure(figsize=(8,6))
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
        plt.axvline(x=otsu_threshold, color='green', linestyle='--', label='Otsu Threshold')
        plt.axvline(x=water_peak_val, color='blue', linestyle='-', label=f'Water Peak: {water_peak_val:.2f}')
        plt.axvline(x=land_peak_val, color='brown', linestyle='-', label=f'Land Peak: {land_peak_val:.2f}')
        plt.axhline(y=otsu_height, color='green', linestyle='--', label='Otsu Height')
        plt.axhline(y=water_prominence, color='blue', linestyle='-', label=f'Water Prominence: {water_prominence:.2f}')
        plt.axhline(y=land_prominence, color='brown', linestyle='-', label=f'Land Prominence: {land_prominence:.2f}')
        plt.axvline(x=land_ndwi, color='brown', linestyle=':', label=f'Land NDWI: {land_ndwi:.2f}')
        plt.axvline(x=water_ndwi, color='blue', linestyle=':', label=f'Water NDWI: {water_ndwi:.2f}')
        plt.legend()
        plt.xlabel('NDWI values')
        plt.show()

    return land_ndwi, water_ndwi


"""
-----------------------------------------
Functions calculate water fractions for different parts of the landscape
-----------------------------------------
"""

def calc_total_wtr_frac(
    water_mask: np.array, # Binary water mask from an image 
    ndwi: float # The original NDWI image (to check for valid pixels)
):
    """
    Calculates the water fraction relative to the number of valid NDWI pixels in the original image
    # TODO: Modify to explicitly address shorelines???
    """
    water_pixels = np.sum(water_mask == 1)
    valid_pixels = np.sum(~np.isnan(ndwi))
    water_frac = water_pixels / valid_pixels * 100

    return water_frac

def calc_lake_wtr_frac_common_grid(
    water_mask: np.array, # Binary water mask from an image
    ndwi: float, # The original NDWI image (to check for valid pixels)
    pld_fp: str, 
    image_window_params: dict, # The window parameters for reading the PLD mask (georeferenced properly)
    buff_lake: bool
):
    """
    Calculates the water fraction inside the PLD lake mask
    """
    if buff_lake:
        buffer_delim = 60
    else:
        buffer_delim = 0

    pld_mask = make_measure_mask(
        pld_fp, 
        image_window_params, 
        zone='lake', 
        buffer_delim=buffer_delim, 
        buffer_delim_outer=None
    )

    lake_water_pixels = np.sum((water_mask == 1) & (pld_mask == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (pld_mask == 1))
    water_frac = lake_water_pixels / valid_pixels * 100

    return water_frac

def calc_lake_wtr_frac_native_tile(
    img_fp: str,
    water_mask: np.array, # Binary water mask from an image
    ndwi: float, # The original NDWI image (to check for valid pixels)
    pld_fp: str,
    buff_lake: bool
):
    """
    Calculates the water fraction inside the PLD lake mask
    First reprojects the PLD mask into the native grid of the image
    """
    # NOTE: Hard-coded the band index for lakes (PLD + 0 meters)
    if buff_lake:
        band_idx = 6 # PLD + 60 meters
    else:
        band_idx = 4 # PLD + 0 meters

    # Reproject the PLD mask into the native grid of the image
    with rio.open(img_fp) as tgt, rio.open(pld_fp) as src:
        src_data = src.read(band_idx) 
        pld_lake_reproj = np.zeros((tgt.height, tgt.width), dtype=src.dtypes[0])

        reproject(
            source=src_data,
            destination=pld_lake_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )
    
    lake_water_pixels = np.sum((water_mask == 1) & (pld_lake_reproj == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (pld_lake_reproj == 1))
    water_frac = lake_water_pixels / valid_pixels * 100

    return water_frac


def calc_shoreline_wtr_frac_common_grid(
    water_mask: np.array, # Binary water mask from an image
    ndwi: np.array, # The original NDWI image (to check for valid pixels)
    pld_fp: str, 
    image_window_params: dict, # The window parameters for reading the PLD mask (georeferenced properly)
):
    """
    Calculates the water fraction inside the shoreline zone (-60m to +60m)
    """
    shoreline_mask = make_measure_mask(
        pld_fp, 
        image_window_params, 
        zone='shoreline', 
        buffer_delim=-60, 
        buffer_delim_outer=60
    )

    shoreline_water_pixels = np.sum((water_mask == 1) & (shoreline_mask == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (shoreline_mask == 1))
    water_frac = shoreline_water_pixels / valid_pixels * 100

    return water_frac

def calc_shoreline_wtr_frac_native_tile(
    img_fp: str, # The filepath to the image to get reprojection info for PLD
    water_mask: np.array, # Binary water mask from an image
    ndwi: np.array, # The original NDWI image (to check for valid pixels)
    roi: str
):
    
    """
    Calculates the water fraction inside the shoreline zone (-60m to +60m)
    First reprojects the PLD mask into the native grid of the image
    """
    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res30.tif'
    with rio.open(img_fp) as tgt, rio.open(pld_fp) as src:
        src_lakes_outer = src.read(6) # NOTE: Hard-coded the band index 
        src_lakes_inner = src.read(3) 
        pld_outer_reproj = np.zeros((tgt.height, tgt.width), dtype=src.dtypes[0])
        pld_inner_reproj = np.zeros((tgt.height, tgt.width), dtype=src.dtypes[0])
        # Reproject the shoreline's outer band
        reproject(
            source=src_lakes_outer,
            destination=pld_outer_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )
        # Reproject the shoreline's inner band
        reproject(
            source=src_lakes_inner,
            destination=pld_inner_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tgt.transform,
            dst_crs=tgt.crs,
            resampling=Resampling.nearest
        )
    # Shoreline mask is the outer band - inner band
    pld_shoreline_reproj = (pld_outer_reproj == 1) & (pld_inner_reproj != 1)
    
    shoreline_water_pixels = np.sum((water_mask == 1) & (pld_shoreline_reproj == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (pld_shoreline_reproj == 1))
    water_frac = shoreline_water_pixels / valid_pixels * 100

    return water_frac

"""
-----------------------------------------
Functions to gather water fractions for different parts of the landscape
Using different functions for resampled images vs. native tiles
Could refactor to reduce redundancy if there's time
-----------------------------------------
"""
def lake_and_shoreline_frac_native_tile(
    # Binary water masks
    ls_water_otsu: np.array, 
    s2_water_otsu: np.array,
    ls_water_adaptive: np.array,
    s2_water_adaptive: np.array,
    # Original NDWI images for valid pixel counts
    ls_ndwi: np.array,
    s2_ndwi: np.array,
    # roi and file paths for pld file and info to reproject pld mask into image grid
    roi: str,
    s2_fp: str,
    ls8_fp: str
):

    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res30.tif'
    # # Calculate the lake water fractions
    lake_ls_water_frac_otsu = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_otsu, ndwi=ls_ndwi, pld_fp=pld_fp, buff_lake=False)
    lake_s2_water_frac_otsu = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_otsu, ndwi=s2_ndwi, pld_fp=pld_fp, buff_lake=False)
    lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp, buff_lake=False)
    lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp, buff_lake=False)

    # Calculate buffered lake water fraction (+ 60 meters)
    buff_lake_ls_water_frac_otsu = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_otsu, ndwi=ls_ndwi, pld_fp=pld_fp, buff_lake=True)
    buff_lake_s2_water_frac_otsu = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_otsu, ndwi=s2_ndwi, pld_fp=pld_fp, buff_lake=True)
    buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp, buff_lake=True)
    buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp, buff_lake=True)

    # Calculate the shoreline water fractions
    shoreline_ls_water_frac_otsu = calc_shoreline_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_otsu, ndwi=ls_ndwi, roi=roi)
    shoreline_s2_water_frac_otsu = calc_shoreline_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_otsu, ndwi=s2_ndwi, roi=roi)
    shoreline_ls_water_frac_adaptive = calc_shoreline_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, roi=roi)
    shoreline_s2_water_frac_adaptive = calc_shoreline_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, roi=roi)

    # Calculate the fractions over "smallest" lakes
    pld_fp_smallest = f'./data/pld_rasterized/{roi}_lake_masks_res30_smallest.tif'
    smallest_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_smallest, buff_lake=False)
    smallest_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_smallest, buff_lake=True)
    smallest_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_smallest, buff_lake=False)
    smallest_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_smallest, buff_lake=True)

    pld_fp_small = f'./data/pld_rasterized/{roi}_lake_masks_res30_small.tif'
    small_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_small, buff_lake=False)
    small_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_small, buff_lake=True)
    small_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_small, buff_lake=False)
    small_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_small, buff_lake=True)

    pld_fp_medium = f'./data/pld_rasterized/{roi}_lake_masks_res30_medium.tif'
    medium_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_medium, buff_lake=False)
    medium_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_medium, buff_lake=True)
    medium_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_medium, buff_lake=False)
    medium_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_medium, buff_lake=True)

    pld_fp_large = f'./data/pld_rasterized/{roi}_lake_masks_res30_large.tif'
    large_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_large, buff_lake=False)
    large_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=ls8_fp, water_mask=ls_water_adaptive, ndwi=ls_ndwi, pld_fp=pld_fp_large, buff_lake=True)
    large_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_large, buff_lake=False)
    large_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_native_tile(img_fp=s2_fp, water_mask=s2_water_adaptive, ndwi=s2_ndwi, pld_fp=pld_fp_large, buff_lake=True)


    return {
        'lake_ls_water_frac_otsu': lake_ls_water_frac_otsu,
        'lake_s2_water_frac_otsu': lake_s2_water_frac_otsu,
        'lake_ls_water_frac_adaptive': lake_ls_water_frac_adaptive,
        'lake_s2_water_frac_adaptive': lake_s2_water_frac_adaptive,
        'shoreline_ls_water_frac_otsu': shoreline_ls_water_frac_otsu,
        'shoreline_s2_water_frac_otsu': shoreline_s2_water_frac_otsu,
        'shoreline_ls_water_frac_adaptive': shoreline_ls_water_frac_adaptive,
        'shoreline_s2_water_frac_adaptive': shoreline_s2_water_frac_adaptive,
        # Buffered lake water fractions
        'buff_lake_ls_water_frac_otsu': buff_lake_ls_water_frac_otsu,
        'buff_lake_s2_water_frac_otsu': buff_lake_s2_water_frac_otsu,
        'buff_lake_ls_water_frac_adaptive': buff_lake_ls_water_frac_adaptive,
        'buff_lake_s2_water_frac_adaptive': buff_lake_s2_water_frac_adaptive,
        # Smallest lakes
        'smallest_lake_ls_water_frac_adaptive': smallest_lake_ls_water_frac_adaptive,
        'smallest_buff_lake_ls_water_frac_adaptive': smallest_buff_lake_ls_water_frac_adaptive,
        'smallest_lake_s2_water_frac_adaptive': smallest_lake_s2_water_frac_adaptive,
        'smallest_buff_lake_s2_water_frac_adaptive': smallest_buff_lake_s2_water_frac_adaptive,
        # Small lakes
        'small_lake_ls_water_frac_adaptive': small_lake_ls_water_frac_adaptive,
        'small_buff_lake_ls_water_frac_adaptive': small_buff_lake_ls_water_frac_adaptive,
        'small_lake_s2_water_frac_adaptive': small_lake_s2_water_frac_adaptive,
        'small_buff_lake_s2_water_frac_adaptive': small_buff_lake_s2_water_frac_adaptive,
        # Medium lakes
        'medium_lake_ls_water_frac_adaptive': medium_lake_ls_water_frac_adaptive,
        'medium_buff_lake_ls_water_frac_adaptive': medium_buff_lake_ls_water_frac_adaptive,
        'medium_lake_s2_water_frac_adaptive': medium_lake_s2_water_frac_adaptive,
        'medium_buff_lake_s2_water_frac_adaptive': medium_buff_lake_s2_water_frac_adaptive,
        # Large lakes
        'large_lake_ls_water_frac_adaptive': large_lake_ls_water_frac_adaptive,
        'large_buff_lake_ls_water_frac_adaptive': large_buff_lake_ls_water_frac_adaptive,
        'large_lake_s2_water_frac_adaptive': large_lake_s2_water_frac_adaptive,
        'large_buff_lake_s2_water_frac_adaptive': large_buff_lake_s2_water_frac_adaptive
    }


def lake_and_shoreline_frac_common_grid(
    # Binary water masks
    ls_water_otsu: np.array, 
    s2_water_otsu: np.array,
    ls_water_adaptive: np.array,
    s2_water_adaptive: np.array,
    # Original NDWI images for valid pixel counts
    ls_ndwi: np.array,
    s2_ndwi: np.array,
    # Items to read the appropriate PLD mask, already on the same grid as images. 
    roi: str,
    res: int, 
    image_window_params: dict
):
    """
    Calculates the water fractions for different parts of the landscape and lake sizes. 
    Only intended common grid images (i.e., resampled), doesn't work on native tiles.
    ?Maybe worth refactoring for less reducdancy?
    """

    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif'
    # # Calculate the lake water fractions
    lake_ls_water_frac_otsu = calc_lake_wtr_frac_common_grid(ls_water_otsu, ls_ndwi, pld_fp, image_window_params, buff_lake=False)
    lake_s2_water_frac_otsu = calc_lake_wtr_frac_common_grid(s2_water_otsu, s2_ndwi, pld_fp, image_window_params, buff_lake=False)
    lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp, image_window_params, buff_lake=False)
    lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp, image_window_params, buff_lake=False)
    # Calculate buffered lake water fraction (+ 60 meters)
    buff_lake_ls_water_frac_otsu = calc_lake_wtr_frac_common_grid(ls_water_otsu, ls_ndwi, pld_fp, image_window_params, buff_lake=True)
    buff_lake_s2_water_frac_otsu = calc_lake_wtr_frac_common_grid(s2_water_otsu, s2_ndwi, pld_fp, image_window_params, buff_lake=True)
    buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp, image_window_params, buff_lake=True)
    buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp, image_window_params, buff_lake=True)

    # Calculate the shoreline water fractions
    shoreline_ls_water_frac_otsu = calc_shoreline_wtr_frac_common_grid(ls_water_otsu, ls_ndwi, pld_fp, image_window_params)
    shoreline_s2_water_frac_otsu = calc_shoreline_wtr_frac_common_grid(s2_water_otsu, s2_ndwi, pld_fp, image_window_params)
    shoreline_ls_water_frac_adaptive = calc_shoreline_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp, image_window_params)
    shoreline_s2_water_frac_adaptive = calc_shoreline_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp, image_window_params)

    # Calculate the fractions over "smallest" lakes
    pld_fp_smallest = f'./data/pld_rasterized/{roi}_lake_masks_res{res}_smallest.tif'
    smallest_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp_smallest, image_window_params, buff_lake=False)
    smallest_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp_smallest, image_window_params, buff_lake=True)
    smallest_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp_smallest, image_window_params, buff_lake=False)
    smallest_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp_smallest, image_window_params, buff_lake=True)

    pld_fp_small = f'./data/pld_rasterized/{roi}_lake_masks_res{res}_smallest.tif'
    small_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp_small, image_window_params, buff_lake=False)
    small_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_fp_small, image_window_params, buff_lake=True)
    small_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp_small, image_window_params, buff_lake=False)
    small_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_fp_small, image_window_params, buff_lake=True)

    pld_medium = f'./data/pld_rasterized/{roi}_lake_masks_res{res}_medium.tif'
    medium_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_medium, image_window_params, buff_lake=False)
    medium_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_medium, image_window_params, buff_lake=True)
    medium_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_medium, image_window_params, buff_lake=False)
    medium_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_medium, image_window_params, buff_lake=True)

    pld_large = f'./data/pld_rasterized/{roi}_lake_masks_res{res}_large.tif'
    large_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_large, image_window_params, buff_lake=False)
    large_buff_lake_ls_water_frac_adaptive = calc_lake_wtr_frac_common_grid(ls_water_adaptive, ls_ndwi, pld_large, image_window_params, buff_lake=True)
    large_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_large, image_window_params, buff_lake=False)
    large_buff_lake_s2_water_frac_adaptive = calc_lake_wtr_frac_common_grid(s2_water_adaptive, s2_ndwi, pld_large, image_window_params, buff_lake=True)

    
    return {
        'lake_ls_water_frac_otsu': lake_ls_water_frac_otsu,
        'lake_s2_water_frac_otsu': lake_s2_water_frac_otsu,
        'lake_ls_water_frac_adaptive': lake_ls_water_frac_adaptive,
        'lake_s2_water_frac_adaptive': lake_s2_water_frac_adaptive,
        'shoreline_ls_water_frac_otsu': shoreline_ls_water_frac_otsu,
        'shoreline_s2_water_frac_otsu': shoreline_s2_water_frac_otsu,
        'shoreline_ls_water_frac_adaptive': shoreline_ls_water_frac_adaptive,
        'shoreline_s2_water_frac_adaptive': shoreline_s2_water_frac_adaptive,
        # Buffered lake water fractions
        'buff_lake_ls_water_frac_otsu': buff_lake_ls_water_frac_otsu,
        'buff_lake_s2_water_frac_otsu': buff_lake_s2_water_frac_otsu,
        'buff_lake_ls_water_frac_adaptive': buff_lake_ls_water_frac_adaptive,
        'buff_lake_s2_water_frac_adaptive': buff_lake_s2_water_frac_adaptive,
        # Smallest lakes
        'smallest_lake_ls_water_frac_adaptive': smallest_lake_ls_water_frac_adaptive,
        'smallest_buff_lake_ls_water_frac_adaptive': smallest_buff_lake_ls_water_frac_adaptive,
        'smallest_lake_s2_water_frac_adaptive': smallest_lake_s2_water_frac_adaptive,
        'smallest_buff_lake_s2_water_frac_adaptive': smallest_buff_lake_s2_water_frac_adaptive,
        # Small lakes
        'small_lake_ls_water_frac_adaptive': small_lake_ls_water_frac_adaptive,
        'small_buff_lake_ls_water_frac_adaptive': small_buff_lake_ls_water_frac_adaptive,
        'small_lake_s2_water_frac_adaptive': small_lake_s2_water_frac_adaptive,
        'small_buff_lake_s2_water_frac_adaptive': small_buff_lake_s2_water_frac_adaptive,
        # Medium lakes
        'medium_lake_ls_water_frac_adaptive': medium_lake_ls_water_frac_adaptive,
        'medium_buff_lake_ls_water_frac_adaptive': medium_buff_lake_ls_water_frac_adaptive,
        'medium_lake_s2_water_frac_adaptive': medium_lake_s2_water_frac_adaptive,
        'medium_buff_lake_s2_water_frac_adaptive': medium_buff_lake_s2_water_frac_adaptive,
        # Large lakes
        'large_lake_ls_water_frac_adaptive': large_lake_ls_water_frac_adaptive,
        'large_buff_lake_ls_water_frac_adaptive': large_buff_lake_ls_water_frac_adaptive,
        'large_lake_s2_water_frac_adaptive': large_lake_s2_water_frac_adaptive,
        'large_buff_lake_s2_water_frac_adaptive': large_buff_lake_s2_water_frac_adaptive
    }

def write_mask_rasters(
    ls_water: np.array,
    s2_water: np.array,
    image_info: dict
):
    """
    Writes the binary water masks to disk

    """
    roi_name = image_info['roi']
    level = image_info['level']
    date = image_info['date']
    resample_method = image_info['resample_method']

    out_dir = './data/processed_water_masks/'
    ref_fp = f'./data/roi_shapes/rois/rasterized_{roi_name}_shape_res30.tif'
    ls_out_fp = f'{out_dir}LS8_water_mask_{level}_{roi_name}_{date}_{resample_method}.tif'
    s2_out_fp = f'{out_dir}S2_water_mask_{level}_{roi_name}_{date}_{resample_method}.tif'

    with rio.open(ref_fp) as ref:
        meta = ref.meta.copy()

    with rio.open(ls_out_fp, 'w', **meta) as dst:
        dst.write(ls_water.astype(rio.uint8), 1)
    
    with rio.open(s2_out_fp, 'w', **meta) as dst:
        dst.write(s2_water.astype(rio.uint8), 1)

    print(f'Wrote {ls_out_fp}')
    print(f'Wrote {s2_out_fp}')

    

"""
-----------------------------------------
Function contains the following steps:
1) Determine if calculations are on common resampled grid or the native tiles
2) Check if the images are valid (enough pixels)
3) Calculate the Otsu and adaptive thresholds for each image pair
4) Make binary water masks using the thresholds
5) Calculates the water fractions for each image pair along different PLD zones
-----------------------------------------
"""

def image_wtr_area(
    image_info: dict,
    write_mask: bool,
    hist_return: bool, 
    write_rasters: bool,
):
    """
    Returns a dictionary with the otsu threshold, and water fraction from each image
    """
    # Initialize all variables at the start
    # Thresholds
    ls_otsu_threshold = None
    ls_adaptive_land = None
    ls_adaptive_water = None
    s2_otsu_threshold = None
    s2_adaptive_land = None
    s2_adaptive_water = None
    # Otsu water fractions
    total_ls_water_frac_otsu = None 
    total_s2_water_frac_otsu = None
    # Adaptive water fractions
    total_ls_water_frac_adaptive = None
    total_s2_water_frac_adaptive = None

    # Histograms
    ls_hist = None
    s2_hist = None

    level, date, roi, band_name, resample_method = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['band_name'],
        image_info['resample_method']
    )
    
    # Images are resampled to a common grid with PLD
    if resample_method != 'noresample':
        s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
        ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'
        res = re.search(r"(\d{2}$)", resample_method).group(1) # Gets the resolution digits from resample method
        if res == '30':
            s2_valid_threshold = ls_valid_threshold = 25_000
        elif res == '60':
            s2_valid_threshold = ls_valid_threshold = 6_250
        else:
            print("ERROR: Invalid resolution, must be 30 or 60")
        if not check_match_imgs(ls8_fp, s2_fp):
            return None
    
        ls_ndwi, s2_ndwi, image_window_params = make_ndwi_images(image_info)
        ls_ndwi_lakes = mask_ndwi_images_on_common_grid(ls_ndwi, image_window_params, roi, res)
        s2_ndwi_lakes = mask_ndwi_images_on_common_grid(s2_ndwi, image_window_params, roi, res)

    # Noresample requires bringing PLD into the Landsat8 and Sentinel-2 tiles grid.
    else: 
        s2_fp = f'./data/{level}_images/roi_{roi}_noresample/Sentinel2_{level}_date_{date}_roi_{roi}.tif'
        ls8_fp = f'./data/{level}_images/roi_{roi}_noresample/Landsat8_{level}_date_{date}_roi_{roi}.tif'
        res = 'native'
        ls_valid_threshold = 25_000
        s2_valid_threshold = 225_000
        if not check_match_imgs(ls8_fp, s2_fp):
            return None
        ls_ndwi, s2_ndwi, image_window_params = make_ndwi_images(image_info)
        ls_ndwi_lakes = mask_ndwi_images_native_grid(
            ls_ndwi, ls8_fp, roi
        )
        s2_ndwi_lakes = mask_ndwi_images_native_grid(
            s2_ndwi, s2_fp, roi
        )
    print(f"Working {level} for {date} over the {roi} region")
    # CHECK: ensure there's enough quality lake pixels in the image
    if np.sum(~np.isnan(ls_ndwi_lakes)) < ls_valid_threshold or np.sum(~np.isnan(s2_ndwi_lakes)) < s2_valid_threshold:
        print('ERROR: Skipping water area calculations -- Bad Image')
        bad_val = "Poor Quality Image Data"
        results = bad_val
        return results
    # Valid pixel check passed
    else:
        # Find otsu and adaptive thresholds
        print("----- LandSat Histogram --------------")
        ls_otsu_threshold, ls_hist = find_otsu_threshold(ls_ndwi_lakes, show_hist=False)
        ls_adaptive_land, ls_adaptive_water = find_adaptive_thresholds(ls_hist, ls_otsu_threshold, show_hist=False)
        print("----- Sentinel-2 Histogram --------------")
        s2_otsu_threshold, s2_hist = find_otsu_threshold(s2_ndwi_lakes, show_hist=False)
        s2_adaptive_land, s2_adaptive_water = find_adaptive_thresholds(s2_hist, s2_otsu_threshold, show_hist=False)

        # Make binary water masks using the thresholds
        ls_water_otsu = (ls_ndwi > ls_otsu_threshold).astype(int)
        s2_water_otsu = (s2_ndwi > s2_otsu_threshold).astype(int)
        ls_water_adaptive_array = ((ls_ndwi - ls_adaptive_land) / (ls_adaptive_water - ls_adaptive_land))
        s2_water_adaptive_array = ((s2_ndwi - s2_adaptive_land) / (s2_adaptive_water - s2_adaptive_land))
        ls_water_adaptive = (ls_water_adaptive_array > 0.75).astype(int)
        s2_water_adaptive = (s2_water_adaptive_array > 0.75).astype(int)
                
        # Calculate the total water fractions
        total_ls_water_frac_otsu = calc_total_wtr_frac(ls_water_otsu, ls_ndwi)
        total_s2_water_frac_otsu = calc_total_wtr_frac(s2_water_otsu, s2_ndwi)
        total_ls_water_frac_adaptive = calc_total_wtr_frac(ls_water_adaptive, ls_ndwi)
        total_s2_water_frac_adaptive = calc_total_wtr_frac(s2_water_adaptive, s2_ndwi)

        # Calculate the water fractions in relation to the PLD mask (1. Lake, 2. Lake Buffered, 3. Shoreline)
        if resample_method != 'noresample':
            pld_zone_wtr_fracs = lake_and_shoreline_frac_common_grid(
                ls_water_otsu, s2_water_otsu,
                ls_water_adaptive, s2_water_adaptive,
                ls_ndwi, s2_ndwi,
                roi, res, image_window_params
            )
        else:
            pld_zone_wtr_fracs = lake_and_shoreline_frac_native_tile(
                ls_water_otsu, s2_water_otsu,
                ls_water_adaptive, s2_water_adaptive,
                ls_ndwi, s2_ndwi,
                roi, s2_fp, ls8_fp
            )

        if write_rasters == True:
            write_mask_rasters(
                ls_water=ls_water_adaptive,
                s2_water=s2_water_adaptive, 
                image_info=image_info
            )

        if hist_return == False:
            ls_hist = s2_hist = None
    
    partial_results = {
        # Thresholds
        'ls_otsu_threshold': ls_otsu_threshold,
        'ls_adaptive_land': ls_adaptive_land,
        'ls_adaptive_water': ls_adaptive_water,
        's2_otsu_threshold': s2_otsu_threshold,
        's2_adaptive_land': s2_adaptive_land,
        's2_adaptive_water': s2_adaptive_water,
        # Total Water Fractions
        'total_ls_water_frac_otsu': total_ls_water_frac_otsu,
        'total_s2_water_frac_otsu': total_s2_water_frac_otsu,
        'total_ls_water_frac_adaptive': total_ls_water_frac_adaptive,
        'total_s2_water_frac_adaptive': total_s2_water_frac_adaptive,
        # Histograms
        'ls_hist': ls_hist,
        's2_hist': s2_hist
    }
    # Combine partial results with lake and shoreline fractions
    results = {**partial_results, **pld_zone_wtr_fracs}

    return results
    
"""
-----------------------------------------
This function iterates through all the rois, processing levels (SR & TOA), and image dates
Generates a dataframe with image info, thresholds, and water fractions
-----------------------------------------
"""
    
def make_area_thresholding_summaries(
    image_info: dict, 
    levels: list, 
    rois: list, 
    dates: list,
    hist_return: bool,
    write_rasters: bool = False,
):

    """
    Takes all the levels, rois, and dates for analysis
    Returns a dataframe with metrics for each image
    """
    area_summaries = []
    resample_method = image_info['resample_method']
    for level in levels:
        image_info['level'] = level
        for roi in rois:
            image_info['roi'] = roi
            for date in dates:
                image_info['date'] = date

                # Calculate the Otsu thresholds and water fractions for each image
                area_items = image_wtr_area(
                    image_info, 
                    write_mask=False, 
                    hist_return=hist_return, 
                    write_rasters=write_rasters
                )

                if area_items is None:
                    continue
                if area_items == "Poor Quality Image Data":
                    continue

                # Convert the numpy histogram objects to lists for storage in .csv
                # NOTE: The NDWI histograms are not always returned, so check for None
                if area_items.get('ls_hist') is None:
                    ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = None
                else:
                    ls_hist_counts = numpy_to_list(area_items.get('ls_hist')[0])
                    ls_hist_bins = numpy_to_list(area_items.get('ls_hist')[1])
                    s2_hist_counts = numpy_to_list(area_items.get('s2_hist')[0])
                    s2_hist_bins = numpy_to_list(area_items.get('s2_hist')[1])

                # Print the total water fractions for each image
                print(f'LS8 Adaptive Total = {area_items.get('total_ls_water_frac_adaptive')}, S2 adaptive Total = {area_items.get('total_s2_water_frac_adaptive')}')
                print(f'LS8 Adaptive Lake = {area_items.get('lake_ls_water_frac_adaptive')}, S2 adaptive Lake = {area_items.get('lake_s2_water_frac_adaptive')}')

                summary = {
                    'date': date,
                    'roi': roi,
                    'level': level,
                    'resample_method': resample_method,
                    # Thresholds
                    'ls_otsu_threshold': area_items.get('ls_otsu_threshold'),
                    'ls_adaptive_land': area_items.get('ls_adaptive_land'),
                    'ls_adaptive_water': area_items.get('ls_adaptive_water'),
                    's2_otsu_threshold': area_items.get('s2_otsu_threshold'),
                    's2_adaptive_land': area_items.get('s2_adaptive_land'),
                    's2_adaptive_water': area_items.get('s2_adaptive_water'),

                    # Total Water Fractions
                    'total_ls_water_frac_otsu': area_items.get('total_ls_water_frac_otsu'),
                    'total_s2_water_frac_otsu': area_items.get('total_s2_water_frac_otsu'),
                    'total_ls_water_frac_adaptive': area_items.get('total_ls_water_frac_adaptive'),
                    'total_s2_water_frac_adaptive': area_items.get('total_s2_water_frac_adaptive'),

                    # Otsu Lake and Shoreline Water Fractions
                    'lake_ls_water_frac_otsu': area_items.get('lake_ls_water_frac_otsu'),
                    'shoreline_ls_water_frac_otsu': area_items.get('shoreline_ls_water_frac_otsu'),
                    'buff_lake_ls_water_frac_otsu': area_items.get('buff_lake_ls_water_frac_otsu'),                    
                    'lake_s2_water_frac_otsu': area_items.get('lake_s2_water_frac_otsu'),
                    'shoreline_s2_water_frac_otsu': area_items.get('shoreline_s2_water_frac_otsu'),
                    'buff_lake_s2_water_frac_otsu': area_items.get('buff_lake_s2_water_frac_otsu'),  

                    # Adaptive Lake and Shoreline Water Fractions
                    'lake_ls_water_frac_adaptive': area_items.get('lake_ls_water_frac_adaptive'),
                    'shoreline_ls_water_frac_adaptive': area_items.get('shoreline_ls_water_frac_adaptive'),
                    'buff_lake_ls_water_frac_adaptive': area_items.get('buff_lake_ls_water_frac_adaptive'),                    
                    'lake_s2_water_frac_adaptive': area_items.get('lake_s2_water_frac_adaptive'),
                    'shoreline_s2_water_frac_adaptive': area_items.get('shoreline_s2_water_frac_adaptive'),
                    'buff_lake_s2_water_frac_adaptive': area_items.get('buff_lake_s2_water_frac_adaptive'),                    
                    # Smallest lakes (Adaptive)
                    'smallest_lake_ls_water_frac_adaptive': area_items.get('smallest_lake_ls_water_frac_adaptive'),
                    'smallest_buff_lake_ls_water_frac_adaptive': area_items.get('smallest_buff_lake_ls_water_frac_adaptive'),
                    'smallest_lake_s2_water_frac_adaptive': area_items.get('smallest_lake_s2_water_frac_adaptive'),
                    'smallest_buff_lake_s2_water_frac_adaptive': area_items.get('smallest_buff_lake_s2_water_frac_adaptive'),                    
                    # Small lakes (Adaptive)
                    'small_lake_ls_water_frac_adaptive': area_items.get('small_lake_ls_water_frac_adaptive'),
                    'small_buff_lake_ls_water_frac_adaptive': area_items.get('small_buff_lake_ls_water_frac_adaptive'),
                    'small_lake_s2_water_frac_adaptive': area_items.get('small_lake_s2_water_frac_adaptive'),
                    'small_buff_lake_s2_water_frac_adaptive': area_items.get('small_buff_lake_s2_water_frac_adaptive'),                    
                    # Medium lakes (Adaptive)
                    'medium_lake_ls_water_frac_adaptive': area_items.get('medium_lake_ls_water_frac_adaptive'),
                    'medium_buff_lake_ls_water_frac_adaptive': area_items.get('medium_buff_lake_ls_water_frac_adaptive'),
                    'medium_lake_s2_water_frac_adaptive': area_items.get('medium_lake_s2_water_frac_adaptive'),
                    'medium_buff_lake_s2_water_frac_adaptive': area_items.get('medium_buff_lake_s2_water_frac_adaptive'),                    
                    # Large lakes (Adaptive)
                    'large_lake_ls_water_frac_adaptive': area_items.get('large_lake_ls_water_frac_adaptive'),
                    'large_buff_lake_ls_water_frac_adaptive': area_items.get('large_buff_lake_ls_water_frac_adaptive'),
                    'large_lake_s2_water_frac_adaptive': area_items.get('large_lake_s2_water_frac_adaptive'),
                    'large_buff_lake_s2_water_frac_adaptive': area_items.get('large_buff_lake_s2_water_frac_adaptive'),                    
                    # Histograms
                    'ls_hist_counts': ls_hist_counts,
                    'ls_hist_bins': ls_hist_bins,
                    's2_hist_counts': s2_hist_counts,
                    's2_hist_bins': s2_hist_bins
                }

                # Append the summary to the list

                area_summaries.append(summary)

    return pd.DataFrame(area_summaries)