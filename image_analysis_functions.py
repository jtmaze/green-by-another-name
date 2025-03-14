# %% 1.0
import random
import ast
from typing import Optional
import os
import re
import glob
import pprint as pp

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling


random.seed(20)

"""
####################################
------------------------------------
Processing functions
------------------------------------
####################################
"""
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
    ls_fp: str, 
    s2_fp: str, 
    image_info: dict
):

    """
    Checks to ensure both images exist in directory
    If both images do not exist, then 
    Otherwise, specifies if one (ls or s2) file is missing. 
    """
    if os.path.exists(s2_fp) and os.path.exists(ls_fp):
        return True
    elif not os.path.exists(s2_fp) and not os.path.exists(ls_fp):
        #print(f'No data for {image_info}')
        return False
    elif not os.path.exists(s2_fp) and os.path.exists(ls_fp):
        print(f'Missing file {s2_fp}')
        return False
    else:
        print(f'Missing file {ls_fp}')
        return False

def read_band_by_description(
    raster_path: str,
    description: str, 
    image_window_params: Optional[dict]
):
    """
    Returns an array from the band description (string, not index) in a raster file. 
    Optional argument to crop the band to a window.
    """
    
    data = None
    with rio.open(raster_path) as src:
        # Get a description list
        desc_list = src.descriptions
        # Read the data
        for idx, desc in enumerate(desc_list, start=1):
            if desc == description and image_window_params is None:
                data = src.read(idx)
                return data
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
     
        if data is None:
            print("Error, could not find a band description to match your target band")
            return None
        
def apply_measure_mask(
    image_data: np.array,
    measure_mask: np.array
):
    """
    Uses the binary measure_mask to select pixels based on lake, land, or shoreline.
    """
    masked_data = image_data.copy()
    masked_data = np.where(measure_mask == 1, image_data, np.nan)

    return masked_data

###############################################
### 1.0 Functions to read data
###############################################

def rio_get_data_arrays(
    ls_path: str, 
    s2_path: str, 
    band_name: str
):
    """
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands
    """

    ls_data = read_band_by_description(ls_path, band_name, image_window_params=None)
    s2_data = read_band_by_description(s2_path, band_name, image_window_params=None)

    ls_data = ls_data.copy()
    s2_data = s2_data.copy()

    # Zero's values should already be nan, but just in case
    ls_data = np.where(ls_data >= 0, ls_data, np.nan)
    s2_data = np.where(s2_data >= 0, s2_data, np.nan)

    # Get the data's (LS and S2 grid will already be the same) 
    # bounds and transform as a window for the PLD mask
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
    
    # The data should have the same shapes to broadcast together.
    if ls_data.shape != s2_data.shape:
        print('The Sentinel-2 and Landsat8 data shapes are different, cannot compare')

    return ls_data, s2_data, image_window_params
    

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
    ls_data: np.array,
    s2_data: np.array,
    measure_mask: np.array
):
    """
    Returns two numpy arrays masked to measure specific regions (land, lake, or shoreline)
    Opperations:
    1) Selects shoreline, lake, or land pixels within measure mask
    3) Ensures both images have common set of nans
    """

    ls_masked = apply_measure_mask(ls_data, measure_mask)
    s2_masked = apply_measure_mask(s2_data, measure_mask)
    valid_ls_mask = ~np.isnan(ls_masked)
    valid_s2_mask = ~np.isnan(s2_masked)

    # Make the same nan values from filtering common to each dataset
    ls_data_out = np.where(valid_s2_mask, ls_masked, np.nan)
    s2_data_out = np.where(valid_ls_mask, s2_masked, np.nan)

    return ls_data_out, s2_data_out
    

def downsample_image_arrays(
    ls_pixels: np.array,
    s2_pixels: np.array,
    sample_size: int
):
    """
    Inputs: 2D images with identical masked pixels = np.nan
    Returns: 1D arrays with randomly downsampled to the sample_size if above pixel count.
    """

    s2_flat = s2_pixels.flatten()
    s2_flat = s2_flat[~np.isnan(s2_flat)]
    ls_flat = ls_pixels.flatten()
    ls_flat = ls_flat[~np.isnan(ls_flat)]

    valid_pix_cnt = ls_flat.size # ls_flat.size and s2_flat.size will be the same
    if valid_pix_cnt < sample_size:
        print(f"Not downsampling the number of measured pixels {valid_pix_cnt} < {sample_size}")
        return ls_flat, s2_flat, valid_pix_cnt
    else:
        sample_idx = np.random.choice(ls_flat.size, sample_size, replace=False)
        # Applies sample_idx to pixels_flat
        ls_sampled = ls_flat[sample_idx]
        s2_sampled = s2_flat[sample_idx]
        return ls_sampled, s2_sampled, valid_pix_cnt
    

def regress_reflectance(
    ls_sample: np.array, # Array should be 1D i.e. flat
    s2_sample: np.array, # 1D array
    outlier_frac: float, # The fraction of outliers to remove
    hist_return: bool
): 

    model = None
    model_domain = None
    ls_histogram = s2_histogram = None
    above_frac = below_frac = None

    sample_size = ls_sample.size
    if sample_size < 500:
        print(f'Error: Insuffcient quality pixels given parameters (less than 500)')
        model = 'Poor Quality Image Data'
        model_domain = 'Poor Quality Image Data'
        ls_histogram = s2_histogram = 'Poor Quality Image Data'
        above_frac = below_frac = 'Poor Quality Image Data'

    # Run the pixel regression analysis
    else:
        def regression_outlier_filter(sample_data: np.array, outlier_frac: float):
            """
            Replaces the lowest and highest fraction of values in sample_data with numpy.nan.
            For example, if outlier_frac is 0.05, the lower 5% and upper 5% of values are replaced with nan.
            Returns the filtered array.
            """
            # Compute thresholds for lower and upper percentiles
            lower_thresh = np.percentile(sample_data, outlier_frac * 100)
            upper_thresh = np.percentile(sample_data, 100 - outlier_frac * 100)

            # Create a copy of the sample data
            filtered_data = sample_data.copy()
            # Replace values below the lower threshold or above the upper threshold with np.nan
            filtered_data = np.where(filtered_data < lower_thresh, np.nan, filtered_data)
            filtered_data = np.where(filtered_data > upper_thresh, np.nan, filtered_data)

            return filtered_data

        ls_filtered = regression_outlier_filter(ls_sample, outlier_frac)
        s2_filtered = regression_outlier_filter(s2_sample, outlier_frac)
        
        # Filter both arrays using same mask
        nan_mask = ~np.isnan(ls_filtered) & ~np.isnan(s2_filtered)
        
        # Filter both arrays using same mask
        ls_modeled = ls_sample[nan_mask]
        s2_modeled = s2_sample[nan_mask]

        def rma_regression(x: np.array, y: np.array):
            """
            Returns the slope, intercept, and r-squared of an RMA regression model
            Code adapted from (https://github.com/OceanOptics/pylr2/blob/master/pylr2/regress2.py)
            """

            xy_ols = stats.linregress(x, y)
            yx_ols = stats.linregress(y, x)
            
            xy_slope = xy_ols.slope
            #xy_intercept = xy_ols.intercept
            yx_slope = 1 / yx_ols.slope # Need to invert the second slope so they're in y = mx + b form
            #yx_intercept = - yx_ols.intercept / yx_ols.slope

            # Check that slopes have the same sign
            if np.sign(xy_slope) != np.sign(yx_slope):
                print('Warning: Slopes have different signs')
                return None
            
            slope = np.sign(xy_slope) * (np.std(y) / np.std(x))
            intercept = np.mean(y) - slope * np.mean(x) 

            # Calculate R-squared
            r, _ = stats.pearsonr(x, y)
            r_squared = r ** 2
           
            return {'slope': slope, 'intercept': intercept, 'r_squared': r_squared}

        model = rma_regression(ls_modeled, s2_modeled)
        model_domain = (
            np.min([ls_modeled.min(), s2_modeled.min()]), 
            np.max([ls_modeled.max(), s2_modeled.max()])
        )

        # Make a fit-line from the model
        # Becuase Landsat is x-axis use it to make the domain
        xmin_val = np.nanmin(ls_modeled)
        xmax_val = np.nanmax(ls_modeled)
        ymin_val = np.nanmin(s2_modeled)
        ymax_val = np.nanmax(s2_modeled)
        min_modeled = model['slope'] * xmin_val + model['intercept']
        max_modeled = model['slope'] * xmax_val + model['intercept']

        plt.figure(figsize=(8,6))
        plt.scatter(ls_modeled, s2_modeled, s=1, marker='.', alpha=0.4)
        plt.plot([xmin_val, xmax_val], [min_modeled, max_modeled], color = 'red', linestyle='-', label='RMA Fit')
        # Add a 45 degree line for comparison
        plt.plot([min(xmin_val, ymin_val), max(xmax_val, ymax_val)], 
                [min(xmin_val, ymin_val), max(xmax_val, ymax_val)], 
                color='blue', 
                linestyle='--', 
                label='1:1 Line')
        textstr = f'$R^2 = {model['r_squared']:.4f}$\nSlope = {model['slope']:.4f}'
        box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=box_props)
        plt.xlabel('Landsat Reflectance')
        plt.ylabel('Sentinel-2 Reflectance')
        plt.legend(loc='lower right')
        plt.show()

        if hist_return == True:
            ls_histogram = np.histogram(ls_modeled, bins=100)
            s2_histogram = np.histogram(s2_modeled, bins=100)

        # Find the portion of pixels above/below the 45 degree line   
    
        below = np.where(ls_modeled > s2_modeled, 1, 0)
        above = np.where(ls_modeled < s2_modeled, 1, 0)
        below_frac = np.sum(below) / sample_size * 100
        above_frac = np.sum(above) / sample_size * 100
        print(f'Pixels above 45 degree line: {above_frac:.2f}')
        print(f'Pixels below 45 degree line: {below_frac:.2f}')

    return {'model': model, 
            'above_frac': above_frac,
            'below_frac': below_frac,
            'model_domain': model_domain, 
            'ls_histogram': ls_histogram, 
            's2_histogram': s2_histogram}

def get_pixel_samples(
    ls8_path: str,
    s2_path: str,
    pld_path: str,
    band_name: str,
    sample_size: int,
    zone: str,
    buffer_delim: int,
    buffer_delim_outer: int
):
    
    """
    Takes file paths and returns two downsampled 1-D arrays of matched pixels
    Order of opperations:
    1) Read a specific LandSat8 and Sentinel-2 band as matching numpy arrays
    2) Generate a measurement mask from PLD with dilation and errosion zones specified
    3) Filter both image's array data by the mask and low/high thresholds
    4) Resample the arrays for regression based on a pre-defined sample size
    """
    
    ls_data, s2_data, image_window_params = rio_get_data_arrays(
        ls8_path, s2_path, band_name
    )
    measure_mask = make_measure_mask(
        pld_path,
        image_window_params,
        zone,
        buffer_delim,
        buffer_delim_outer,
    )
    ls_pixels, s2_pixels = find_measured_pixels(
        ls_data, 
        s2_data, 
        measure_mask
    )
    ls_sample, s2_sample, valid_pix_cnt = downsample_image_arrays(
        ls_pixels, 
        s2_pixels, 
        sample_size
    )

    return ls_sample, s2_sample, valid_pix_cnt

def get_ndwi_samples(
    image_info: dict,
    pld_fp: str,
    zone: str,
    buffer_delim: int,
    buffer_delim_outer: int,
    sample_size: int
):
    """
    Steps:
        1) Generates coincident NDWI images given image info dictionary
        2) Creates a measurement mask from PLD with specified zone and buffer values
        3) Applies the measurement mask to the NDWI images
        4) Downsamples the NDWI images to 1D (flat) arrays with a specified sample size
    """

    ls_ndwi, s2_ndwi, img_window_params = make_ndwi_images(image_info)
    measure_mask = make_measure_mask(pld_fp, img_window_params, zone, buffer_delim, buffer_delim_outer)
    ls_pixels, s2_pixels = find_measured_pixels(ls_ndwi, s2_ndwi, measure_mask)
    ls_sample, s2_sample, valid_pix_cnt = downsample_image_arrays(ls_pixels, s2_pixels, sample_size)

    return ls_sample, s2_sample, valid_pix_cnt

def regress_image_pairs(
    image_info: dict,
    mask_params: dict,
    regression_params: dict,
    hist_return: bool,
) -> dict:
    
    """
    Perform regression analysis between Landsat8 and Sentinel2 image pairs.
    
    Args:
        image_info: Dict containing level, date, roi, band_name, and resample_method
        mask_params: Dict containing zone specification and buffer parameters
        regression_params: Dict containing sample_size and model_domain
    
    Returns:
        dict: Summary of regression results including:
         1. The orginal image, mask and regression parameters
         2. The regression output (model above/below frac, etc.) as a dictionary nested within
    """
    # Image params
    # Extract parameters
    level, date, roi, band_name, resample_method = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['band_name'],
        image_info['resample_method']
    )
    
    zone, buffer_delim, buffer_delim_outer = (
        mask_params['zone'],
        mask_params['buffer_delim'],
        mask_params['buffer_delim_outer']
    )
    
    sample_size, outlier_frac = (
        regression_params['sample_size'],
        regression_params['outlier_frac']
    )
    
    # Make file paths
    s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
    ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'
    res = re.search(r"(\d{2}$)", resample_method).group(1)
    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif'
    # Check if images exist
    if check_match_imgs(ls8_fp, s2_fp, image_info):
        sample_params = {
            'zone': zone,
            'buffer_delim': buffer_delim,
            'buffer_delim_outer': buffer_delim_outer,
            'sample_size': sample_size
        }
        # Get the NDWI  samples
        if band_name == "NDWI":
            ls_sample, s2_sample, valid_pix_cnt = get_ndwi_samples(image_info, pld_fp, **sample_params)
        # Get the single band (G or NIR) samples
        else:
            ls_sample, s2_sample, valid_pix_cnt = get_pixel_samples(
                ls8_fp, s2_fp, pld_fp, band_name=band_name, **sample_params
            )
        # Run regression function
        print(f'{level} {resample_method} regression for {band_name} for date {date} in the {roi} region with PLD {zone} {buffer_delim}m')
        regression_output = regress_reflectance(ls_sample, s2_sample, outlier_frac, hist_return)
    else: # Return no image data if matching images not found
        valid_pix_cnt = regression_output = "No Image Data"

    return {
        'level': level,
        'resample_method': resample_method,
        'date': date,
        'roi': roi,
        'band_name': band_name,
        'zone': zone,
        'buffer_delim': buffer_delim,
        'buffer_delim_outer': buffer_delim_outer,
        'sample_size': sample_size,
        'outlier_frac': outlier_frac,
        'valid_pix_cnt': valid_pix_cnt,
        'regression_output': regression_output
    }

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

def make_ndwi_images(
    image_info: dict
):
    
    """
    Takes the file paths for two coincident images
    Returns two NDWI images as numpy arrays. 
    Plots the NDWI images for visual inspection.
    """
    
    level, date, roi, band_name, resample_method = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['band_name'],
        image_info['resample_method']
    )

    s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
    ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'
    
    # Read the raster data necessary to calculate NDWI
    ls_green, s2_green, image_window_params = rio_get_data_arrays(
        ls8_fp, s2_fp, 'Green'
    )
    ls_nir, s2_nir, image_window_params = rio_get_data_arrays(
        ls8_fp, s2_fp, 'NIR'
    )

    # Calculate NDWI
    ls_ndwi = calc_ndwi(ls_green, ls_nir)
    s2_ndwi = calc_ndwi(s2_green, s2_nir)

    # Plot the NDWI images
    green_white_blue = LinearSegmentedColormap.from_list("GreenWhiteBlue", ["green", "white", "blue"])
    fig, ax = plt.subplots()
    ax.set_facecolor('darkgrey')
    ax.imshow(ls_ndwi, cmap=green_white_blue)
    ax.set_title('Landsat NDWI')
    plt.colorbar(ax.images[0], ax=ax)
    plt.show()
    fig, ax = plt.subplots()
    ax.set_facecolor('darkgrey')
    ax.imshow(s2_ndwi, cmap=green_white_blue)
    ax.set_title('Sentinel-2 NDWI')
    plt.colorbar(ax.images[0], ax=ax)
    plt.show()
    
    return ls_ndwi, s2_ndwi, image_window_params

def mask_ndwi_images(
    ndwi: np.array,
    image_window_params: dict,
    roi: str,
    res: int
):
    """
    This masks NDWI images to lakes with a 60m buffer. 
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

def clean_ndwi_data(ndwi_data: np.array):
    """
    Removes NaN values from NDWI before computing Otsu Thresholds or Histograms
    """
    flat_ndwi = ndwi_data.flatten()
    valid_mask = ~np.isnan(flat_ndwi)
    valid_data = flat_ndwi[valid_mask]
    valid_data = np.clip(valid_data, -1, 1)
    
    return valid_data


def find_otsu_threshold(
    ndwi: np.array, 
    show_hist: bool,
):
    """
    Input: NDWI array (-1,1)
    Output: Otsu Threshold for a given NDWI image
    """
    
    valid_data = clean_ndwi_data(ndwi)

    n_bins = 1_000
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

def calc_lake_wtr_frac(
    water_mask: np.array, # Binary water mask from an image
    ndwi: float, # The original NDWI image (to check for valid pixels)
    roi: str,
    res: int, # Resolution to ensure we read the correct PLD mask
    image_window_params: dict # The window parameters for reading the PLD mask (georeferenced properly)
):
    """
    Calculates the water fraction inside the PLD lake mask
    """

    pld_mask = make_measure_mask(
        f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif', 
        image_window_params, 
        zone='lake', 
        buffer_delim=60, 
        buffer_delim_outer=None
    )

    lake_water_pixels = np.sum((water_mask == 1) & (pld_mask == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (pld_mask == 1))
    water_frac = lake_water_pixels / valid_pixels * 100

    return water_frac

def calc_shoreline_wtr_frac(
    water_mask: np.array, # Binary water mask from an image
    ndwi: float, # The original NDWI image (to check for valid pixels)
    roi: str,
    res: int, # Resolution to ensure we read the correct PLD mask
    image_window_params: dict # The window parameters for reading the PLD mask (georeferenced properly)
):
    """
    Calculates the water fraction inside the shoreline zone
    """
    shoreline_mask = make_measure_mask(
        f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif', 
        image_window_params, 
        zone='shoreline', 
        buffer_delim=-60, 
        buffer_delim_outer=60
    )

    shoreline_water_pixels = np.sum((water_mask == 1) & (shoreline_mask == 1))
    valid_pixels = np.sum((~np.isnan(ndwi)) & (shoreline_mask == 1))
    water_frac = shoreline_water_pixels / valid_pixels * 100

    return water_frac

def image_wtr_area(
    image_info: dict,
    write_mask: bool,
    hist_return: bool, 
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
    lake_ls_water_frac_otsu = None
    shoreline_ls_water_frac_otsu = None
    total_s2_water_frac_otsu = None
    lake_s2_water_frac_otsu = None
    shoreline_s2_water_frac_otsu = None
    # Adaptive water fractions
    total_ls_water_frac_adaptive = None
    lake_ls_water_frac_adaptive = None
    shoreline_ls_water_frac_adaptive = None
    total_s2_water_frac_adaptive = None
    lake_s2_water_frac_adaptive = None
    shoreline_s2_water_frac_adaptive = None
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

    s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
    ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'

    if check_match_imgs(ls8_fp, s2_fp, image_info):
        print(f"Working {level} for {date} over the {roi} region")
        ls_ndwi, s2_ndwi, image_window_params = make_ndwi_images(image_info)
        res = re.search(r"(\d{2}$)", resample_method).group(1) # Gets the resolution digits from resample method
        ls_ndwi_lakes = mask_ndwi_images(ls_ndwi, image_window_params, roi, res)
        s2_ndwi_lakes = mask_ndwi_images(s2_ndwi, image_window_params, roi, res)

        # Ensure there's enough quality lake pixels in the image.
        if np.sum(~np.isnan(ls_ndwi_lakes)) < 25_000 or np.sum(~np.isnan(s2_ndwi_lakes)) < 25_000:
            print('ERROR: Skipping water area calculations -- Bad Image')

        else:
            # Find otsu and adaptive thresholds
            print("----- LandSat Histogram --------------")
            ls_otsu_threshold, ls_hist = find_otsu_threshold(ls_ndwi_lakes, show_hist=False)
            ls_adaptive_land, ls_adaptive_water = find_adaptive_thresholds(ls_hist, ls_otsu_threshold, show_hist=True)
            print("----- Sentinel-2 Histogram --------------")
            s2_otsu_threshold, s2_hist = find_otsu_threshold(s2_ndwi_lakes, show_hist=False)
            s2_adaptive_land, s2_adaptive_water = find_adaptive_thresholds(s2_hist, s2_otsu_threshold, show_hist=True)

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

            # Calculate the lake water fractions
            lake_ls_water_frac_otsu = calc_lake_wtr_frac(ls_water_otsu, ls_ndwi, roi, res, image_window_params)
            lake_s2_water_frac_otsu = calc_lake_wtr_frac(s2_water_otsu, s2_ndwi, roi, res, image_window_params)
            lake_ls_water_frac_adaptive = calc_lake_wtr_frac(ls_water_adaptive, ls_ndwi, roi, res, image_window_params)
            lake_s2_water_frac_adaptive = calc_lake_wtr_frac(s2_water_adaptive, s2_ndwi, roi, res, image_window_params)

            # Calculate the shoreline water fractions
            shoreline_ls_water_frac_otsu = calc_shoreline_wtr_frac(ls_water_otsu, ls_ndwi, roi, res, image_window_params)
            shoreline_s2_water_frac_otsu = calc_shoreline_wtr_frac(s2_water_otsu, s2_ndwi, roi, res, image_window_params)
            shoreline_ls_water_frac_adaptive = calc_shoreline_wtr_frac(ls_water_adaptive, ls_ndwi, roi, res, image_window_params)
            shoreline_s2_water_frac_adaptive = calc_shoreline_wtr_frac(s2_water_adaptive, s2_ndwi, roi, res, image_window_params)

            if write_mask == True:
                print("No code to export the water masks, yet...")

            if hist_return == False:
                ls_hist = s2_hist = None

    return {
        # Thresholds
        'ls_otsu_threshold': ls_otsu_threshold,
        'ls_adaptive_land': ls_adaptive_land,
        'ls_adaptive_water': ls_adaptive_water,
        's2_otsu_threshold': s2_otsu_threshold,
        's2_adaptive_land': s2_adaptive_land,
        's2_adaptive_water': s2_adaptive_water,
        # Otsu Water Fractions
        'total_ls_water_frac_otsu': total_ls_water_frac_otsu,
        'lake_ls_water_frac_otsu': lake_ls_water_frac_otsu,
        'shoreline_ls_water_frac_otsu': shoreline_ls_water_frac_otsu,
        'total_s2_water_frac_otsu': total_s2_water_frac_otsu,
        'lake_s2_water_frac_otsu': lake_s2_water_frac_otsu,
        'shoreline_s2_water_frac_otsu': shoreline_s2_water_frac_otsu,
        # Adaptive Water Fractions
        'total_ls_water_frac_adaptive': total_ls_water_frac_adaptive,
        'lake_ls_water_frac_adaptive': lake_ls_water_frac_adaptive,
        'shoreline_ls_water_frac_adaptive': shoreline_ls_water_frac_adaptive,
        'total_s2_water_frac_adaptive': total_s2_water_frac_adaptive,
        'lake_s2_water_frac_adaptive': lake_s2_water_frac_adaptive,
        'shoreline_s2_water_frac_adaptive': shoreline_s2_water_frac_adaptive,
        # Histograms
        'ls_hist': ls_hist,
        's2_hist': s2_hist
    }

def numpy_to_list(data):
    """
    Converts numpy arrays to lists for storage in a dataframe
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def make_otsu_area_summaries(
    image_info: dict, 
    levels: list, 
    rois: list, 
    dates: list,
    hist_return: bool
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
                area_items = image_wtr_area(image_info, write_mask=False, hist_return=hist_return)
                # Convert the numpy histogram objects to lists for storage
                if area_items.get('ls_hist') is None:
                    ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = 'Poor Quality Image'
                else:
                    ls_hist_counts = numpy_to_list(area_items.get('ls_hist')[0])
                    ls_hist_bins = numpy_to_list(area_items.get('ls_hist')[1])
                    s2_hist_counts = numpy_to_list(area_items.get('s2_hist')[0])
                    s2_hist_bins = numpy_to_list(area_items.get('s2_hist')[1])

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
                    # Otsu Water Fractions
                    'total_ls_water_frac_otsu': area_items.get('total_ls_water_frac_otsu'),
                    'lake_ls_water_frac_otsu': area_items.get('lake_ls_water_frac_otsu'),
                    'shoreline_ls_water_frac_otsu': area_items.get('shoreline_ls_water_frac_otsu'),
                    'total_s2_water_frac_otsu': area_items.get('total_s2_water_frac_otsu'),
                    'lake_s2_water_frac_otsu': area_items.get('lake_s2_water_frac_otsu'),
                    'shoreline_s2_water_frac_otsu': area_items.get('shoreline_s2_water_frac_otsu'),
                    # Adaptive Water Fractions
                    'total_ls_water_frac_adaptive': area_items.get('total_ls_water_frac_adaptive'),
                    'lake_ls_water_frac_adaptive': area_items.get('lake_ls_water_frac_adaptive'),
                    'shoreline_ls_water_frac_adaptive': area_items.get('shoreline_ls_water_frac_adaptive'),
                    'total_s2_water_frac_adaptive': area_items.get('total_s2_water_frac_adaptive'),
                    'lake_s2_water_frac_adaptive': area_items.get('lake_s2_water_frac_adaptive'),
                    'shoreline_s2_water_frac_adaptive': area_items.get('shoreline_s2_water_frac_adaptive'),
                    # Histograms
                    'ls_hist_counts': ls_hist_counts,
                    'ls_hist_bins': ls_hist_bins,
                    's2_hist_counts': s2_hist_counts,
                    's2_hist_bins': s2_hist_bins
                }
                area_summaries.append(summary)

    return pd.DataFrame(area_summaries)

def make_reflectance_summaries(
    image_info: dict,
    mask_params: dict,
    regression_params: dict,
    levels: list,
    rois: list,
    dates: list,
    hist_return: bool
) -> pd.DataFrame:
    """
    Generates regression summaries for multiple image pairs across different processing levels,
    regions of interest, and dates.
    
    Parameters:
    -----------
    image_info : dict
        Dictionary containing image metadata (band_name, resample_method)
    mask_params : dict
        Parameters for masking pixels (zone, buffer_delim, buffer_delim_outer)
    regression_params : dict
        Parameters for regression analysis (sample_size, outlier_frac)
    levels : list
        List of processing levels to analyze
    rois : list
        List of regions of interest to analyze
    dates : list
        List of dates to analyze
    hist_return : bool
        Whether to return histogram data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing regression metrics for all image pairs
    """
    
    regression_summaries = []
    
    # Process each combination of level, ROI, and date
    for level in levels:
        image_info['level'] = level
        for roi in rois:
            image_info['roi'] = roi
            for date in dates:
                image_info['date'] = date

                # Perform regression analysis on the current image pair
                regression_result = regress_image_pairs(
                    image_info, mask_params, regression_params, hist_return=hist_return
                )
                # Handle case when no image data is available
                regression_items_str = regression_result['regression_output']
                if regression_items_str == 'No Image Data':
                    model = slope = intercept = r_squared = "No Image Data"
                    above_frac = below_frac = model_domain = "No Image Data"
                    ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "No Image Data"
                else: 
                    # Convert string representation to dict if needed
                    if isinstance(regression_items_str, str):
                        regression_items = ast.literal_eval(regression_items_str)
                    else:
                        regression_items = regression_items_str

                    model = regression_items['model']
                    
                    # Extract regression metrics if image quality is sufficient
                    if model != "Poor Quality Image Data":
                        # Extract model parameters
                        slope = model['slope']
                        intercept = model['intercept']
                        r_squared = model['r_squared']
                        
                        # Extract additional metrics
                        above_frac = regression_items['above_frac']  # % pixels above 45° line
                        below_frac = regression_items['below_frac']  # % pixels below 45° line
                        model_domain = regression_items['model_domain']
                        
                        # Handle histogram data if requested
                        if hist_return:
                            ls_hist_counts = numpy_to_list(regression_items.get('ls_histogram')[0])
                            ls_hist_bins = numpy_to_list(regression_items.get('ls_histogram')[1])
                            s2_hist_counts = numpy_to_list(regression_items.get('s2_histogram')[0])
                            s2_hist_bins = numpy_to_list(regression_items.get('s2_histogram')[1])
                        else:
                            ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "Histogram Not Returned"
                    else:
                        # Set default values for poor quality images
                        slope = intercept = r_squared = "Poor Quality Image Data"
                        model_domain = "Poor Quality Image Data"
                        above_frac = below_frac = "Poor Quality Image Data"
                        ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "Poor Quality Image Data"

                # Create summary dictionary for this image pair
                summary = {
                    'level': regression_result['level'],
                    'resample_method': regression_result['resample_method'],
                    'roi': regression_result['roi'],
                    'date': regression_result['date'],
                    'band_name': regression_result['band_name'], 
                    'buffer_delim': regression_result['buffer_delim'],
                    'buffer_delim_outer': regression_result['buffer_delim_outer'],
                    'sample_size': regression_result['sample_size'],
                    'outlier_frac': regression_result['outlier_frac'], 
                    'valid_pix_cnt': regression_result['valid_pix_cnt'],
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'above_frac': above_frac,
                    'below_frac': below_frac,
                    'model_domain': model_domain,
                    'ls_hist_counts': ls_hist_counts,
                    'ls_hist_bins': ls_hist_bins,
                    's2_hist_counts': s2_hist_counts,
                    's2_hist_bins': s2_hist_bins
                }

                regression_summaries.append(summary)

    return pd.DataFrame(regression_summaries)


# %%
