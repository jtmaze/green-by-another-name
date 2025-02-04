# %% 1.0
import random
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
def extract_unique(files: list, pattern: re.Pattern[str]):
    unique_items = set()
    for f in files:
        match = re.search(pattern, f)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

def check_match_imgs(ls_fp: str, s2_fp: str, image_info: dict):

    """
    Checks to ensure both images exist in directory
    Then, specifies if one file is missing or there's no data for img params
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

def read_band_by_description(raster_path: str, description: str, image_window_params: Optional[dict]):
    """
    Returns an array from the band description (not index) in a raster file. 
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
        
def apply_measure_mask(image_data: np.array, measure_mask: np.array):
    """
    Uses the binary measure_mask to select pixels based on lake, land, or shoreline.
    """
    masked_data = image_data.copy()
    masked_data = np.where(measure_mask == 1, image_data, np.nan)

    return masked_data

###############################################
### 1.0 Functions to read data
###############################################

def rio_get_data_arrays(ls_path: str, s2_path: str, band_name: str):
    """
    Returns two numpy arrays for corresponding Sentinel-2 and Landsat8 bands
    """

    ls_data = read_band_by_description(ls_path, band_name, image_window_params=None)
    s2_data = read_band_by_description(s2_path, band_name, image_window_params=None)

    ls_data = ls_data.copy()
    s2_data = s2_data.copy()

    ls_data = np.where(ls_data >= 0.001, ls_data, np.nan)
    s2_data = np.where(s2_data >= 0.001, s2_data, np.nan)

    # Get the data's (LandSat's) bounds and transform as a window for the PLD mask
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
    

def make_measure_mask(pld_path: str, 
                     image_window_params: dict, 
                     zone: str, # 1) lake, 2) land, or 3) shoreline
                     buffer_delim: int,
                     buffer_delim_outer: Optional[int]):
    """
    Returns a binary mask denoting which pixels to compare across images (0=ingore, 1=compare)
    Determined from PLD attributes to define lakes, land, and shoreline at different buffer values.
    """
    desc = f'buffered_{buffer_delim}m'
    pld_mask = read_band_by_description(pld_path, desc, image_window_params)

    measure_mask = np.zeros_like(pld_mask)
    if zone == 'lake':
        measure_mask = np.where(pld_mask == 1, 1, measure_mask)
    elif zone == 'land':
        measure_mask = np.where(pld_mask == 1, measure_mask, 1)
    elif zone == 'shoreline':
        if buffer_delim_outer is None:
            raise ValueError("buffer_delim_outer required for shoreline zone")
        outer_desc = f'buffered_{buffer_delim_outer}m'
        outer_mask = read_band_by_description(pld_path, outer_desc, image_window_params)
        measure_mask = np.where((pld_mask == 0) & (outer_mask == 1),
                                1,
                                measure_mask)
    else:
        raise ValueError("Zone must be 'lake', 'land' or 'shoreline'")

    return measure_mask

def find_measured_pixels(ls_data: np.array,
                           s2_data: np.array,
                           measure_mask: np.array):
    """
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
    

def downsample_image_arrays(ls_pixels: np.array,
                            s2_pixels: np.array,
                            sample_size: int):
    """
    Inputs: 2D images with identical masked pixels = np.nan
    Returns: 1D arrays with randomly downsampled to the sample_size if above.
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
    if sample_size < 10_000:
        print(f'Error: Insuffcient quality pixels given parameter (less than 10,000)')
        model = 'Poor Quality Image Data'
        model_domain = 'Poor Quality Image Data'
        ls_histogram = s2_histogram = 'Poor Quality Image Data'
        above_frac = below_frac = 'Poor Quality Image Data'

    else:
        def outlier_filter(sample_data: np.array, outlier_frac: float):
            """
            Replaces the lowest and highest outlier_frac fraction of values in sample_data with numpy.nan.
            For example, if outlier_frac is 0.05, the lower 5% and upper 5% of values are replaced with nan.
            Returns the filtered array.
            """
            # Compute thresholds for lower and upper percentiles
            #lower_thresh = np.percentile(sample_data, outlier_frac * 100)
            upper_thresh = np.percentile(sample_data, 100 - outlier_frac * 100)

            # Create a copy of the sample data
            filtered_data = sample_data.copy()
            # Replace values below the lower threshold or above the upper threshold with np.nan
            #filtered_data = np.where(filtered_data < lower_thresh, np.nan, filtered_data)
            filtered_data = np.where(filtered_data > upper_thresh, np.nan, filtered_data)

            return filtered_data

        ls_filtered = outlier_filter(ls_sample, outlier_frac)
        s2_filtered = outlier_filter(s2_sample, outlier_frac)
        
        # Filter both arrays using same mask
        nan_mask = ~np.isnan(ls_filtered) & ~np.isnan(s2_filtered)
        
        # Filter both arrays using same mask
        ls_modeled = ls_sample[nan_mask]
        s2_modeled = s2_sample[nan_mask]

        def rma_regression(x: np.array, y: np.array):
            """
            Returns the slope, intercept, and r-squared of an RMA regression model
            Code and steps adopted from (https://github.com/OceanOptics/pylr2/blob/master/pylr2/regress2.py)
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
        print(xmin_val, xmax_val, ymin_val, ymax_val)
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

def get_pixel_samples(ls8_path: str,
                      s2_path: str,
                      pld_path: str,
                      band_name: str,
                      sample_size: int,
                      zone: str,
                      buffer_delim: int,
                      buffer_delim_outer: int):
    
    """
    Takes file paths and returns two downsampled 1-D arrays of matched pixels
    Order of opperations:
    1) Read a specific LandSat8 and Sentinel-2 band as matching numpy arrays
    2) Generate a measurement mask from PLD with dilation and errosion specified
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

def get_ndwi_samples(image_info: dict,
                     pld_fp: str,
                     zone: str,
                     buffer_delim: int,
                     buffer_delim_outer: int,
                     sample_size: int
    ):

    ls_ndwi, s2_ndwi, img_window_params = make_ndwi_images(image_info)
    measure_mask = make_measure_mask(pld_fp, img_window_params, zone, buffer_delim, buffer_delim_outer)
    ls_pixels, s2_pixels = find_measured_pixels(ls_ndwi, s2_ndwi, measure_mask)
    ls_sample, s2_sample, valid_pix_cnt = downsample_image_arrays(ls_pixels, s2_pixels, sample_size)

    return ls_sample, s2_sample, valid_pix_cnt

def regress_image_pairs(image_info: dict,
                        mask_params: dict,
                        regression_params: dict,
                        hist_return: bool,
    ) -> dict:
    
    """
    Perform regression analysis between Landsat8 and Sentinel2 image pairs.
    
    Args:
        image_info: Dict containing level, date, roi, and band_name
        mask_params: Dict containing zone and buffer parameters
        regression_params: Dict containing sample_size and model_domain
    
    Returns:
        dict: Summary of regression results including model and excluded fraction of pixels
    """
    # Image params
    # Extract parameters
    level, date, roi, band_name = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['band_name']
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
    s2_fp = f'./data/{level}_images/Sentinel2-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'
    ls8_fp = f'./data/{level}_images/LandSat8-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'
    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks.tif'

    if check_match_imgs(ls8_fp, s2_fp, image_info):
        sample_params = {
            'zone': zone,
            'buffer_delim': buffer_delim,
            'buffer_delim_outer': buffer_delim_outer,
            'sample_size': sample_size
        }
        # Get the NDWI or single band pixel samples
        if band_name == "NDWI":
            ls_sample, s2_sample, valid_pix_cnt = get_ndwi_samples(image_info, pld_fp, **sample_params)
        else:
            ls_sample, s2_sample, valid_pix_cnt = get_pixel_samples(
                ls8_fp, s2_fp, pld_fp, band_name=band_name, **sample_params
            )
        # Run regression function
        print(f'{level} regression for {band_name} for date {date} in the {roi} region with PLD {zone} {buffer_delim}m')
        regression_output = regress_reflectance(ls_sample, s2_sample, outlier_frac, hist_return)
    else: # Return no image data if matching images not found
        valid_pix_cnt = regression_output = "No Image Data"

    return {
        'level': level,
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
    Given the Green and NIR data arrays, the function calculates NDWI for each value
    """
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
    """
    
    level = image_info['level']
    date = image_info['date']
    roi = image_info['roi']
    
    s2_fp = f'./data/{level}_images/Sentinel2-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'
    ls8_fp = f'./data/{level}_images/Landsat8-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'
    
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
    roi: str
):

    pld_path = f'./data/pld_rasterized/{roi}_lake_masks.tif'
    pld_plus = make_measure_mask(pld_path, 
                                 image_window_params, 
                                 zone='lake', 
                                 buffer_delim=60, #TODO: What's the most appropriate value for this??
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
    Input: NDWI array [-1,1]
    Output: Otsu Threshold, and possibly the data
    """
    
    valid_data = clean_ndwi_data(ndwi)

    n_bins = 500
    hist, bin_edges = np.histogram(valid_data, bins=n_bins, range=(-1, 1))
    total_pixels = hist.sum()
    pdf = hist / total_pixels
    cumulative_prob = np.cumsum(pdf)               
    cumulative_intensity = np.cumsum(pdf * np.arange(n_bins))
    best_threshold = 0
    best_variance = 0.0

    for t in range(n_bins):
        w0 = cumulative_prob[t]
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            # This means all data is on one side of the threshold => skip
            continue

        m0 = cumulative_intensity[t] / w0
        m1 = (cumulative_intensity[-1] - cumulative_intensity[t]) / w1

        # Between-class variance
        var_between = w0 * w1 * (m0 - m1) ** 2

        if var_between > best_variance:
            best_variance = var_between
            best_threshold = t

    threshold = 0.5 * (bin_edges[best_threshold] + bin_edges[best_threshold + 1])

    if show_hist == True:
        plt.hist(valid_data, bins=50, edgecolor='black')
        plt.axvline(x=threshold, color='red', label=f'Threshold = {threshold}')
        plt.xlabel('NDWI values')
        plt.legend()
        plt.show()
    
    return threshold

def otsu_image_wtr_area(
        image_info: dict,
        write_mask: bool,
        hist_return: bool, 
    ):
    """
    Returns a dictionary with the otsu threshold, and water fraction from each image
    """

    # Initialize all variables at the start
    ls_threshold = None
    ls_water_frac = None 
    s2_threshold = None
    s2_water_frac = None
    ls_hist = None
    s2_hist = None

    level, date, roi, band_name = (
        image_info['level'], 
        image_info['date'], 
        image_info['roi'],
        image_info['band_name']
    )
    s2_fp = f'./data/{level}_images/Sentinel2-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'
    ls_fp = f'./data/{level}_images/Landsat8-{level}_date_{date}_roi_{roi}_resampled_bilinear30.tif'

    if check_match_imgs(ls_fp, s2_fp, image_info):
        print(f"Working {level} for {date} over the {roi} region")
        ls_ndwi, s2_ndwi, image_window_params = make_ndwi_images(image_info)
        ls_ndwi_lakes = mask_ndwi_images(ls_ndwi, image_window_params, roi)
        s2_ndwi_lakes = mask_ndwi_images(s2_ndwi, image_window_params, roi)

        # Ensure there's enough quality lake pixels in the image.
        if np.sum(~np.isnan(ls_ndwi_lakes)) < 25_000 or np.sum(~np.isnan(s2_ndwi_lakes)) < 25_000:
            print('ERROR: Skipping water area calculations -- Bad Image')
            ls_threshold = ls_water_frac = s2_threshold = s2_water_frac = 'Poor Quality Image'

        else:

            print("----- LandSat Histogram --------------")
            ls_threshold = find_otsu_threshold(ls_ndwi_lakes, show_hist=False)
            print("----- Sentinel-2 Histogram --------------")
            s2_threshold = find_otsu_threshold(s2_ndwi_lakes, show_hist=False)

            ls_water = (ls_ndwi > ls_threshold).astype(int)
            s2_water = (s2_ndwi > s2_threshold).astype(int)

            def calc_wtr_frac(water_mask, ndwi):

                water_pixels = np.sum(water_mask == 1)
                valid_pixels = np.sum(~np.isnan(ndwi))
                water_frac = water_pixels / valid_pixels * 100

                return water_frac
            
            ls_water_frac = calc_wtr_frac(ls_water, ls_ndwi)
            s2_water_frac = calc_wtr_frac(s2_water, s2_ndwi)

            if write_mask == True:
                print("No code to export the water masks, yet...")

            if hist_return == True:
                ls_out_data = clean_ndwi_data(ls_ndwi_lakes)
                s2_out_data = clean_ndwi_data(s2_ndwi_lakes)

                bins = 500
                ls_hist = np.histogram(ls_out_data, bins=bins, density=True)
                s2_hist = np.histogram(s2_out_data, bins=bins, density=True)

    else:
        ls_threshold = ls_water_frac = s2_threshold = s2_water_frac = ls_hist = s2_hist = 'No Image Data'

    return {'ls_threshold': ls_threshold, 
            'ls_water_frac': ls_water_frac, 
            's2_threshold': s2_threshold, 
            's2_water_frac': s2_water_frac, 
            'ls_hist': ls_hist, 
            's2_hist': s2_hist
        }

def plot_otsu_histograms(
    sr: pd.Series,
    toa: pd.Series,
    date: str,
    roi: str):
    
    """
    Plots the NDWI histograms for Landsat and Sentinel-2 images
    """
    
    # Access dictionary elements using iloc[0] and key names
    ls_sr_threshold = sr['otsu_items'].iloc[0]['ls_threshold']
    ls_sr_hist = sr['otsu_items'].iloc[0]['ls_hist'][0]
    ls_sr_water_frac = sr['otsu_items'].iloc[0]['ls_water_frac']
    ls_toa_threshold = toa['otsu_items'].iloc[0]['ls_threshold']
    ls_toa_hist = toa['otsu_items'].iloc[0]['ls_hist'][0]
    ls_toa_water_frac = toa['otsu_items'].iloc[0]['s2_water_frac']

    s2_sr_threshold = sr['otsu_items'].iloc[0]['s2_threshold']
    s2_sr_hist = sr['otsu_items'].iloc[0]['s2_hist'][0]
    s2_sr_water_frac = sr['otsu_items'].iloc[0]['s2_water_frac']
    s2_toa_threshold = toa['otsu_items'].iloc[0]['s2_threshold']
    s2_toa_hist = toa['otsu_items'].iloc[0]['s2_hist'][0]
    s2_toa_water_frac = toa['otsu_items'].iloc[0]['s2_water_frac']

    # Should all have the same bin edges...
    bin_edges = sr['otsu_items'].iloc[0]['ls_hist'][1]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Plot histogram lines
    ax.plot(bin_centers, ls_sr_hist, linestyle='-', color='red', label='Landsat SR', linewidth=2)
    ax.plot(bin_centers, s2_sr_hist, linestyle='-', color='green', label='Sentinel-2 SR', linewidth=2)
    ax.plot(bin_centers, ls_toa_hist, linestyle=':', color='red', label='Landsat TOA', linewidth=4)
    ax.plot(bin_centers, s2_toa_hist, linestyle=':', color='green', label='Sentinel-2 TOA', linewidth=4)

    # Add threshold lines
    ax.axvline(x=ls_sr_threshold, color='red', linestyle='-', linewidth=3)
    ax.axvline(x=s2_sr_threshold, color='green', linestyle='-', linewidth=3)
    ax.axvline(x=ls_toa_threshold, color='red', linestyle=':', linewidth=3)
    ax.axvline(x=s2_toa_threshold, color='green', linestyle=':', linewidth=3)

    # Text boxes
    ax.text(
        0.95, 0.95,
        f'LS SR water frac: {ls_sr_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='red')
    )
    ax.text(
        0.95, 0.89,
        f'S2 SR water frac: {s2_sr_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='green')
    )
    ax.text(
        0.95, 0.80,
        f'LS TOA water frac: {ls_toa_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='red')
    )
    ax.text(
        0.95, 0.74,
        f'S2 TOA water frac: {s2_toa_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='green')
    )

    # Customize plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of NDWI Values for {date} in {roi}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reflectance_histograms(
    sr: pd.Series, 
    toa: pd.Series, 
    band: str,
    date: str, 
    roi: str,
    hist_range: tuple[float, float] = (0.0, 0.1)
    ):

    """
    Plots the Green or NIR histograms for Landsat and Sentinel-2 TOA/SR images
    """
    print(sr['regression_output'].iloc[0])
    # Access histograms for ls and s2 images
    ls_sr_hist = sr['regression_output'].iloc[0]['ls_histogram'][0]
    ls_toa_hist = toa['regression_output'].iloc[0]['ls_histogram'][0] 
    s2_sr_hist = sr['regression_output'].iloc[0]['s2_histogram'][0]
    s2_toa_hist = toa['regression_output'].iloc[0]['s2_histogram'][0]

    # Should all have the same bin edges...
    bin_edges = sr['regression_output'].iloc[0]['ls_histogram'][1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ls_sr_mean = np.average(bin_centers, weights=np.array(ls_sr_hist))
    ls_toa_mean = np.average(bin_centers, weights=np.array(ls_toa_hist))
    s2_sr_mean = np.average(bin_centers, weights=np.array(s2_sr_hist))
    s2_toa_mean = np.average(bin_centers, weights=np.array(s2_toa_hist))

    mask = (bin_centers >= hist_range[0]) & (bin_centers <= hist_range[1])
    
    # Crop the bin centers and histograms by the mask
    bin_centers_cropped = bin_centers[mask]
    ls_sr_hist_cropped = np.array(ls_sr_hist)[mask]
    ls_toa_hist_cropped = np.array(ls_toa_hist)[mask]
    s2_sr_hist_cropped = np.array(s2_sr_hist)[mask]
    s2_toa_hist_cropped = np.array(s2_toa_hist)[mask]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_centers_cropped, ls_sr_hist_cropped, linestyle='-', color='red', label='Landsat SR', linewidth=2)
    ax.plot(bin_centers_cropped, s2_sr_hist_cropped, linestyle='-', color='green', label='Sentinel-2 SR', linewidth=2)
    ax.plot(bin_centers_cropped, ls_toa_hist_cropped, linestyle=':', color='red', label='Landsat TOA', linewidth=4)
    ax.plot(bin_centers_cropped, s2_toa_hist_cropped, linestyle=':', color='green', label='Sentinel-2 TOA', linewidth=4)

    # Add histogram means
    ax.axvline(x=ls_sr_mean, color='red', linestyle='-', linewidth=3)
    ax.axvline(x=s2_sr_mean, color='green', linestyle='-', linewidth=3)
    ax.axvline(x=ls_toa_mean, color='red', linestyle=':', linewidth=3)
    ax.axvline(x=s2_toa_mean, color='green', linestyle=':', linewidth=3)

    # Customize plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {band} values for {date} in {roi}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    #plt.show()

def overlay_sr_toa_hist_otsu(summary_data: pd.DataFrame, roi: str, date: str):
    
    sr = summary_data[
        (summary_data['roi'] == roi) & 
        (summary_data['date'] == date) & 
        (summary_data['level'] == 'sr')
    ]

    toa = summary_data[
        (summary_data['roi'] == roi) & 
        (summary_data['date'] == date) & 
        (summary_data['level'] == 'toa')
    ]

    if sr.empty or toa.empty:
        print('Error: a level is missing')
        return None

    plot_otsu_histograms(sr, toa, date, roi)

def overlay_sr_toa_hist_reflect(regression_data: pd.DataFrame,
                                roi: str,
                                date: str,
                                band_name: str,
                                hist_range: tuple[float, float] = (0.0, 0.1)
                                ):
    
    sr = regression_data[
        (regression_data['roi'] == roi) &
        (regression_data['date'] == date) &
        (regression_data['level'] == 'sr') &
        (regression_data['band_name'] == band_name)
    ]

    toa = regression_data[
        (regression_data['roi'] == roi) &
        (regression_data['date'] == date) &
        (regression_data['level'] == 'toa') &
        (regression_data['band_name'] == band_name)
    ]

    if sr.empty or toa.empty:
        print('Error: a level is missing')
        return None

    plot_reflectance_histograms(sr, toa, band_name, date, roi, hist_range)

