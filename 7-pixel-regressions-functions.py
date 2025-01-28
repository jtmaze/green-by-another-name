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

import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling


random.seed(20)

# %% 

"""
####################################
------------------------------------
Processing functions
------------------------------------
####################################
"""


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
        regression_domain: tuple # (Lower, Upper)
    ): 
    sample_size = ls_sample.size
    # Remove samples outside the modeling domain
    lower, upper = regression_domain
    domain_mask = (ls_sample > lower) & (ls_sample < upper) & (s2_sample > lower) & (s2_sample < upper)
    
    # Filter both arrays using same mask
    ls_modeled = ls_sample[domain_mask]
    s2_modeled = s2_sample[domain_mask]

    if sample_size < 5_000:
        print(f'Error: Insuffcient quality pixels given parameter (less than 1000)')
        model = 'Poor Quality Data'
        excluded_frac = 'Poor Quality Image Data'

    else: 
        # Count excluded samples
        excluded_frac = ((sample_size - len(ls_modeled)) / sample_size) * 100
        
        model = stats.linregress(ls_modeled, s2_modeled)
        slope = model[0]
        intercept = model[1]
        r_squared = model[2] ** 2 #r_squared is the 3rd value in model tuple

        # Make a fit-line from the model
        # Becuase Landsat is x-axis use it to make the domain
        xmin_val = np.nanmin(ls_modeled)
        xmax_val = np.nanmax(ls_modeled)
        ymin_val = np.nanmin(s2_modeled)
        ymax_val = np.nanmax(s2_modeled)
        min_modeled = slope * xmin_val + intercept
        max_modeled = slope * xmax_val + intercept

        plt.figure(figsize=(8,6))
        plt.scatter(ls_modeled, s2_modeled, s=1, marker='.')
        plt.plot([xmin_val, xmax_val], [min_modeled, max_modeled], color = 'red', linestyle='-', label='OLS Fit')
        # Add a 45 degree line for comparison
        plt.plot([min(xmin_val, ymin_val), max(xmax_val, ymax_val)], 
                [min(xmin_val, ymin_val), max(xmax_val, ymax_val)], 
                color='blue', 
                linestyle='--', 
                label='1:1 Line')
        textstr = f'$R^2 = {r_squared:.4f}$\nSlope = {slope:.4f}'
        box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=box_props)
        plt.xlabel('Landsat Reflectance')
        plt.ylabel('Sentinel-2 Reflectance')
        plt.legend(loc='lower right')
        plt.show()

        if excluded_frac > 10:
            print(f'Warning fraction of pixels excluded from the model domain is high ({excluded_frac}%)')
    
    return model, excluded_frac

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
                        regression_params: dict
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
    
    sample_size, model_domain = (
        regression_params['sample_size'],
        regression_params['model_domain']
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
        model, excluded_frac = regress_reflectance(ls_sample, s2_sample, model_domain)
    else: # Return no image data if matching images not found
        model = excluded_frac = valid_pix_cnt = "No Image Data"

    return {
        'level': level,
        'date': date,
        'roi': roi,
        'band_name': band_name,
        'zone': zone,
        'buffer_delim': buffer_delim,
        'buffer_delim_outer': buffer_delim_outer,
        'sample_size': sample_size,
        'model_domain': model_domain,
        'model': model,
        'excluded_frac': excluded_frac,
        'valid_pix_cnt': valid_pix_cnt
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

    plt.imshow(ls_ndwi, cmap='viridis')
    plt.title('Landsat NDWI')
    plt.colorbar()
    plt.show()

    plt.imshow(s2_ndwi, cmap='viridis')
    plt.title('Sentinel-2 NDWI')
    plt.colorbar()
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


def find_otsu_threshold(
        ndwi: np.array, 
        show_hist: bool,
    ):
    """
    Input: NDWI array [-1,1]
    Output: Otsu Threshold, and possibly the data
    """
    
    flat_ndwi = ndwi.flatten()
    valid_mask = ~np.isnan(flat_ndwi)
    valid_data = flat_ndwi[valid_mask]
    valid_data = np.clip(valid_data, -1, 1)

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
        data_return: bool, 
    ):
    """
    Returns a dictionary with the otsu threshold, and water fraction from each image
    """

    # Initialize all variables at the start
    ls_threshold = None
    ls_water_frac = None 
    s2_threshold = None
    s2_water_frac = None
    ls_data = None
    s2_data = None

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

            if data_return == True:
                ls_flat = ls_ndwi.flatten()
                valid_ls = ~np.isnan(ls_flat)
                ls_data = ls_flat[valid_ls]
                ls_data = np.clip(ls_data, -1, 1)

                s2_flat = s2_ndwi.flatten()
                valid_s2 = ~np.isnan(s2_flat)
                s2_data = s2_flat[valid_s2]
                s2_data = np.clip(s2_data, -1, 1)
            else:
                s2_data = ls_data = None

    else:
        ls_threshold = ls_water_frac = s2_threshold = s2_water_frac = ls_data = s2_data = 'No Image Data'

    return {'ls_threshold': ls_threshold, 
            'ls_water_frac': ls_water_frac, 
            's2_threshold': s2_threshold, 
            's2_water_frac': s2_water_frac, 
            'ls_data': ls_data, 
            's2_data': s2_data}

# %% Run the functions:
"""
####################################
------------------------------------
Make dictionaries to hold parameters
------------------------------------
####################################
"""

image_info = {
    'level': 'toa',
    'date': '2021-07-01', # Dates will be itterated through
    'roi': 'YKF_sub1', # ROIs will be itterated through
    'band_name': 'Green'
}
mask_params = {
    'zone': 'shoreline',
    'buffer_delim': -60,
    'buffer_delim_outer': 60,
}
regression_params = {
    'sample_size': 10_000,
    'model_domain': (-1, 1) 
}

# %%

"""
####################################
------------------------------------
Gather all the combinations of rois and dates
------------------------------------
####################################
"""

toa_files = glob.glob('./data/toa_images/*tif')
sr_files = glob.glob('./data/sr_images/*.tif')
full_files = toa_files + sr_files

date_pattern = r'_date_(.*?)_roi'
roi_pattern = r'_roi_(.*?)_resampled'

def extract_unique(files: list, pattern: re.Pattern[str]):
    unique_items = set()
    for f in files:
        match = re.search(pattern, f)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

image_dates = extract_unique(full_files, date_pattern)
rois = extract_unique(full_files, roi_pattern)
levels = ['sr', 'toa']

# %%

"""
####################################
------------------------------------
Make Green Band Regression Models
------------------------------------
####################################
"""

image_info['band_name'] = 'Green'
regression_summaries = []

for level in levels:
    for roi in rois:
            for dt in image_dates:

                image_info['level'] = level
                image_info['roi'] = roi
                image_info['date'] = dt

                regression_params['model_domain'] = (0, 0.7)

                summary = regress_image_pairs(
                    image_info=image_info,
                    mask_params=mask_params,
                    regression_params=regression_params
                )

                regression_summaries.append(summary)

print("Done making regression summaries")

# %% 

"""
####################################
------------------------------------
Make NIR Band Regression Models
------------------------------------
####################################
"""

image_info['band_name'] = 'NIR'
regression_summaries = []

for level in levels:
    for roi in rois:
            for dt in image_dates:

                image_info['level'] = level
                image_info['roi'] = roi
                image_info['date'] = dt

                regression_params['model_domain'] = (0, 0.7)

                summary = regress_image_pairs(
                    image_info=image_info,
                    mask_params=mask_params,
                    regression_params=regression_params
                )

                regression_summaries.append(summary)

print("Done making regression summaries")

# %%

"""
####################################
------------------------------------
Make NDWI Regression Models
------------------------------------
####################################
"""

image_info['band_name'] = 'NDWI'
regression_summaries = []

for level in levels:
    for roi in rois:
            for dt in image_dates:

                image_info['level'] = level
                image_info['roi'] = roi
                image_info['date'] = dt

                regression_params['model_domain'] = (-1, 1)

                summary = regress_image_pairs(
                    image_info=image_info,
                    mask_params=mask_params,
                    regression_params=regression_params
                )

                regression_summaries.append(summary)

print("Done making regression summaries")


# %%

df_regression_summary = pd.DataFrame(regression_summaries)
df_regression_summary.to_csv('./data/regression_results_60m_-60m_shoreline.csv', index=False)
#regression_summary_clean = df_regression_summary[df_regression_summary['excluded_frac'] != 'No Image Data']

# %%
image_info = {
    'level': 'toa',
    'date': '2021-07-01', # Dates will be itterated through
    'roi': 'YKF_sub1', # ROIs will be itterated through
    'band_name': 'Green'
}

rois = extract_unique(full_files, roi_pattern)
levels = ['sr', 'toa']
image_dates = ['2021-09-09', '2021-08-02', '2020-07-15', '2024-06-15']

# %%

"""
####################################
------------------------------------
Calculate water area for different images
------------------------------------
####################################
"""

area_summaries = []

for level in levels:
    for roi in rois:
        for dt in image_dates:
            image_info['date'] = dt
            image_info['roi'] = roi
            image_info['level'] = level



            otsu_items = otsu_image_wtr_area(image_info, write_mask=False, data_return=True)
            if otsu_items['ls_threshold'] == 'No Image Data' or otsu_items['ls_threshold'] == 'Poor Quality Image':
                ls_s2_percent_diff = 'No Image Data'
            else:
                ls_s2_percent_diff = (
                    (otsu_items['ls_water_frac'] - otsu_items['s2_water_frac']) 
                    / otsu_items['ls_water_frac']
                ) * 100

            summary = {
                'date': dt,
                'roi': roi,
                'level': level,
                'otsu_items': otsu_items,
                'ls_s2_percent_diff': ls_s2_percent_diff
            }

            area_summaries.append(summary)

print("Done calculating water area")

# %%
df_area_summary = pd.DataFrame(area_summaries)
df_area = df_area_summary[df_area_summary['ls_s2_percent_diff'] != 'No Image Data']
df_area.head(20)

# %% 
   
def overlay_sr_toa_cdf(summary_data: pd.DataFrame, roi: str, date: str):
    
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
    
    # Access dictionary elements using iloc[0] and key names
    sr_threshold = sr['otsu_items'].iloc[0]['ls_threshold']
    sr_data = sr['otsu_items'].iloc[0]['ls_data'] 
    toa_threshold = toa['otsu_items'].iloc[0]['ls_threshold']
    toa_data = toa['otsu_items'].iloc[0]['ls_data']

    def plot_cdfs(sr_data, toa_data, sr_threshold, toa_threshold):
    # Calculate CDFs
        sr_sorted = np.sort(sr_data)
        toa_sorted = np.sort(toa_data[~np.isnan(toa_data)])
        
        # Create points for CDF
        sr_p = np.linspace(0, 1, len(sr_sorted))
        toa_p = np.linspace(0, 1, len(toa_sorted))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot CDFs
        ax.plot(sr_sorted, sr_p, 'b-', label='SR')
        ax.plot(toa_sorted, toa_p, 'r-', label='TOA')
        
        # Add threshold lines
        ax.axvline(x=sr_threshold, color='b', linestyle='--', label='SR Threshold')
        ax.axvline(x=toa_threshold, color='r', linestyle='--', label='TOA Threshold')
        
        # Customize plot
        ax.set_xlabel('NDWI Value')
        ax.set_ylabel('Cumulative')
        ax.set_title('CDF of NDWI Values')
        ax.legend(loc='lower left')
        
        plt.grid(True)
        plt.show()

    plot_cdfs(sr_data, toa_data, sr_threshold, toa_threshold)

    return None


# %%

overlay_sr_toa_cdf(summary_data=df_area_summary, roi='AKCP_sub1', date='2020-07-15')


# %% 

df_area_summary = pd.DataFrame(area_summaries)
df_area_clean = df_area_summary[df_area_summary['ls_s2_percent_diff'] != 'No Image Data']
print(df_area_clean)

#df_area_summary.to_csv('./data/area_results.csv', index=False)
