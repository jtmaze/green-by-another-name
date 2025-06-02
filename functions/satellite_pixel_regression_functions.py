"""
Satellite Pixel Regression Functions:
These functions generate reflectance comparisons for coincident images. They evaluate how specific
Landsat8 and Sentinel-2 bands diverge throughout zones on the landscape. The PLD mask defines zones as lake, 
shoreline, and land. 
"""
import re
import ast
import pandas as pd
import numpy as np
import random

# ----- Custom Functions -----
from functions.img_data_fetching_functions import (
    rio_get_data_arrays_with_common_trans,
    make_measure_mask,
    find_measured_pixels,
    make_sat_ndwi_images, 
    check_match_imgs,
    downsample_image_arrays
)

from functions.general_pixel_regression_functions import (
    regress_reflectance,
    numpy_to_list
)

random.seed(20)

"""
-----------------------------------------
Functions to get coregistered pixels for regression
-----------------------------------------
"""
    
def get_band_sat_pixel_samples(
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
    Takes file path pair from image_info and returns two down-sampled 1-D arrays of matched pixels from the 
    Landsat8 and Sentinel-2 images pairs
    NOTE: Images need to be resampled on a common grid.
    Order of steps:
    1) Read a specific LandSat8 and Sentinel-2 band as matching numpy arrays
    2) Generate a measurement mask from PLD with dilation and errosion zones specified
    3) Filter both image's array data by the mask and low/high thresholds
    4) Resample the arrays for regression based on a pre-defined sample size
    """
    
    ls_data, s2_data, image_window_params = rio_get_data_arrays_with_common_trans(
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

def get_ndwi_sat_samples(
    image_info: dict,
    pld_fp: str,
    zone: str,
    buffer_delim: int,
    buffer_delim_outer: int,
    sample_size: int
):
    """
    Similar to get_band_sat_pixels_samples, but generates samples from the Landsat8 and Sentinel-2
    NDWI images instead of the single band. 
    Steps:
        1) Generates coincident NDWI images given image info dictionary
        2) Creates a measurement mask from PLD with specified zone and buffer values
        3) Applies the measurement mask to the NDWI images
        4) Downsamples the NDWI images to 1D (flat) arrays with a specified sample size
    """

    ls_ndwi, s2_ndwi, img_window_params = make_sat_ndwi_images(image_info)
    measure_mask = make_measure_mask(pld_fp, img_window_params, zone, buffer_delim, buffer_delim_outer)
    ls_pixels, s2_pixels = find_measured_pixels(ls_ndwi, s2_ndwi, measure_mask)
    ls_sample, s2_sample, valid_pix_cnt = downsample_image_arrays(ls_pixels, s2_pixels, sample_size)

    return ls_sample, s2_sample, valid_pix_cnt


"""
----------------------------------------------------
Function to aggregate pixel regressions across image pairs
----------------------------------------------------
"""

def regress_sat_pairs(
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
    if resample_method == "noresample":
        raise ValueError ("Images must be resampled to a common grid for regression analysis.")

    zone, buffer_delim, buffer_delim_outer = (
        mask_params['zone'],
        mask_params['buffer_delim'],
        mask_params['buffer_delim_outer']
    )
    
    sample_size, outlier_frac = (
        regression_params['sample_size'],
        regression_params['outlier_frac']
    )
    
    # Make file paths from image_info
    s2_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_{level}_date_{date}_roi_{roi}.tif'
    ls8_fp = f'./data/{level}_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_LandSat8_{level}_date_{date}_roi_{roi}.tif'
    res = re.search(r"(\d{2}$)", resample_method).group(1)
    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif'

    # Check if images exist
    if check_match_imgs(ls8_fp, s2_fp):
        sample_params = {
            'zone': zone,
            'buffer_delim': buffer_delim,
            'buffer_delim_outer': buffer_delim_outer,
            'sample_size': sample_size
        }
        # Get the NDWI  samples
        if band_name == "NDWI":
            ls_sample, s2_sample, valid_pix_cnt = get_ndwi_sat_samples(
                image_info, pld_fp, **sample_params
            )
            # NOTE:
            # Becuase we kept negative reflectance values NDWI can be enormously negative or positive
            # For regression plots, we bound NDWI at [-1, 1]
            ls_sample = np.clip(ls_sample, -1, 1)
            s2_sample = np.clip(s2_sample, -1, 1)
            ls_marginal_percent = np.sum((ls_sample > -0.25) & (ls_sample < 0.25)) / len(ls_sample) * 100
            s2_marginal_percent = np.sum((s2_sample > -0.25) & (s2_sample < 0.25)) / len(s2_sample) * 100
        # Get the single band (G or NIR) samples
        else:
            ls_sample, s2_sample, valid_pix_cnt = get_band_sat_pixel_samples(
                ls8_fp, s2_fp, pld_fp, band_name=band_name, **sample_params
            )
            ls_marginal_percent = np.sum(ls_sample < 0.1) / len(ls_sample) * 100
            s2_marginal_percent = np.sum(s2_sample < 0.1) / len(s2_sample) * 100

        # Run regression function
        print(f'{level} {resample_method} regression for {band_name} for date {date} in the {roi} region with PLD {zone} {buffer_delim}m')
        regression_output = regress_reflectance(ls_sample, s2_sample, outlier_frac, hist_return, comparison='Satellite')
    else: # Return "No Image Data" if matching image pairs not found
        ls_marginal_percent = s2_marginal_percent = valid_pix_cnt = regression_output = "No Image Data"

    if not isinstance(ls_marginal_percent, str):
        print(f'LS8 marginal percent: {ls_marginal_percent:.2f}% for {band_name} band')
        print(f'S2 marginal percent: {s2_marginal_percent:.2f}% for {band_name} band')

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
        'regression_output': regression_output, 
        'ls_marginal_percent': ls_marginal_percent,  
        's2_marginal_percent': s2_marginal_percent,  
    }

def make_satellite_reflectance_summaries(
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
                regression_result = regress_sat_pairs(
                    image_info, mask_params, regression_params, hist_return=hist_return
                )
                # Handle case when no image data is available
                regression_items_str = regression_result['regression_output']
                if regression_items_str == 'No Image Data':
                    model = slope = intercept = r_squared = "No Image Data"
                    above_frac = below_frac = model_domain = "No Image Data"
                    ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "No Image Data"
                    ls_marginal_percent = "No Image Data"
                    s2_marginal_percent = "No Image Data"
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
                        ls_marginal_percent = regression_result['ls_marginal_percent']
                        s2_marginal_percent = regression_result['s2_marginal_percent']

                        # Handle histogram data if requested
                        if hist_return:
                            ls_hist_counts = numpy_to_list(regression_items.get('arr1_histogram')[0])
                            ls_hist_bins = numpy_to_list(regression_items.get('arr1_histogram')[1])
                            s2_hist_counts = numpy_to_list(regression_items.get('arr2_histogram')[0])
                            s2_hist_bins = numpy_to_list(regression_items.get('arr2_histogram')[1])
                        else:
                            ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "Histogram Not Returned"
                    else:
                        # Set default values for poor quality images
                        slope = intercept = r_squared = "Poor Quality Image Data"
                        model_domain = "Poor Quality Image Data"
                        above_frac = below_frac = "Poor Quality Image Data"
                        ls_hist_counts = ls_hist_bins = s2_hist_counts = s2_hist_bins = "Poor Quality Image Data"
                        ls_marginal_percent = s2_marginal_percent = "Poor Quality Image Data"

                # Create summary dictionary for this image pair
                summary = {
                    # Image/pld zone information
                    'level': regression_result['level'],
                    'resample_method': regression_result['resample_method'],
                    'roi': regression_result['roi'],
                    'date': regression_result['date'],
                    'band_name': regression_result['band_name'], 
                    'buffer_delim': regression_result['buffer_delim'],
                    'buffer_delim_outer': regression_result['buffer_delim_outer'],
                    # Regression params/output
                    'sample_size': regression_result['sample_size'],
                    'outlier_frac': regression_result['outlier_frac'], 
                    'valid_pix_cnt': regression_result['valid_pix_cnt'],
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'above_frac': above_frac,
                    'below_frac': below_frac,
                    'model_domain': model_domain,
                    'ls_marginal_percent': ls_marginal_percent,
                    's2_marginal_percent': s2_marginal_percent,
                    # Data histograms
                    'ls_hist_counts': ls_hist_counts,
                    'ls_hist_bins': ls_hist_bins,
                    's2_hist_counts': s2_hist_counts,
                    's2_hist_bins': s2_hist_bins
                }

                regression_summaries.append(summary)

    return pd.DataFrame(regression_summaries)

