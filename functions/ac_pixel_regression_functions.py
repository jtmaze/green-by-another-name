"""
Atmospheric Correction Pixel Regression Functions:
These functions generate reflectance comparisons for coincident images. They evaluate how specific
TOA and SR bands diverge throughout zones on the landscape. The PLD mask defines zones as lake, 
shoreline, and land. 
"""

import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

# ----- Custom Functions -----
from functions.img_data_fetching_functions import (
    rio_get_ac_arrays,
    make_measure_mask,
    find_measured_pixels,
    make_ac_ndwi_images, 
    check_match_imgs,
    downsample_image_arrays
)

from functions.general_pixel_regressoin_functions import (
    regress_reflectance,
    numpy_to_list
)

random.seed(20)

def get_ndwi_ac_samples(
    image_info: dict,
    pld_fp: str,
    zone: str, 
    buffer_delim: int,
    buffer_delim_outer: int, 
    sample_size: int
):
    """
    Similar to get_band_ac_pixel_samples(), but generates samples from the TOA and SR
    NDWI images instead of the single band. 
    Steps:
        1) Generates coincident NDWI images given image info dictionary
        2) Creates a measurement mask from PLD with specified zone and buffer values
        3) Applies the measurement mask to the NDWI images
        4) Downsamples the NDWI images to 1D (flat) arrays with a specified sample size
    """
    
    toa_ndwi, sr_ndwi, image_window_params = make_ac_ndwi_images(image_info)
    measure_mask = make_measure_mask(pld_fp, image_window_params, zone, buffer_delim, buffer_delim_outer)
    toa_pixels, sr_pixels = find_measured_pixels(toa_ndwi, sr_ndwi, measure_mask)
    toa_sample, sr_sample, valid_pix_cnt = downsample_image_arrays(toa_pixels, sr_pixels, sample_size)

    return toa_sample, sr_sample, valid_pix_cnt
    
def get_band_ac_pixel_samples(
    toa_path: str,
    sr_path: str,
    pld_path: str,
    band_name: str,
    sample_size: int, 
    zone: str, 
    buffer_delim: int, 
    buffer_delim_outer: int
):
    """
    Takes file path pair from image_info and returns two down-sampled 1-D arrays of matched pixels from the 
    TOA and SR images pairs
    Order of steps:
    1) Read a specific TOA and SR band as matching numpy arrays
    2) Generate a measurement mask from PLD with dilation and errosion zones specified
    3) Filter both image's array data by the mask and low/high thresholds
    4) Resample the arrays for regression based on a pre-defined sample size
    """
    
    toa_data, sr_data, image_window_params = rio_get_ac_arrays(
        toa_path=toa_path,
        sr_path=sr_path, 
        band_name=band_name
    )

    measure_mask = make_measure_mask(
        pld_path=pld_path,
        image_window_params=image_window_params,
        zone=zone,
        buffer_delim=buffer_delim,
        buffer_delim_outer=buffer_delim_outer
    )

    toa_pixels, sr_pixels = find_measured_pixels(
        toa_data,
        sr_data,
        measure_mask
    )

    toa_sample, sr_sample, valid_pix_cnt = downsample_image_arrays(
        toa_pixels,
        sr_pixels,
        sample_size
    )

    return toa_sample, sr_sample, valid_pix_cnt

def regress_ac_pairs(
    image_info: dict, 
    mask_params: dict,
    regression_params: dict,
    hist_return: bool
):
    # Extract parameters to find file path
    sat, roi, date, band_name, resample_method = (
        image_info['satellite'],
        image_info['roi'],
        image_info['date'],
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

    # Find the file paths
    toa_fp = f'./data/toa_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_{sat}_toa_date_{date}_roi_{roi}.tif'
    sr_fp = f'./data/sr_images/roi_{roi}_resampled_{resample_method}/reprojected_{resample_method}_{sat}_sr_date_{date}_roi_{roi}.tif'

    res = re.search(r"(\d{2}$)", resample_method).group(1)
    pld_fp = f'./data/pld_rasterized/{roi}_lake_masks_res{res}.tif'

    if check_match_imgs(toa_fp, sr_fp) == False:
        valid_pix_cnt = regression_output = "No Image Data"
    else:
        sample_params = {
            'zone': zone,
            'buffer_delim': buffer_delim,
            'buffer_delim_outer': buffer_delim_outer,
            'sample_size': sample_size
        }
        # NDWI samples requires getting two bands
        if band_name == 'NDWI':
            toa_sample, sr_sample, valid_pix_cnt = get_ndwi_ac_samples(
                image_info, pld_fp, **sample_params
            )
        # Get single band (G or NIR) samples
        else: 
            toa_sample, sr_sample, valid_pix_cnt = get_band_ac_pixel_samples(
                toa_fp, sr_fp, pld_fp, band_name=band_name, **sample_params
            )
        
        print(f'{sat} {resample_method} regression for {band_name} for date {date} in the {roi} region with PLD {zone} {buffer_delim}m')
        regression_output = regress_reflectance(toa_sample, sr_sample, outlier_frac, hist_return, comparison='AC')

    return {
        'satellite': sat,
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

def make_ac_reflectance_summaries(
    image_info: dict,
    mask_params: dict,
    regression_params: dict,
    satellites: list, # Should just be Sentinel2 or Landsat8,
    rois: list,
    dates:list,
    hist_return: bool = False
):
    regression_summaries = []

    for sat in satellites:
        image_info['satellite'] = sat
        for r in rois:
            image_info['roi'] = r
            for d in dates:
                image_info['date'] = d

                regression_result = regress_ac_pairs(
                    image_info, mask_params, regression_params, hist_return=hist_return
                )
                # Handle case when no image data is available
                regression_items_str = regression_result['regression_output']
                if regression_items_str == 'No Image Data':
                    model = slope = intercept = r_squared = "No Image Data"
                    above_frac = below_frac = model_domain = "No Image Data"
                    toa_hist_counts = toa_hist_bins = sr_hist_counts = sr_hist_bins = "No Image Data"
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
                            toa_hist_counts = numpy_to_list(regression_items.get('arr1_histogram')[0])
                            toa_hist_bins = numpy_to_list(regression_items.get('arr1_histogram')[1])
                            sr_hist_counts = numpy_to_list(regression_items.get('arr2_histogram')[0])
                            sr_hist_bins = numpy_to_list(regression_items.get('arr2_histogram')[1])
                        else:
                            toa_hist_counts = toa_hist_bins = sr_hist_counts = sr_hist_bins = "Histogram Not Returned"
                    else:
                        # Set default values for poor quality images
                        slope = intercept = r_squared = "Poor Quality Image Data"
                        model_domain = "Poor Quality Image Data"
                        above_frac = below_frac = "Poor Quality Image Data"
                        toa_hist_counts = toa_hist_bins = sr_hist_counts = sr_hist_bins = "Poor Quality Image Data"

                # Create summary dictionary for this image pair
                summary = {
                    # Image/pld zone information
                    'satellite': regression_result['satellite'],
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
                    # Data histograms
                    'toa_hist_counts': toa_hist_counts,
                    'toa_hist_bins': toa_hist_bins,
                    'sr_hist_counts': sr_hist_counts,
                    'sr_hist_bins': sr_hist_bins
                }

                regression_summaries.append(summary)

    return pd.DataFrame(regression_summaries)
