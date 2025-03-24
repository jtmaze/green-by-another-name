"""
Pixel Regression Functions:
These functions generate reflectance comparisons for coincident images. They evaluate how specific
Landsat8 and Sentinel-2 bands diverge throughout zones on the landscape. The PLD mask defines zones as lake, 
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
from img_data_fetching_functions import (
    rio_get_data_arrays_with_common_trans,
    make_measure_mask,
    find_measured_pixels,
    make_ndwi_images, 
    check_match_imgs,
)

random.seed(20)

"""
-----------------------------------------
Functions to get coregistered pixels for regression
-----------------------------------------
"""

def downsample_image_arrays(
    ls_pixels: np.array,
    s2_pixels: np.array,
    sample_size: int
):
    """
    Downsampling makes the pixel regressions more efficient.
    Inputs: 2D images with identical masked pixels = np.nan
    Returns: 1D arrays with randomly downsampled to the sample_size if above pixel count.
    """

    if ls_pixels.shape != s2_pixels.shape:
        raise ValueError(
        f"Incompatible array shapes: Landsat8 shape {ls_pixels.shape} != "
        f"Sentinel-2 shape {s2_pixels.shape}. Images must be resampled to identical dimensions "
    )
    # Create a common mask so that we drop the same pixels in both arrays
    # (exclude NaNs or zeros in either array).
    common_mask = (
        ~np.isnan(ls_pixels) & (ls_pixels != 0) &
        ~np.isnan(s2_pixels) & (s2_pixels != 0)
    )

    # Flatten both arrays using the same mask
    ls_flat = ls_pixels[common_mask].flatten()
    s2_flat = s2_pixels[common_mask].flatten()

    # Double-check that both arrays have the same length after masking
    if ls_flat.size != s2_flat.size:
        raise ValueError(
            "After applying the common mask, the Landsat8 and Sentinel-2 arrays "
            "do not have the same valid pixel count."
        )

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
    
def get_single_band_pixel_samples(
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

def get_ndwi_band_samples(
    image_info: dict,
    pld_fp: str,
    zone: str,
    buffer_delim: int,
    buffer_delim_outer: int,
    sample_size: int
):
    """
    Similar to get_single_band_pixels_samples, but generates samples from the Landsat8 and Sentinel-2
    NDWI images instead of the single band. 
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

"""
----------------------------------------------------
Functions to perform the by-pixel regression
----------------------------------------------------
"""

    
def regression_outlier_filter(
    sample_data: np.array, 
    outlier_frac: float
):
    """
    Replaces the lowest and highest fraction of values in sample_data with numpy.nan.
    For example, if outlier_frac is 0.01, the lower 1% and upper 1% of values are replaced with nan.
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

def regression_vis(
    ls_modeled: np.array, # 1D array
    s2_modeled: np.array, # 1D array
    model: dict, # from the RMA regression
):
    """
    Visualizes the pixel regresion for an image pair. 
    """

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
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=box_props)
    plt.xlabel('Landsat Reflectance')
    plt.ylabel('Sentinel-2 Reflectance')
    plt.legend(loc='lower right')
    plt.show()


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

    sample_size = ls_sample.size # Sample sizes will be the same from downsample_img_arrays()
    # If there's a to few-pixels 
    # This is prone to happen in narrow shoreline zones (e.g. only pixels in PLD buffered 0 to + 30 meters)
    if sample_size < 500:
        print(f'Error: Insuffcient quality pixels given parameters (less than 500)')
        model = 'Poor Quality Image Data'
        model_domain = 'Poor Quality Image Data'
        ls_histogram = s2_histogram = 'Poor Quality Image Data'
        above_frac = below_frac = 'Poor Quality Image Data'

    # With sufficient valid pixels run the pixel regression analysis
    else:
        ls_filtered = regression_outlier_filter(ls_sample, outlier_frac)
        s2_filtered = regression_outlier_filter(s2_sample, outlier_frac)
        
        # Triple chech that both arrays are using same mask
        nan_mask = ~np.isnan(ls_filtered) & ~np.isnan(s2_filtered)
        ls_modeled = ls_sample[nan_mask]
        s2_modeled = s2_sample[nan_mask]

        # Run the RMA regression
        model = rma_regression(ls_modeled, s2_modeled)
        model_domain = (
            np.min([ls_modeled.min(), s2_modeled.min()]), 
            np.max([ls_modeled.max(), s2_modeled.max()])
        )

        regression_vis(ls_modeled, s2_modeled, model)

        if hist_return == True:
            ls_histogram = np.histogram(ls_modeled, bins=100)
            s2_histogram = np.histogram(s2_modeled, bins=100)

        # Find the portion of pixels above/below the 45 degree line   
        below_frac = np.mean(ls_modeled > s2_modeled) * 100  # The mean of boolean array gives proportion
        above_frac = np.mean(ls_modeled < s2_modeled) * 100
        equal_frac = np.mean(ls_modeled == s2_modeled) * 100
        print(f'Pixels above 45 degree line: {above_frac:.2f}%')
        print(f'Pixels below 45 degree line: {below_frac:.2f}%')
        print(f'Pixels equal: {equal_frac:.2f}%')

    return {'model': model, 
            'above_frac': above_frac,
            'below_frac': below_frac,
            'model_domain': model_domain, 
            'ls_histogram': ls_histogram, 
            's2_histogram': s2_histogram}

"""
----------------------------------------------------
Function to aggregate pixel regressions across image pairs
----------------------------------------------------
"""

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
            ls_sample, s2_sample, valid_pix_cnt = get_ndwi_band_samples(
                image_info, pld_fp, **sample_params
            )
        # Get the single band (G or NIR) samples
        else:
            ls_sample, s2_sample, valid_pix_cnt = get_single_band_pixel_samples(
                ls8_fp, s2_fp, pld_fp, band_name=band_name, **sample_params
            )
        # Run regression function
        print(f'{level} {resample_method} regression for {band_name} for date {date} in the {roi} region with PLD {zone} {buffer_delim}m')
        regression_output = regress_reflectance(ls_sample, s2_sample, outlier_frac, hist_return)
    else: # Return "No Image Data" if matching image pairs not found
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


def numpy_to_list(data):
    """
    Converts numpy arrays to lists for storage in a dataframe
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

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
                    # Data histograms
                    'ls_hist_counts': ls_hist_counts,
                    'ls_hist_bins': ls_hist_bins,
                    's2_hist_counts': s2_hist_counts,
                    's2_hist_bins': s2_hist_bins
                }

                regression_summaries.append(summary)

    return pd.DataFrame(regression_summaries)

