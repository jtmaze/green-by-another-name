"""
General pixel regresssion functions are used by both
ac_pixel_regression_functions.py and satellite_pixel_regression_functions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

    
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

def rma_regression(
    x: np.array, 
    y: np.array
):
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
    arr1_modeled: np.array, # 1D array
    arr2_modeled: np.array, # 1D array
    model: dict, # from the RMA regression
    comparison: str # used for plot title
):
    """
    Visualizes the pixel regresion for an image pair. 
    """

    # Make a fit-line from the model
    # Becuase Landsat is x-axis use it to make the domain
    xmin_val = np.nanmin(arr1_modeled)
    xmax_val = np.nanmax(arr2_modeled)
    ymin_val = np.nanmin(arr2_modeled)
    ymax_val = np.nanmax(arr2_modeled)
    min_modeled = model['slope'] * xmin_val + model['intercept']
    max_modeled = model['slope'] * xmax_val + model['intercept']

    plt.figure(figsize=(8,6))
    plt.scatter(arr1_modeled, arr2_modeled, s=1, marker='.', alpha=0.4)
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
    if comparison == 'AC':
        xlab = 'TOA Reflectance'
        ylab = 'SR Reflectance'
    elif comparison == 'Satellite':
        xlab = 'Landsat Reflectance'
        ylab = 'Sentinel-2 Reflectance'
        
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='lower right')
    plt.show()


def regress_reflectance(
    arr1_sample: np.array, # Array should be 1D i.e. flat
    arr2_sample: np.array, # 1D array
    outlier_frac: float, # The fraction of outliers to remove
    hist_return: bool,
    comparison: str, # used for making plot labels should be either satellite or AC 
): 

    model = None
    model_domain = None
    arr1_histogram = arr2_histogram = None
    above_frac = below_frac = None

    sample_size = arr1_sample.size # Sample sizes will be the same from downsample_img_arrays()
    # If there's a to few-pixels 
    # This is prone to happen in narrow shoreline zones (e.g. only pixels in PLD buffered 0 to + 30 meters)
    if sample_size < 1500: # TODO: how to make this in relation to number of relevant pixels
        print(f'Error: Insuffcient quality pixels given parameters (less than 1500)')
        model = 'Poor Quality Image Data'
        model_domain = 'Poor Quality Image Data'
        arr1_histogram = arr2_histogram = 'Poor Quality Image Data'
        above_frac = below_frac = 'Poor Quality Image Data'

    # With sufficient valid pixels run the pixel regression analysis
    else:
        arr1_filtered = regression_outlier_filter(arr1_sample, outlier_frac)
        arr2_filtered = regression_outlier_filter(arr2_sample, outlier_frac)
        
        # Triple chech that both arrays are using same mask
        nan_mask = ~np.isnan(arr1_filtered) & ~np.isnan(arr2_filtered)
        arr1_modeled = arr1_sample[nan_mask]
        arr2_modeled = arr2_sample[nan_mask]

        # Run the RMA regression
        model = rma_regression(arr1_modeled, arr2_modeled)
        model_domain = (
            np.min([arr1_modeled.min(), arr2_modeled.min()]), 
            np.max([arr1_modeled.max(), arr2_modeled.max()])
        )

        regression_vis(arr1_modeled, arr2_modeled, model, comparison)

        if hist_return == True:
            arr1_histogram = np.histogram(arr1_modeled, bins=100)
            arr2_histogram = np.histogram(arr2_modeled, bins=100)

        # Find the portion of pixels above/below the 45 degree line   
        below_frac = np.mean(arr1_modeled > arr2_modeled) * 100  # The mean of boolean array gives proportion
        above_frac = np.mean(arr1_modeled < arr2_modeled) * 100

        print(f'Pixels above 45 degree line: {above_frac:.2f}%')
        print(f'Pixels below 45 degree line: {below_frac:.2f}%')

    return {'model': model, 
            'above_frac': above_frac,
            'below_frac': below_frac,
            'model_domain': model_domain, 
            'arr1_histogram': arr1_histogram, 
            'arr2_histogram': arr2_histogram}

def numpy_to_list(data):
    """
    Converts numpy arrays to lists for storage in a dataframe.
    Most usefull when returning the histograms
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data