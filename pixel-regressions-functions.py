# %% 1.0

import random
from typing import Optional
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

###############################################
### Helper Functions
###############################################
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

    # Get the data's (LandSat's) bounds and transform as a window for the PLD mask
    with rio.open(ls_path) as src: 
        meta = src.meta
        ls_bounds = src.bounds
        ls_transform = src.transform
        ls_shape = src.shape

        print(ls_bounds)
        print(ls_transform)

    image_window_params = {
        'bounds': ls_bounds,
        'transform': ls_transform,
        'shape': ls_shape
    }
    
    # The data should have the same shapes to broadcast together.
    if ls_data.shape != s2_data.shape:
        print('The Sentinel-2 and Landsat8 data shapes are different, cannot compare')

    return ls_data, s2_data, image_window_params, meta
    

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

def filter_measured_pixels(ls_data: np.array,
                           s2_data: np.array,
                           measure_mask: np.array,
                           filter_low: float,
                           filter_high: float,
                           out_meta: dict):
    """
    Opperations:
    1) Selects shoreline, lake, or land pixels within measure mask
    2) Filters pixel values between thresholds for both image arrays
    3) Ensures both images have common set of nans after filtering. 
    """
    if out_meta is None: 
        ls_masked = apply_measure_mask(ls_data, measure_mask)
        s2_masked = apply_measure_mask(s2_data, measure_mask)
        ls_filtered = np.where(
            (ls_masked > filter_low) & (ls_masked < filter_high), 
            ls_masked, 
            np.nan
        )
        valid_ls_mask = ~np.isnan(ls_filtered)
        s2_filtered = np.where(
            (s2_masked > filter_low) & (s2_masked < filter_high),
            s2_masked,
            np.nan
        )
        valid_s2_mask = ~np.isnan(s2_filtered)

        # Make the same nan values from filtering common to each dataset
        ls_data_out = np.where(valid_s2_mask, ls_filtered, np.nan)
        s2_data_out = np.where(valid_ls_mask, s2_filtered, np.nan)

        return ls_data_out, s2_data_out
    
    if out_meta is not None:
        out_path1 = './data/testLS.tiff'
        out_path2 = './data/testS2.tiff'
        ls_masked = apply_measure_mask(ls_data, measure_mask)
        s2_masked = apply_measure_mask(s2_data, measure_mask)
        ls_filtered = np.where(
            (ls_masked > filter_low) & (ls_masked < filter_high), 
            ls_masked, 
            np.nan
        )
        valid_ls_mask = ~np.isnan(ls_filtered)
        s2_filtered = np.where(
            (s2_masked > filter_low) & (s2_masked < filter_high),
            s2_masked,
            np.nan
        )
        valid_s2_mask = ~np.isnan(s2_filtered)

        # Make the same nan values from filtering common to each dataset
        ls_data_out = np.where(valid_s2_mask, ls_filtered, np.nan)
        s2_data_out = np.where(valid_ls_mask, s2_filtered, np.nan)

        with rio.open(out_path1, 'w', **out_meta) as dst1:
            dst1.write(ls_data_out, 1)

        with rio.open(out_path2, 'w', **out_meta) as dst2:
            dst2.write(s2_data_out, 1)

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

    if ls_flat.size < sample_size:
        print(f"Not downsampling the number of measured pixels {ls_flat.size} < {sample_size}")
        return ls_flat, s2_flat
    else:
        sample_idx = np.random.choice(ls_flat.size, sample_size, replace=False)
        # Applies sample_idx to pixels_flat
        ls_sampled = ls_flat[sample_idx]
        s2_sampled = s2_flat[sample_idx]
        return ls_sampled, s2_sampled
    

def regress_reflectance(
        ls_sample: np.array, # Array should be 1D i.e. flat
        s2_sample: np.array): # 1D array
    
    print(ls_sample.mean())
    print(s2_sample.mean())
    model = stats.linregress(ls_sample, s2_sample)
    slope = model[0]
    intercept = model[1]
    r_squared = model[2] ** 2 #r_squared is the 3rd value in model tupple
    # Make a fit-line from the model
    # Becuase Landsat is x-axis use it to make the domain
    xmin_val = np.min(ls_sample)
    xmax_val = np.max(ls_sample)
    ymin_val = np.min(s2_sample)
    ymax_val = np.max(s2_sample)
    min_modeled = slope * xmin_val + intercept
    max_modeled = slope * xmax_val + intercept

    plt.figure(figsize=(8,6))
    plt.scatter(ls_sample, s2_sample, s=1, marker='.')
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
    
    return model

def get_pixel_samples(ls8_path: str,
                      s2_path: str,
                      pld_path: str,
                      band_name: str,
                      sample_size: int,
                      zone: str,
                      buffer_delim: int,
                      buffer_delim_outer: int,
                      filter_low: float,
                      filter_high: float):
    
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
    ls_pixels, s2_pixels = filter_measured_pixels(
        ls_data, 
        s2_data, 
        measure_mask, 
        filter_low,
        filter_high
    )
    ls_sample, s2_sample = downsample_image_arrays(
        ls_pixels, 
        s2_pixels, 
        sample_size
    )

    return ls_sample, s2_sample


# %% 

test_fp_s2 = './data/sr_images/Sentinel2-sr_date_2019-05-16_roi_YKF_sub1_resampled_bilinear30.tif'
test_fp_ls8 = './data/sr_images/Landsat8-sr_date_2019-05-16_roi_YKF_sub1_resampled_bilinear30.tif'
pld_path = './data/pld_rasterized/YKF_sub1_lake_masks.tif'

ls_data, s2_data, image_window_params, meta = rio_get_data_arrays(test_fp_ls8, test_fp_s2, band_name='Green')
measure_mask = make_measure_mask(pld_path, 
                                 image_window_params, 
                                 zone='lake', 
                                 buffer_delim=60, 
                                 buffer_delim_outer=None)

ls_pixels, s2_pixels = filter_measured_pixels(ls_data, s2_data, measure_mask, filter_low=0.0000001, filter_high=0.999999, out_meta=meta)


print(ls_pixels.size, s2_pixels.size)
print()
sample_size = 100
ls_sample, s2_sample = downsample_image_arrays(ls_pixels, s2_pixels, sample_size) 
rsq = regress_reflectance(ls_sample, s2_sample)

# %%

green_ls_sample, green_s2_sample = get_pixel_samples(
    ls8_path=test_fp_ls8,
    s2_path=test_fp_s2,
    pld_path=pld_path,
    band_name='Green',
    sample_size=10_000,
    zone='lake',
    buffer_delim=0,
    buffer_delim_outer=None,
    filter_low=0.00001,
    filter_high=0.99999,
)

nir_ls_sample, nir_s2_sample = get_pixel_samples(
    ls8_path=test_fp_ls8,
    s2_path=test_fp_s2,
    pld_path=pld_path,
    band_name='NIR',
    sample_size=10_000,
    zone='lake',
    buffer_delim=0,
    buffer_delim_outer=None,
    filter_low=0.00001,
    filter_high=0.99999,
)

# %%
ndwi_ls_sample = np.divide(
    (green_ls_sample - nir_ls_sample),
    (green_ls_sample + nir_ls_sample),
    out=np.zeros_like(green_ls_sample),
    where=(green_ls_sample + nir_ls_sample) != 0
)

ndwi_s2_sample = np.divide(
    (green_s2_sample - nir_s2_sample),
    (green_s2_sample + nir_s2_sample),
    out=np.zeros_like(green_s2_sample),
    where=(green_s2_sample + nir_s2_sample) != 0
)

rsq = regress_reflectance(ndwi_ls_sample, ndwi_s2_sample)


# %%
