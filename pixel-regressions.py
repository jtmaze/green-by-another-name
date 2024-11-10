# %% 1.0 Libraries and file paths
import random
import rasterio as rio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd

from scipy import stats
import matplotlib.pyplot as plt

random.seed(69)

pld_mask_path = './data/pld_shapes/pld_masks.tif'
masked_green_path = './data/masked_images/green-masked.tif'

with rio.open(pld_mask_path) as pld_src:
    pld_band_count = pld_src.count
    for band_idx in range(1, pld_band_count + 1):
        description = pld_src.descriptions[band_idx - 1]
        print(f'Band {band_idx} is PLD {description}')

# %% Some functions

def threshold_histogram_pixels(pixel_data, high_threshold, color, satellite, location):
    # Flatten the pixel data
    pixel_data_flat = pixel_data.flatten()
    
    # Replace zeros with NaN
    pixel_data_flat = np.where(pixel_data_flat != 0, pixel_data_flat, np.nan)
    
    # Calculate the percentage of high values
    high_values = np.sum(pixel_data_flat > high_threshold)
    total_values = np.sum(~np.isnan(pixel_data_flat))
    if total_values > 0:
        percentage_removed = (high_values / total_values) * 100
    else:
        percentage_removed = 0
    print(f'Removed {percentage_removed:.2f}% of high pixels from {satellite}')
    
    # Remove values above high_threshold
    pixel_data_flat = np.where(pixel_data_flat < high_threshold, pixel_data_flat, np.nan)
    # Remove NaN values
    pixel_data_flat = pixel_data_flat[~np.isnan(pixel_data_flat)]
    
    # Plot histogram
    plt.hist(pixel_data_flat, bins=100, alpha=0.5)
    plt.xlim(0, high_threshold)
    plt.title(f'{satellite} {color.capitalize()} Reflectance {location}')
    plt.xlabel('Reflectance')
    plt.ylabel('Frequency')
    plt.show()

def downsample_regress_reflectance(ls_data, s2_data, ls_high_threshold, s2_high_threshold, sample_size, location, color, point_alpha=0.01):
    # Apply threshold to high values and zero values
    band_ls = np.where(ls_data < ls_high_threshold, ls_data, np.nan)
    band_ls = np.where(band_ls > 0, band_ls, np.nan)
    band_s2 = np.where(s2_data < s2_high_threshold, s2_data, np.nan)
    band_s2 = np.where(band_s2 > 0, band_s2, np.nan)

    # Make Nan Values common to both datasets
    band_ls = np.where(~np.isnan(band_s2), band_ls, np.nan)
    band_s2 = np.where(~np.isnan(band_ls), band_s2, np.nan)
    band_ls = band_ls[~np.isnan(band_ls)]
    band_s2 = band_s2[~np.isnan(band_s2)]

    flat_s2 = band_s2.flatten()
    flat_ls = band_ls.flatten()
    if flat_s2.size != flat_ls.size:
        print('Error: flat_s2 and flat_ls are not the same size.')
        return
    
    # Downsample the data (if necessary)
    if sample_size > flat_ls.size:
        band_ls_sample = flat_ls
        band_s2_sample = flat_s2
        print(f'Only {flat_ls.size} pixels. Less than downsampling threshold.')     
    else:
        print(f'Downsampling {sample_size} pixels from {flat_ls.size} pixels.')
        sample_idx = np.random.choice(flat_ls.size, sample_size, replace=False)
        band_ls_sample = flat_ls[sample_idx]
        band_s2_sample = flat_s2[sample_idx]

    # Linear regression on downsampled data
    slope, intercept, r_value, p_value, std_err = stats.linregress(band_ls_sample, band_s2_sample)

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(band_ls_sample, band_s2_sample, s=1, alpha=point_alpha, marker='.', label='Data Points')
    min_ls = np.nanmin(band_ls_sample)
    max_ls = np.nanmax(band_ls_sample)
    min_model = slope * min_ls + intercept
    max_model = slope * max_ls + intercept
    plt.plot([min_ls, max_ls], [min_model, max_model], color='red', linestyle='-', label='Best Fit Line')
    
    # Add 1:1 line
    min_val = min(np.nanmin(band_ls_sample), np.nanmin(band_s2_sample))
    max_val = max(np.nanmax(band_ls_sample), np.nanmax(band_s2_sample))
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='1:1 Line')
    
    plt.xlabel('Landsat Reflectance')
    plt.ylabel('Sentinel-2 Reflectance')
    plt.title(f'Scatter plot over {location} of {color} reflectance: Landsat vs Sentinel-2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return slope, intercept, r_value, p_value, std_err


# %% 2.0 Spectral Reflectance Curves on "Land"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_land = pld_src.read(5, window=window_img)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    green_band_ls = green_src.read(1)
    green_band_s2 = green_src.read(2)

    green_band_ls = np.where(pld_mask_land != 1, green_band_ls, np.nan)
    # !!! Rescale the landsat data to 10000x to match the Sentinel-2 data
    green_band_ls = green_band_ls * 10000

    green_band_s2 = np.where(pld_mask_land != 1, green_band_s2, np.nan)
    
    # Histogram of green band reflectance
    ls_green_high_threshold = 1500
    s2_green_high_threshold = 1500
    #threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', 'Land')
    #threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', 'Land')

    # Scatter plot of green band reflectance
    model_land_green = downsample_regress_reflectance(
        green_band_ls,
        green_band_s2,
        ls_green_high_threshold,
        s2_green_high_threshold,
        1_000_000,
        'land',
        'green',
        point_alpha=0.005
    )

    print(model_land_green)

# %% 3.0 Spectral Reflectance Curves on "Water"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_water = pld_src.read(1, window=window_img)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    green_band_ls = green_src.read(1)
    green_band_s2 = green_src.read(2)
    
    green_band_ls = np.where(pld_mask_water == 1, green_band_ls, np.nan)
    green_band_ls = green_band_ls * 10000
    green_band_s2 = np.where(pld_mask_water == 1, green_band_s2, np.nan)

    ls_green_high_threshold = 1500
    s2_green_high_threshold = 1500
    threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', 'Water')
    threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', 'Water')

    model_water_green = downsample_regress_reflectance(
        green_band_ls,
        green_band_s2,
        ls_green_high_threshold,
        s2_green_high_threshold,
        1_000_000,
        'water',
        'green',
        point_alpha=0.1
    )

    print(model_water_green)


# %% 4.0 Spectral Reflectance Curves on "Shoreline / Mixed Pixels (-30m to 60m)"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_inner_shore = pld_src.read(3, window=window_img)
    pld_mask_outer_shore = pld_src.read(4, window=window_img)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    green_band_ls = green_src.read(1)
    green_band_s2 = green_src.read(2)
    
    # Combine the inner and outer shorelines
    shoreline_mask = np.empty_like(pld_mask_inner_shore)
    shoreline_mask = np.where(pld_mask_outer_shore == 1, 1, 0)
    shoreline_mask = np.where(pld_mask_inner_shore == 1, 0, shoreline_mask)
    green_band_ls = np.where(shoreline_mask == 1, green_band_ls, np.nan)
    green_band_ls = green_band_ls * 10000
    green_band_s2 = np.where(shoreline_mask == 1, green_band_s2, np.nan)


    ls_green_high_threshold = 1500
    s2_green_high_threshold = 1500
    threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', 'Shoreline')
    threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', 'Shoreline')

    model_shoreline_green = downsample_regress_reflectance(
        green_band_ls,
        green_band_s2,
        ls_green_high_threshold,
        s2_green_high_threshold,
        1_000_000,
        'shoreline',
        'green',
        point_alpha=0.1
    )

    print(model_shoreline_green)


# %% Model outputs to a table

keys = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
models = [
    {'Model': 'Land Green', **{k: round(v, 4) for k, v in zip(keys, model_land_green)}},
    {'Model': 'Water Green', **{k: round(v, 4) for k, v in zip(keys, model_water_green)}},
    {'Model': 'Shoreline Green', **{k: round(v, 4) for k, v in zip(keys, model_shoreline_green)}}
]

model_df = pd.DataFrame(models)
print(model_df)

# %%
