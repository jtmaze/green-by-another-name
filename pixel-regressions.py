# %% 1.0 Libraries and file paths
import random
import rasterio as rio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd

from scipy import stats
import matplotlib.pyplot as plt

random.seed(40)

pld_mask_path = './data/pld_shapes/pld_masks.tif'
masked_green_path = './data/masked_images/green-masked-conservative.tif'
masked_nir_path = './data/masked_images/nir-masked-conservative.tif'

pld_bands = []
with rio.open(pld_mask_path) as pld_src:
    pld_band_count = pld_src.count
    for band_idx in range(1, pld_band_count + 1):
        description = pld_src.descriptions[band_idx - 1]
        print(f'Band {band_idx} is PLD {description}')
        info = {'band': band_idx, 'description': description}
        pld_bands.append(info)

pld_bands = pd.DataFrame(pld_bands)

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

def histogram_ndwi_pixels(ndwi_data, high_threshold, low_threshold, satellite, location, water_threshold=0.1):

    # Flatten the pixel data
    pixel_data_flat = ndwi_data.flatten()
    
    # Low and high thresholds
    total_values = np.sum(~np.isnan(pixel_data_flat))
    print(f'Total values: {total_values}')
    low_values = np.sum(pixel_data_flat < low_threshold)
    print(f'Total low values: {low_values}')
    high_values = np.sum(pixel_data_flat > high_threshold)
    print(f'Total high values: {high_values}')
    
    percentage_low_removed = (low_values / total_values) * 100
    percentage_high_removed = (high_values / total_values) * 100
    percentage_total_removed = ((low_values + high_values) / total_values) * 100
    percentage_water = np.sum(pixel_data_flat > water_threshold) / total_values * 100
    print(f'Removed {percentage_high_removed:.2f}% of high pixels from {satellite}')
    print(f'Removed {percentage_low_removed:.2f}% of low pixels from {satellite}')
    print(f'Removed {percentage_total_removed:.2f}% of total pixels from {satellite}')
    print(f'Percentage of water pixels: {percentage_water:.2f}% from {satellite} at DNWI threshold {water_threshold}')
    
    # Remove values above high_threshold and below low_threshold
    pixel_data_flat = np.where(pixel_data_flat < high_threshold, pixel_data_flat, np.nan)
    pixel_data_flat = np.where(pixel_data_flat > low_threshold, pixel_data_flat, np.nan)
    # Remove NaN values
    pixel_data_flat = pixel_data_flat[~np.isnan(pixel_data_flat)]
    
    # Plot histogram
    plt.hist(pixel_data_flat, bins=100, alpha=0.5)
    plt.axvline(x=water_threshold, color='red')
    plt.xlim(low_threshold, high_threshold)
    plt.title(f'{satellite} NDWI {location}')
    plt.xlabel('NDWI')
    plt.ylabel('Frequency')
    plt.show()

def downsample_regress_reflectance(ls_data, s2_data, ls_high_threshold, s2_high_threshold, sample_size, location, color, point_alpha=0.01):
    # Apply threshold to high values and zero values, replace with NaN
    band_ls = np.where(ls_data < ls_high_threshold, ls_data, np.nan)
    band_ls = np.where(band_ls > 0, band_ls, np.nan)
    band_s2 = np.where(s2_data < s2_high_threshold, s2_data, np.nan)
    band_s2 = np.where(band_s2 > 0, band_s2, np.nan)

    # Make NaN values common to both datasets
    valid_ls_mask = ~np.isnan(band_ls)
    band_s2 = np.where(valid_ls_mask, band_s2, np.nan)
    valid_s2_mask = ~np.isnan(band_s2)
    band_ls = np.where(valid_s2_mask, band_ls, np.nan)

    flat_s2 = band_s2.flatten()
    flat_ls = band_ls.flatten()
    # The flattened arrays should have the same size
    if flat_s2.size != flat_ls.size:
        print('Error: flat_s2 and flat_ls are not the same size.')
        return

    # Remove the NaN values before downsampling
    band_ls_sample = flat_ls[~np.isnan(flat_ls)]
    band_s2_sample = flat_s2[~np.isnan(flat_s2)]

    # Downsample the data (if necessary)
    if sample_size > band_ls_sample.size:
        print(f'Only {band_ls_sample.size} pixels. Less than the downsampling threshold.')
    else:
        print(f'Downsampling {sample_size} pixels from {band_ls_sample.size} pixels.')
        sample_idx = np.random.choice(band_ls_sample.size, sample_size, replace=False)
        band_ls_sample = band_ls_sample[sample_idx]
        band_s2_sample = band_s2_sample[sample_idx]

    # OLS regression on downsampled data
    slope, intercept, r_value, p_value, std_err = stats.linregress(band_ls_sample, band_s2_sample)
    r_squared = r_value ** 2

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(band_ls_sample, band_s2_sample, s=1, alpha=point_alpha, marker='.')
    min_ls = np.nanmin(band_ls_sample)
    max_ls = np.nanmax(band_ls_sample)
    min_model = slope * min_ls + intercept
    max_model = slope * max_ls + intercept
    plt.plot([min_ls, max_ls], [min_model, max_model], color='red', linestyle='-', label='Best Fit Line')

    # Add 1:1 line
    min_val = min(np.nanmin(band_ls_sample), np.nanmin(band_s2_sample))
    max_val = max(np.nanmax(band_ls_sample), np.nanmax(band_s2_sample))
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='1:1 Line')

    # Add text box with slope and R-squared
    textstr = f'$R^2 = {r_squared:.4f}$\nSlope = {slope:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.xlabel('Landsat Reflectance')
    plt.ylabel('Sentinel-2 Reflectance')
    plt.title(f'{location} of {color} reflectance')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return slope, intercept, r_value, p_value, std_err

def downsample_regress_ndwi(
        ls_data, s2_data, 
        ls_high_threshold, s2_high_threshold, 
        ls_low_threshold, s2_low_threshold,
        sample_size, location, point_alpha=0.01
):
    # Apply threshold to high values and low values, replace with NaN
    band_ls = np.where(ls_data < ls_high_threshold, ls_data, np.nan)
    band_ls = np.where(band_ls > ls_low_threshold, band_ls, np.nan)
    band_s2 = np.where(s2_data < s2_high_threshold, s2_data, np.nan)
    band_s2 = np.where(band_s2 > s2_low_threshold, band_s2, np.nan)

    # Make NaN values common to both datasets
    valid_ls_mask = ~np.isnan(band_ls)
    band_s2 = np.where(valid_ls_mask, band_s2, np.nan)
    valid_s2_mask = ~np.isnan(band_s2)
    band_ls = np.where(valid_s2_mask, band_ls, np.nan)

    flat_s2 = band_s2.flatten()
    flat_ls = band_ls.flatten()
    # The flattened arrays should have the same size
    if flat_s2.size != flat_ls.size:
        print('Error: flat_s2 and flat_ls are not the same size.')
        return

    # Remove the NaN values before downsampling
    band_ls_sample = flat_ls[~np.isnan(flat_ls)]
    band_s2_sample = flat_s2[~np.isnan(flat_s2)]

    # Downsample the data (if necessary)
    if sample_size > band_ls_sample.size:
        print(f'Only {band_ls_sample.size} pixels. Less than the downsampling threshold.')
    else:
        print(f'Downsampling {sample_size} pixels from {band_ls_sample.size} pixels.')
        sample_idx = np.random.choice(band_ls_sample.size, sample_size, replace=False)
        band_ls_sample = band_ls_sample[sample_idx]
        band_s2_sample = band_s2_sample[sample_idx]

    # OLS regression on downsampled data
    slope, intercept, r_value, p_value, std_err = stats.linregress(band_ls_sample, band_s2_sample)
    r_squared = r_value ** 2

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(band_ls_sample, band_s2_sample, s=1, alpha=point_alpha, marker='.')
    min_ls = np.nanmin(band_ls_sample)
    max_ls = np.nanmax(band_ls_sample)
    min_model = slope * min_ls + intercept
    max_model = slope * max_ls + intercept
    plt.plot([min_ls, max_ls], [min_model, max_model], color='red', linestyle='-', label='Best Fit Line')

    # Add 1:1 line
    min_val = min(np.nanmin(band_ls_sample), np.nanmin(band_s2_sample))
    max_val = max(np.nanmax(band_ls_sample), np.nanmax(band_s2_sample))
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='1:1 Line')

    # Add text box with slope and R-squared
    textstr = f'$R^2 = {r_squared:.4f}$\nSlope = {slope:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.xlabel('Landsat NDWI')
    plt.ylabel('Sentinel-2 NDWI')
    plt.title(f'{location} of NDWI')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return slope, intercept, r_value, p_value, std_err


# %% 2.0 Spectral Reflectance Curves on "Land"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    observation_area = 'Outside of PLD buffered 120m (Land)'
    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_land = pld_src.read(6, window=window_img)
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
    ls_green_high_threshold = 1000
    s2_green_high_threshold = 1000
    threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', observation_area)
    threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', observation_area)

    # Scatter plot of green band reflectance
    model_land_green = downsample_regress_reflectance(
        green_band_ls,
        green_band_s2,
        ls_green_high_threshold,
        s2_green_high_threshold,
        1_000_000,
        observation_area,
        'green',
        point_alpha=0.1
    )

    print(model_land_green)

# %% 3.0 Spectral Reflectance Curves on "Water"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    observation_area = 'Inside of PLD -60m buffered (Conservative Lake Mask)'
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

    ls_green_high_threshold = 350
    s2_green_high_threshold = 350
    threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', observation_area)
    threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', observation_area)

    model_water_green = downsample_regress_reflectance(
        green_band_ls,
        green_band_s2,
        ls_green_high_threshold,
        s2_green_high_threshold,
        1_000_000,
        observation_area,
        'green',
        point_alpha=0.1
    )

    print(model_water_green)


# %% 4.0 Spectral Reflectance Curves on "Shoreline / Mixed Pixels (-30m to 30m)"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    observation_area = 'Inside of PLD -30m to 30m buffered (Shoreline / Mixed Pixels)'
    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    # Read two PLD masks to make inner and outer shorelines
    pld_mask_inner_shore = pld_src.read(2, window=window_img)
    pld_mask_outer_shore = pld_src.read(4, window=window_img)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    band_ls = green_src.read(1)
    band_s2 = green_src.read(2)
    
    # Combine the inner and outer shorelines
    shoreline_mask = np.empty_like(pld_mask_inner_shore)
    shoreline_mask = np.where(pld_mask_outer_shore == 1, 1, 0)
    shoreline_mask = np.where(pld_mask_inner_shore == 1, 0, shoreline_mask)
    band_ls = np.where(shoreline_mask == 1, band_ls, np.nan)
    band_ls = band_ls * 10000
    band_s2 = np.where(shoreline_mask == 1, band_s2, np.nan)


    ls_high_threshold = 1000
    s2_high_threshold = 1000
    threshold_histogram_pixels(band_ls, ls_high_threshold, 'green', 'Landsat', observation_area)
    threshold_histogram_pixels(band_s2, s2_high_threshold, 'green', 'Sentinel-2', observation_area)

    model_green_shoreline = downsample_regress_reflectance(
        band_ls,
        band_s2,
        ls_high_threshold,
        s2_high_threshold,
        1_000_000,
        observation_area,
        'green',
        point_alpha=0.1
    )

    print(model_green_shoreline)

# %% 5.0 NIR Reflectance Curves on "Land" outside of PLD buffered 120m

with rio.open(pld_mask_path) as pld_src, rio.open(masked_nir_path) as nir_src:

    observation_area = 'Outside of PLD buffered 120m (Land)'
    img_bounds = nir_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_land = pld_src.read(6, window=window_img)
    pld_meta = pld_src.meta

    nir_band_count = nir_src.count
    for band_idx in range(1, nir_band_count + 1):
        description = nir_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    band_ls = nir_src.read(1)
    band_s2 = nir_src.read(2)

    band_ls = np.where(pld_mask_land != 1, band_ls, np.nan)
    # !!! Rescale the landsat data to 10000x to match the Sentinel-2 data
    band_ls = band_ls * 10000

    band_s2 = np.where(pld_mask_land != 1, band_s2, np.nan)
    
    # Histogram of band reflectance
    ls_high_threshold = 3500
    s2_high_threshold = 3500
    threshold_histogram_pixels(band_ls, ls_high_threshold, 'NIR', 'Landsat', observation_area)
    threshold_histogram_pixels(band_s2, s2_high_threshold, 'NIR', 'Sentinel-2', observation_area)

    # Scatter plot of band reflectance
    nir_model_land = downsample_regress_reflectance(
        band_ls,
        band_s2,
        ls_high_threshold,
        s2_high_threshold,
        1_000_000,
        observation_area,
        'NIR',
        point_alpha=0.1
    )

    print(nir_model_land)

# %% 6.0 NIR Reflectance Curves on "Water"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_nir_path) as nir_src:

    observation_area = 'Inside of PLD buffered -60m (Water)'
    img_bounds = nir_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_water = pld_src.read(1, window=window_img)
    pld_meta = pld_src.meta

    nir_band_count = nir_src.count
    for band_idx in range(1, nir_band_count + 1):
        description = nir_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    band_ls = nir_src.read(1)
    band_s2 = nir_src.read(2)

    band_ls = np.where(pld_mask_water == 1, band_ls, np.nan)
    # !!! Rescale the landsat data to 10000x to match the Sentinel-2 data
    band_ls = band_ls * 10000

    band_s2 = np.where(pld_mask_water == 1, band_s2, np.nan)
    
    # Histogram of band reflectance
    ls_high_threshold = 400
    s2_high_threshold = 400
    threshold_histogram_pixels(band_ls, ls_high_threshold, 'NIR', 'Landsat', observation_area)
    threshold_histogram_pixels(band_s2, s2_high_threshold, 'NIR', 'Sentinel-2', observation_area)

    # Scatter plot of band reflectance
    nir_model_water = downsample_regress_reflectance(
        band_ls,
        band_s2,
        ls_high_threshold,
        s2_high_threshold,
        1_000_000,
        observation_area,
        'NIR',
        point_alpha=0.1
    )

    print(nir_model_water)

# %% 7.0 NIR Reflectance Curves on "Shoreline / Mixed Pixels (-30m to 30m)"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_nir_path) as nir_src:

    observation_area = 'Inside of PLD -30m to 30m buffered (Shoreline / Mixed Pixels)'
    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    # Read two PLD masks to make inner and outer shorelines
    pld_mask_inner_shore = pld_src.read(2, window=window_img)
    pld_mask_outer_shore = pld_src.read(4, window=window_img)
    pld_meta = pld_src.meta

    nir_band_count = nir_src.count
    for band_idx in range(1, nir_band_count + 1):
        description = nir_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    band_ls = nir_src.read(1)
    band_s2 = nir_src.read(2)
    
    # Combine the inner and outer shorelines
    shoreline_mask = np.empty_like(pld_mask_inner_shore)
    shoreline_mask = np.where(pld_mask_outer_shore == 1, 1, 0)
    shoreline_mask = np.where(pld_mask_inner_shore == 1, 0, shoreline_mask)
    band_ls = np.where(shoreline_mask == 1, band_ls, np.nan)
    band_ls = band_ls * 10000
    band_s2 = np.where(shoreline_mask == 1, band_s2, np.nan)


    ls_high_threshold = 3500
    s2_high_threshold = 3500
    threshold_histogram_pixels(band_ls, ls_high_threshold, 'NIR', 'Landsat', observation_area)
    threshold_histogram_pixels(band_s2, s2_high_threshold, 'NIR', 'Sentinel-2', observation_area)

    model_shoreline = downsample_regress_reflectance(
        band_ls,
        band_s2,
        ls_high_threshold,
        s2_high_threshold,
        1_000_000,
        observation_area,
        'NIR',
        point_alpha=0.1
    )

    print(model_shoreline)

# %% Compare shoreline NDWI across satellites

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src, rio.open(masked_nir_path) as nir_src:

    observation_area = 'Inside of PLD -30m to 30m buffered (Shoreline / Mixed Pixels)'
    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    # Read two PLD masks to make inner and outer shorelines
    pld_mask_inner_shore = pld_src.read(2, window=window_img)
    pld_mask_outer_shore = pld_src.read(4, window=window_img)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    nir_band_count = nir_src.count
    for band_idx in range(1, nir_band_count + 1):
        description = nir_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    green_band_ls = green_src.read(1)
    green_band_s2 = green_src.read(2)
    nir_band_ls = nir_src.read(1)
    nir_band_s2 = nir_src.read(2)
    
    # Combine the inner and outer shorelines
    shoreline_mask = np.empty_like(pld_mask_inner_shore)
    shoreline_mask = np.where(pld_mask_outer_shore == 1, 1, 0)
    shoreline_mask = np.where(pld_mask_inner_shore == 1, 0, shoreline_mask)

    green_band_ls = np.where(shoreline_mask == 1, green_band_ls, np.nan)
    green_band_ls = green_band_ls * 10000
    nir_band_ls = np.where(shoreline_mask == 1,nir_band_ls, np.nan)
    nir_band_ls = nir_band_ls * 10000

    green_band_s2 = np.where(shoreline_mask == 1, green_band_s2, np.nan)
    nir_band_s2 = np.where(shoreline_mask == 1, nir_band_s2, np.nan)

    ndwi_ls = (green_band_ls - nir_band_ls) / (green_band_ls + nir_band_ls)
    ndwi_s2 = (green_band_s2 - nir_band_s2) / (green_band_s2 + nir_band_s2)

    ls_high_threshold = 1
    s2_high_threshold = 1
    ls_low_threshold = -1
    s2_low_threshold = -1

    histogram_ndwi_pixels(ndwi_ls, ls_high_threshold, ls_low_threshold, 'Landsat', observation_area, 0.1)
    histogram_ndwi_pixels(ndwi_s2, s2_high_threshold, s2_low_threshold, 'Sentinel-2', observation_area, 0.1)

    model_shoreline = downsample_regress_ndwi(
        ndwi_ls,
        ndwi_s2,
        ls_high_threshold,
        s2_high_threshold,
        ls_low_threshold,
        s2_low_threshold,
        1_000_000,
        observation_area,
        point_alpha=0.1
    )

    print(model_shoreline)


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
