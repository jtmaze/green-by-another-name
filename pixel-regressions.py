# %% 1.0 Libraries and file paths

import rasterio as rio
from rasterio.windows import from_bounds
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt

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


# %% 2.0 Spectral Reflectance Curves on "Land"

with rio.open(pld_mask_path) as pld_src, rio.open(masked_green_path) as green_src:

    img_bounds = green_src.bounds
    window_img = from_bounds(*img_bounds, pld_src.transform)

    pld_mask_land = pld_src.read(5, window=window_img)
    print(pld_mask_land.shape)
    pld_meta = pld_src.meta

    green_band_count = green_src.count
    for band_idx in range(1, green_band_count + 1):
        description = green_src.descriptions[band_idx - 1]
        print(f'Band #{band_idx} is {description}')

    green_band_ls = green_src.read(1)
    print(green_band_ls.shape)
    green_band_s2 = green_src.read(2)
    print(green_band_s2.shape)

    green_band_ls = np.where(pld_mask_land != 1, green_band_ls, np.nan)
    # !!! Rescale the landsat data to 10000x to match the Sentinel-2 data
    green_band_ls = green_band_ls * 10000

    green_band_s2 = np.where(pld_mask_land != 1, green_band_s2, np.nan)
    
    # Histogram of green band reflectance
    ls_green_high_threshold = 1500
    s2_green_high_threshold = 1500
    #threshold_histogram_pixels(green_band_ls, ls_green_high_threshold, 'green', 'Landsat', 'Land')
    #threshold_histogram_pixels(green_band_s2, s2_green_high_threshold, 'green', 'Sentinel-2', 'Land')

    # Apply threshold to high values and zero values
    green_band_ls = np.where(green_band_ls < ls_green_high_threshold, green_band_ls, np.nan)
    green_band_ls = np.where(green_band_ls > 0, green_band_ls, np.nan)
    green_band_s2 = np.where(green_band_s2 < s2_green_high_threshold, green_band_s2, np.nan)
    green_band_s2 = np.where(green_band_s2 > 0, green_band_s2, np.nan)

    # Make Nan Values common to both datasets
    green_band_ls = np.where(~np.isnan(green_band_s2), green_band_ls, np.nan)
    green_band_s2 = np.where(~np.isnan(green_band_ls), green_band_s2, np.nan)

    # Plot the scatter plot
    flat_s2 = green_band_s2.flatten()
    flat_ls = green_band_ls.flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(flat_ls, flat_s2, alpha=0.001)
    plt.xlabel('Landsat Green Reflectance')
    plt.ylabel('Sentinel-2 Green Reflectance')
    plt.title('Scatter Plot of Green Reflectance: Landsat vs Sentinel-2')
    plt.grid(True)
    plt.show()


# %% 3.0 Spectral Reflectance Curves on "Water"

# %% 4.0 Spectral Reflectance Curves on "Shoreline / Mixed Pixels"