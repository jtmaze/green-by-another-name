# %%

import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import from_bounds

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

# %%

roi_name = "YKF_sub4"
date = "2021-06-06"
res = 30
resample_method = f"bilinear{res}"

plot_labels = [
    "Landsat8 SR",
    "Sentinel-2 SR",
    "Landsat8 TOA",
    "Sentinel-2 TOA",
]

# Paths to the data
rgb_paths = [
    f'./data/sr_images/roi_{roi_name}_resampled_{resample_method}/reprojected_{resample_method}_Landsat8_sr_date_{date}_roi_{roi_name}.tif',
    f'./data/sr_images/roi_{roi_name}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_sr_date_{date}_roi_{roi_name}.tif',
    f'./data/toa_images/roi_{roi_name}_resampled_{resample_method}/reprojected_{resample_method}_Landsat8_toa_date_{date}_roi_{roi_name}.tif',
    f'./data/toa_images/roi_{roi_name}_resampled_{resample_method}/reprojected_{resample_method}_Sentinel2_toa_date_{date}_roi_{roi_name}.tif',
]

wtr_mask_paths = [
    f'./data/processed_water_masks/LS8_water_mask_sr_{roi_name}_{date}_{resample_method}.tif',
    f'./data/processed_water_masks/S2_water_mask_sr_{roi_name}_{date}_{resample_method}.tif',
    f'./data/processed_water_masks/LS8_water_mask_toa_{roi_name}_{date}_{resample_method}.tif',
    f'./data/processed_water_masks/S2_water_mask_toa_{roi_name}_{date}_{resample_method}.tif',
]

ndwi_paths = [
    f'./data/ndwi_rasters/LS8_ndwi_sr_{roi_name}_{date}_{resample_method}.tif',
    f'./data/ndwi_rasters/S2_ndwi_sr_{roi_name}_{date}_{resample_method}.tif',
    f'./data/ndwi_rasters/LS8_ndwi_toa_{roi_name}_{date}_{resample_method}.tif',
    f'./data/ndwi_rasters/S2_ndwi_toa_{roi_name}_{date}_{resample_method}.tif',
]

pld_path = f'./data/pld_rasterized/{roi_name}_lake_masks_res{res}.tif'

# Stylistic parameters
rgb_scaling = (0, 0.0001) # (min, max)
MASK_ALPHA = 0.5
MASK_COLOR = "pink"
NDWI_CMAP = "Blues"
PLD_EDGE_COLOR = 'red'
PLD_BUFFER_COLOR = 'blue'
PLD_INNER_COLOR = 'green'

# %% Fiddle with the bounds

with rio.open(rgb_paths[0]) as src:
    full_bounds = src.bounds
    # Create a small window in the center
    center_x = (full_bounds.left + full_bounds.right) / 2
    center_y = (full_bounds.bottom + full_bounds.top) / 2
    print(center_x, center_y)

    chosen_x = center_x + 8_500
    chosen_y = center_y - 7_500

spread = 2_000  # 1km in each direction
utm_bounds = (chosen_x - spread, chosen_y - spread, chosen_x + spread, chosen_y + spread)
print("Testing with bounds:", utm_bounds)

# %% functions

def read_window(path: str, bounds: tuple, bands: list):
    """Read a window from a raster file."""
    with rio.open(path) as src:
        print(src.crs)
        window = from_bounds(*bounds, transform=src.transform)
        if bands is None:
            data = src.read(window=window)
        else:
            data = src.read(bands, window=window)
        trans = src.window_transform(window)
        # Ensure nodata is replaced with NaN
        if src.nodata is not None:
            data = data.astype("float32", copy=False)
            data[data == src.nodata] = np.nan

        return data, trans, src.crs
    
def scale_rgb(arr: np.ndarray, min_val: float, max_val: float):
    """Scales the RGB values for display"""
    arr = arr.astype(np.float32)
    arr = (arr - min_val) / (max_val - min_val)
    arr = np.clip(arr, 0, 1)
    return arr    

def plot_rgb_panels(ax, rgb_path: str, wtr_mask_path: str):
    """Plots each satellite's RGB image with the water mask."""

    # Read and set RGB on the plot
    rgb, trans, _ = read_window(rgb_path, utm_bounds, bands=[1, 2, 3])
    print(f"RGB array shape: {rgb.shape}, contains NaN: {np.isnan(rgb).any()}")

    # Rescale the data 0-1
    data_min = np.nanmin(rgb)
    data_max = np.nanmax(rgb)
    print(data_min)
    print(data_max)

    rgb_scaled = scale_rgb(rgb, data_min, data_max)
    print(rgb_scaled.shape)
    # Transpose from (bands,height,width) to (height,width,bands)
    rgb_display = np.transpose(rgb_scaled, (1, 2, 0))
    
    # Calculate the proper extent
    height, width = rgb_display.shape[0], rgb_display.shape[1]
    left, bottom, right, top = rio.transform.array_bounds(height, width, trans)
    extent = [left, right, bottom, top]  # [west, east, south, north]
    
    # Display the image with correct extent
    img = ax.imshow(rgb_display, extent=extent)
    
    # Force aspect ratio to be equal
    ax.set_aspect('equal')
    
    ax.set_axis_off()

    # Read the water mask set it on the plot
    # wtr_mask, _, _ = read_window(wtr_mask_path, utm_bounds, bands=[1])
    # wtr_bool = np.squeeze(wtr_mask) > 0
    # show(
    #     wtr_bool.astype("uint8"),
    #     transform = trans,
    #     ax=ax,
    #     cmap=None,
    #     alpha=0,
    # )
    # ax.contour(wtr_bool, levels=[0.5], colors=MASK_COLOR, linewidths=0.5, alpha=MASK_ALPHA)
    ax.set_axis_off()

def plot_ndwi_panels(ax, ndwi_path: str, pld_path: str):
    """Plots the NDWI image with PLD lakes as an overlay."""

    # Read and set NDWI on the plot
    ndwi, trans, _ = read_window(ndwi_path, utm_bounds, bands=[1])
    ndwi = ndwi.squeeze()
    # Calculate the proper extent
    height, width = ndwi.shape
    left, bottom, right, top = rio.transform.array_bounds(height, width, trans)
    extent = [left, right, bottom, top]  # [west, east, south, north]
    
    # Display with correct extent
    ax.imshow(ndwi, extent=extent, cmap=NDWI_CMAP)
    
    # Force aspect ratio to be equal
    ax.set_aspect('equal')

    # lake, _, _ = read_window(pld_path, utm_bounds, bands=[4]) # lake with 0m buffer
    # lake_bool = lake.squeeze() > 0
    # print(lake_bool.shape)
    # buffer, _,  _ = read_window(pld_path, utm_bounds, bands=[6]) # lake with 60m buffer
    # buffer_bool = buffer.squeeze() > 0
    # inner, _, _ = read_window(pld_path, utm_bounds, bands=[2]) # lake with -60m buffer
    # inner_bool = inner.squeeze() > 0

    # print(f"Lake mask contains True values: {lake_bool.any()}")
    # print(f"Buffer mask contains True values: {buffer_bool.any()}")
    # print(f"Inner mask contains True values: {inner_bool.any()}")

    # ax.contour(lake_bool, levels=[0.5], colors=PLD_EDGE_COLOR, linewidths=1, alpha=MASK_ALPHA)
    # ax.contour(buffer_bool, levels=[0.5], colors=PLD_BUFFER_COLOR, linewidths=1, alpha=MASK_ALPHA)
    # ax.contour(inner_bool, levels=[0.5], colors=PLD_INNER_COLOR, linewidths=1, alpha=MASK_ALPHA)

    ax.set_axis_off()


# %% Make the plots

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)  

# Plot RGB images
for i, (label, rgb_path, wtr_mask_path) in enumerate(zip(plot_labels, rgb_paths, wtr_mask_paths)):
    ax = axes[0, i]
    plot_rgb_panels(ax, rgb_path, wtr_mask_path)
    ax.set_title(f"{label} RGB", fontsize=12)

for i, (label, ndwi_path, pld_path) in enumerate(zip(plot_labels, ndwi_paths, [pld_path]*4)): # PLD path is the same for all
    ax = axes[1, i]
    plot_ndwi_panels(ax, ndwi_path, pld_path)
    ax.set_title(f"{label} NDWI", fontsize=12)

fig.suptitle(f"Image comparison for {roi_name} on {date} resampled {resample_method}", fontsize=16, fontweight='bold')
plt.show()

    


# %%
