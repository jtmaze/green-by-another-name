# %%

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import exposure
import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import from_bounds

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

# %%

roi_name = "AND_sub2"
date = "2021-06-16"
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
MASK_ALPHA = 1
MASK_COLOR = "pink"
ndwi_colors = ['darkgreen', 'white', 'darkblue']
NDWI_CMAP = LinearSegmentedColormap.from_list("custom_ndwi", ndwi_colors, N=100)
PLD_EDGE_COLOR = 'red'
PLD_BUFFER_COLOR = 'red'
PLD_INNER_COLOR = 'red'

# %% Fiddle with the bounds

with rio.open(rgb_paths[0]) as src:
    full_bounds = src.bounds
    # Create a small window in the center
    center_x = (full_bounds.left + full_bounds.right) / 2
    center_y = (full_bounds.bottom + full_bounds.top) / 2
    print(center_x, center_y)

    chosen_x = center_x - 1_300
    chosen_y = center_y + 1_200

spread = 400  # 1km in each direction
utm_bounds = (chosen_x - spread, chosen_y - spread, chosen_x + spread, chosen_y + spread)
print("Testing with bounds:", utm_bounds)

"""
For AND_sub2 on 2021-06-16
Panel A
spread = 1250
x = -670
y = +400
Panel B
spread = 800
"""

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
    
def scale_rgb(arr: np.ndarray, upper: float, lower: float):
    """Scales the RGB values for display"""

    stretched = arr.astype(np.float32).copy()
    for i in range(stretched.shape[0]):
        p_low, p_high = np.nanpercentile(stretched[i], (lower, upper))
        stretched[i] = exposure.rescale_intensity(
            stretched[i], in_range=(p_low, p_high), out_range=(0.0, 1.0)
        )
    return np.clip(stretched, 0, 1)   

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

    rgb_scaled = scale_rgb(rgb, 98, 2)
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
    wtr_mask, _, _ = read_window(wtr_mask_path, utm_bounds, bands=[1])
    wtr_bool = np.squeeze(wtr_mask) > 0
    # Build an X/Y grid that matches the raster window
    ny, nx = wtr_bool.shape
    x = np.linspace(left,  right, nx)          # east-west
    y = np.linspace(top,   bottom, ny)         # north-south (flip)
                                              # NB: top > bottom in UTM
    # Draw the outline
    ax.contour(x, y, wtr_bool, levels=[0.5],
               colors=MASK_COLOR, linewidths=4,
               alpha=MASK_ALPHA, origin='image')


def plot_ndwi_panels(ax, ndwi_path: str, pld_path: str):
    """Plots the NDWI image with PLD lakes as an overlay."""

    # Read and set NDWI on the plot
    ndwi, trans, _ = read_window(ndwi_path, utm_bounds, bands=[1])
    ndwi = ndwi.squeeze()
    # Need to clip for unbounded SR NDWI values
    ndwi = np.clip(ndwi, -1, 1)
    # Calculate the proper extent
    height, width = ndwi.shape
    left, bottom, right, top = rio.transform.array_bounds(height, width, trans)
    extent = [left, right, bottom, top]  # [west, east, south, north]
    
    # Display with correct extent
    im = ax.imshow(ndwi, extent=extent, cmap=NDWI_CMAP, vmin=-1, vmax=1)
    
    # Force aspect ratio to be equal
    ax.set_aspect('equal')

    lake,   _, _ = read_window(pld_path, utm_bounds, bands=[4])
    buffer, _, _ = read_window(pld_path, utm_bounds, bands=[6])
    inner,  _, _ = read_window(pld_path, utm_bounds, bands=[2])

    lake_bool   = lake.squeeze()   > 0
    buffer_bool = buffer.squeeze() > 0
    inner_bool  = inner.squeeze()  > 0

    ny, nx = lake_bool.shape
    x = np.linspace(left,  right, nx)
    y = np.linspace(top,   bottom, ny)

    ax.contour(x, y, lake_bool,   levels=[0.5], colors=PLD_EDGE_COLOR,
               linewidths=2, alpha=MASK_ALPHA, origin='image')
    # ax.contour(x, y, buffer_bool, levels=[0.5], colors=PLD_BUFFER_COLOR,
    #            linewidths=2, alpha=MASK_ALPHA, origin='image')
    # ax.contour(x, y, inner_bool,  levels=[0.5], colors=PLD_INNER_COLOR,
    #            linewidths=2, alpha=MASK_ALPHA, origin='image')

    ax.set_axis_off()

    return im

def plot_rgb_panels_with_pld(ax, rgb_path: str, pld_path: str):
    """Plots each satellite's RGB image with the PLD lakes as an overlay."""

    # Read and set RGB on
    rgb, trans, _ = read_window(rgb_path, utm_bounds, bands=[1, 2, 3])
    data_min = np.nanmin(rgb)
    data_max = np.nanmax(rgb)
    print(data_min)
    print(data_max)

    rgb_scaled = scale_rgb(rgb, 98, 2)
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

    lake,   _, _ = read_window(pld_path, utm_bounds, bands=[4])
    buffer, _, _ = read_window(pld_path, utm_bounds, bands=[6])
    inner,  _, _ = read_window(pld_path, utm_bounds, bands=[2])

    lake_bool   = lake.squeeze()   > 0
    buffer_bool = buffer.squeeze() > 0
    inner_bool  = inner.squeeze()  > 0

    ny, nx = lake_bool.shape
    x = np.linspace(left,  right, nx)
    y = np.linspace(top,   bottom, ny)

    ax.contour(x, y, lake_bool,   levels=[0.5], colors=PLD_EDGE_COLOR,
               linewidths=2.5, alpha=MASK_ALPHA, origin='image')
    ax.contour(x, y, buffer_bool, levels=[0.5], colors=PLD_BUFFER_COLOR,
               linewidths=2.5, alpha=MASK_ALPHA, origin='image')
    ax.contour(x, y, inner_bool,  levels=[0.5], colors=PLD_INNER_COLOR,
               linewidths=2.5, alpha=MASK_ALPHA, origin='image')

    ax.set_axis_off()


# %% Make the plots

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=False)  

# Plot RGB images
for i, (label, rgb_path, wtr_mask_path) in enumerate(zip(plot_labels, rgb_paths, wtr_mask_paths)):
    ax = axes[0, i]
    plot_rgb_panels(ax, rgb_path, wtr_mask_path)
    ax.set_title(f"{label} RGB", fontsize=12)

ndwi_images = []
for i, (label, ndwi_path, pld_path) in enumerate(zip(plot_labels, ndwi_paths, [pld_path]*4)): # PLD path is the same for all
    ax = axes[1, i]
    im = plot_ndwi_panels(ax, ndwi_path, pld_path)
    ndwi_images.append(im)
    ax.set_title(f"{label} NDWI", fontsize=12)

cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # [x, y, width, height]
cbar = fig.colorbar(ndwi_images[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label('NDWI Values', labelpad=15)

fig.suptitle(f"Image comparison for {roi_name} on {date} resampled {resample_method}", 
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

    
# %% Make one map with unresampled Sentinel-2 image and PLD zones

roi_name = "AKCP_sub1"
date = '2020-07-03'
PLD_EDGE_COLOR = 'blue'
PLD_BUFFER_COLOR = 'orange'
PLD_INNER_COLOR = 'red'

img_path = f'./data/sr_images/roi_{roi_name}_noresample/Landsat8_sr_date_{date}_roi_{roi_name}.tif'
pld_path = f'./data/pld_rasterized/{roi_name}_lake_masks_res{30}.tif'

with rio.open(img_path) as src:
    full_bounds = src.bounds
    # Create a small window in the center
    center_x = (full_bounds.left + full_bounds.right) / 2
    center_y = (full_bounds.bottom + full_bounds.top) / 2
    print(center_x, center_y)

    chosen_x = center_x + 5500
    chosen_y = center_y - 3000

spread = 1_200  # 1km in each direction
utm_bounds = (chosen_x - spread, chosen_y - spread, chosen_x + spread, chosen_y + spread)

fig, ax = plt.subplots(figsize=(8, 8))
plot_rgb_panels_with_pld(ax, img_path, pld_path)   

from matplotlib_scalebar.scalebar import ScaleBar

# 1.0 means 1 pixel = 1 meter, which is true for UTM coordinates
scalebar = ScaleBar(1.0, 
                    units='m',
                    length_fraction=0.25,
                    location='upper left',
                    box_alpha=1,
                    color='black',
                    frameon=True,
                    box_color='white',
                    pad=2,
                    #box_kwargs={'edgecolor': 'black', 'linewidth': 2}
)
ax.add_artist(scalebar)

plt.tight_layout()
plt.show()


# %%
