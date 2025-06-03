# %% 1.0 Libraries and directories

import os
import glob
import geopandas as gpd
import pandas as pd
import rasterio 
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from affine import Affine # Might need to install this package if not already available
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

footprint_dir = './data/overlap_dates_for_roi'
footprints_files = glob.glob(f'{footprint_dir}/**_overlap_dates.shp')
main_rois_dir = './data/roi_shapes/ew_rois/'
main_rois_files = glob.glob(f'{main_rois_dir}/**_shape.shp')
# %% 2.0 Read the total overlap footprints, valid roi dates, and main roi shapes

# Image overlap shapes
overlap_list = []
for f in footprints_files:
    roi = f.split('/')[-1].replace('_overlap_dates.shp', '')
    gdf = gpd.read_file(f)
    gdf['roi'] = roi
    overlap_list.append(gdf)
overlap_footprints = pd.concat(overlap_list)

# Valid dates for images with enough pixels and low cloud cover
valid_roi_dates = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch3.csv')
valid_roi_dates = valid_roi_dates[['roi', 'date']].copy()

# Main rois
main_rois_list = []
for f in main_rois_files:
    gdf = gpd.read_file(f)
    main_rois_list.append(gdf)
main_rois = pd.concat(main_rois_list)
main_rois = gpd.GeoDataFrame(
    main_rois,
    geometry='geometry',
    crs='EPSG:4326'
)

footprints = pd.merge(valid_roi_dates, overlap_footprints, on=['date', 'roi'], how='left')
# Group by ROI and get the count of dates plus the first geometry
footprints_grouped = footprints.groupby('roi').agg({
    'date': 'count',  # Count dates per ROI
    'geometry': 'first'  # Get the first geometry for each ROI
}).reset_index()

# Rename the date column to roi_img_count
footprints_grouped = footprints_grouped.rename(columns={'date': 'roi_img_count'})

# Use the grouped data
footprints = footprints_grouped

max_img_count = footprints['roi_img_count'].max()
footprints['rel_img_count'] = footprints['roi_img_count'] / max_img_count

footprints = gpd.GeoDataFrame(
    footprints,
    geometry='geometry',
    crs='EPSG:4326'
)

# %% 3.0 Represent the PLD shapefile as a shaded raster for plotting
pld = gpd.read_file('./data/pld_shapes/pld_unclipped/pld_w_arctic_v3.shp')

# 2. Decide the resolution of the output grid (°/pixel)
res = 0.04         
xmin, ymin, xmax, ymax = pld.total_bounds
width  = int(np.ceil((xmax - xmin) / res))
height = int(np.ceil((ymax - ymin) / res))
transform = Affine(res, 0, xmin, 0, -res, ymax)

# 3. Burn each polygon into the grid, *adding* rather than replacing
lake_density = rasterize(
    ((geom, 1) for geom in pld.geometry),
    out_shape=(height, width),
    transform=transform,
    merge_alg=MergeAlg.add,       # <-- count overlaps
    dtype="uint32"
)                                 # :contentReference[oaicite:0]{index=0}

from scipy import ndimage

# Apply Gaussian smoothing to the lake density array
# sigma controls the amount of blur (higher = more blur)
sigma = 2  # Adjust this value to control blur intensity
smoothed_lake_density = ndimage.gaussian_filter(lake_density.astype(float), sigma)

# %%
# 4.0 Create map showing lake density and image counts per ROI

# 4.1 Set up figure and projection
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# 4.2 Draw basemap features
ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)

# 4.3 Create and apply custom colormap for lake density visualization

colors = [
    '#FFFFFF00',  # Transparent white
    '#F7FBFF',    # Very light blue
    '#DEEBF7',    # Light blue
    '#C6DBEF',    # Light-medium blue
    '#9ECAE1',    # Medium blue

    '#6BAED6',    # Medium-dark blue
    '#4292C6',    # Dark blue
    '#2171B5',    # Darker blue
    '#08519C',    # Very dark blue
    '#08519C',    # Very dark blue (consistent above 0.5)
]
positions = [
    0, 0.02, 0.03, 0.08, 0.12, 
    0.17, 0.25, 0.4, 0.5, 1
]

custom_cmap = LinearSegmentedColormap.from_list('custom_blues', list(zip(positions, colors)))
# Display lake density raster
ax.imshow(
    smoothed_lake_density,
    extent=(xmin, xmax, ymin, ymax),
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap,
    zorder=1
)

# 4.4 Add coordinate grid with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 4.5 Prepare styling for ROI boundaries
unique_rois = main_rois['roi'].unique()
custom_colors = [
    'maroon',
    'maroon',
    'maroon',
    'maroon',
    'maroon',
    'maroon',
]
color_map = dict(zip(unique_rois, custom_colors))

# 4.6 Plot ROI boundaries
for roi, col in color_map.items():
    subset = main_rois[main_rois['roi'] == roi]
    subset.boundary.plot(
        ax=ax,
        edgecolor=col,
        linewidth=1,
        label=str(roi)
    )

# 4.7 Plot image footprints with color indicating count
cmap = mpl.colors.LinearSegmentedColormap.from_list('Warm', 
    ['#FFE4B5', '#FF7F50', '#8B0000'])

# Convert footprints to centroids for point representation
footprints_centroids = footprints.copy()
footprints_centroids.geometry = footprints_centroids.geometry.centroid

footprints_centroids.plot(
    ax=ax,
    column='roi_img_count',
    cmap=cmap,
    alpha=1,
    edgecolor='black',
    linewidth=0.5,
    markersize=142,
    legend=False,
    zorder=8
)

# 4.8 Create colorbar with custom styling
# Set up color normalization
norm = mpl.colors.Normalize(
    vmin=footprints['roi_img_count'].min(),
    vmax=footprints['roi_img_count'].max()
)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []  # dummy array for the ScalarMappable

# 4.9 Create background container for colorbar
cax_img_area = inset_axes(ax,
                    width="25%", height="69%", loc='lower right',
                    bbox_to_anchor=(0.30,-0.05, 0.75, 0.75),
                    bbox_transform=ax.transAxes,
                    borderpad=3)


cax_img_area.patch.set_visible(False)
rounded_rect = FancyBboxPatch(
    (0, 0), 1, 1,
    facecolor='whitesmoke',  # Much lighter than lightgrey
    alpha=1,
    edgecolor='black',
    linewidth=5.5,
    transform=cax_img_area.transAxes,
    zorder=10
)

cax_img_area.add_patch(rounded_rect)

# Clean up background container
cax_img_area.set_xticks([])
cax_img_area.set_yticks([])
cax_img_area.set_frame_on(False)
cax_img_area.set_zorder(ax.get_zorder() + 0.5)

# 4.10 Add the actual colorbar
cax = inset_axes(ax,
                 width="9%", height="52%", loc='lower right',
                 bbox_to_anchor=(0.24, -0.01, 0.75, 0.75),
                 bbox_transform=ax.transAxes,
                 borderpad=3)

cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_title('Image Count', fontsize=12, weight='bold', pad=12)
# Ensure ticks are integers only
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_visible(False)
cax.set_zorder(ax.get_zorder() + 1)

"""Adds lake legend"""
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# # Create a new axis for the colorbar below the plot
# cax_lakes = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]

# # Create a ScalarMappable for the lake density
# sm_lakes = plt.cm.ScalarMappable(cmap=custom_cmap, norm=mpl.colors.Normalize(vmin=0, vmax=smoothed_lake_density.max()))
# sm_lakes.set_array([])

# # Create the colorbar
# lake_cbar = fig.colorbar(sm_lakes, cax=cax_lakes, orientation='horizontal')
# lake_cbar.set_label('PLD Lake Density', fontsize=14, weight='bold', labelpad=10)
# lake_cbar.ax.tick_params(labelsize=10)
# lake_cbar.outline.set_visible(False)

# # Adjust the bottom margin to make room for both legends
# plt.subplots_adjust(bottom=0.25)  # Increase this value to make more room
# # Display the figure
plt.show()

# %%

# 1. Set up figure/projection
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# 2. Draw basemap features
ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.imshow(
    smoothed_lake_density,
    extent=(xmin, xmax, ymin, ymax),
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap,
)

roi_mapping = {
    'YKdelta': 'Yukon Delta',
    'AKCP': 'Alaska Coastal Plain',
    'anderson_plain': 'Anderson Plain',
    'YKflats': 'Yukon Flats',
    'TUK': 'Tuktoyaktuk Peninsula',
    'MRD': 'Mackenzie River Delta',
}

main_rois['roi'] = main_rois['roi'].map(roi_mapping)

# 3. Prepare a custom color list and map it to your ROI names
unique_rois = main_rois['roi'].unique()
# Option 1: High contrast with distinct hues
custom_colors = [
    '#D62728',  # Subdued Red
    '#2CA02C',  # Subdued Green  
    '#E69F00',  # Subdued Orange
    '#CC79A7',  # Subdued Magenta/Pink
    '#DBDB8D',  # Subdued Yellow
    '#A461FF',  # Brighter Purple
]

# # Option 2: Dark saturated colors
# custom_colors2 = [
#     '#8B0000',  # Dark Red
#     '#006400',  # Dark Green
#     '#000080',  # Navy Blue
#     '#800080',  # Purple
#     '#FF8C00',  # Dark Orange
#     '#008B8B',  # Dark Cyan
# ]

# # Option 3: Professional color palette
# custom_colors3 = [
#     '#1f77b4',  # Blue
#     '#ff7f0e',  # Orange
#     '#2ca02c',  # Green
#     '#d62728',  # Red
#     '#9467bd',  # Purple
#     '#8c564b',  # Brown
# ]

# # Option 4: Maximum contrast palette
# custom_colors = [
#     '#000000',  # Black
#     '#FF0000',  # Red
#     '#00FF00',  # Lime
#     '#0000FF',  # Blue
#     '#FFFF00',  # Yellow
#     '#FF00FF',  # Magenta
#]
color_map = dict(zip(unique_rois, custom_colors))

# 4. Plot the main ROIs with colored outlines only (no fill)
for roi, col in color_map.items():
    subset = main_rois[main_rois['roi'] == roi]
    subset.boundary.plot(
        ax=ax,
        edgecolor=col,  # Use the color for the outline
        linewidth=3.5,    # Make the outline a bit thicker for visibility
        label=str(roi)  # Add label for legend
    )

# 5. Add a legend for the main ROIs

# Create a font dictionary with bold weight
title_font = fm.FontProperties(weight='bold', size=18)
# Create patches with both fill color and border color
handles = [mpatches.Patch(facecolor=col, alpha=1, edgecolor=col, linewidth=3, label=roi) for roi, col in color_map.items()]
legend = ax.legend(
    handles=handles,
    loc='lower right',
    title='Hydrographic Regions',
    fontsize=12,
    title_fontproperties=title_font,
    facecolor='whitesmoke',
    frameon=True,                # Enable the bounding box
    handlelength=3.0,
    handleheight=1.5,
    labelspacing=1.2,
    borderpad=1.0,
    handletextpad=1.0, 
)
legend.get_frame().set_alpha(1)  
legend.get_frame().set_edgecolor('white')  # Set the border color to white


plt.show()

# %%



# %%
