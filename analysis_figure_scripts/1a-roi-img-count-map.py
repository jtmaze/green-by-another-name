# %% 1.0 Libraries and directories

import os
import glob
import geopandas as gpd
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
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

# %% 3.0 Keep only the footprints that were valid observations

footprints = pd.merge(valid_roi_dates, overlap_footprints, on=['date', 'roi'], how='left')
footprints['roi_img_count'] = footprints.groupby('roi')['date'].transform('count')
max_img_count = footprints['roi_img_count'].max()
footprints['rel_img_count'] = footprints['roi_img_count'] / max_img_count

footprints = gpd.GeoDataFrame(
    footprints,
    geometry='geometry',
    crs='EPSG:4326'
)

# %%

# 1. Set up figure/projection
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# 2. Draw basemap features
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 3. Prepare a custom color list and map it to your ROI names
unique_rois = main_rois['roi'].unique()
custom_colors = [
    '#000080',
    '#000080',
    '#000080',
    '#000080',
    '#000080',
    '#000080',
]
color_map = dict(zip(unique_rois, custom_colors))

# 4. Loop through main rois and plot only boundaries
for roi, col in color_map.items():
    subset = main_rois[main_rois['roi'] == roi]
    subset.boundary.plot(
        ax=ax,
        edgecolor=col,
        linewidth=1.5,
        label=str(roi)
    )

# 5. Plot footprints on top (no legend here)
cmap = mpl.colors.LinearSegmentedColormap.from_list('RB', ['red','blue'])
footprints.plot(
    ax=ax,
    column='roi_img_count',
    cmap=cmap,
    alpha=1,
    edgecolor='black',
    linewidth=0.1,
    legend=False
)

# 6. Add an inset colorbar with white background, positioned further inside the map

norm = mpl.colors.Normalize(
    vmin=footprints['roi_img_count'].min(),
    vmax=footprints['roi_img_count'].max()
)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []  # dummy array for the ScalarMappable

# Create an inset axes for the colorbar, positioned more inside the map
cax_bg = inset_axes(ax,
                    width="25%", height="69%", loc='lower right',
                    bbox_to_anchor=(0.23, 0.0, 0.75, 0.75),   # a hair smaller pad
                    bbox_transform=ax.transAxes,
                    borderpad=3)

# Create rounded rectangle with white face + navy border
# Remove default rectangle
cax_bg.patch.set_visible(False)
# Add rounded rectangle
rounded_rect = FancyBboxPatch(
    (0, 0), 1, 1,  # (x, y), width, height
    boxstyle="round,pad=-0.02,rounding_size=0.1",
    facecolor='white',
    edgecolor='navy',
    linewidth=1.5,
    transform=cax_bg.transAxes,
    zorder=0
)
cax_bg.add_patch(rounded_rect)

# we don't want ticks or labels on the background
cax_bg.set_xticks([]);  cax_bg.set_yticks([])
cax_bg.set_frame_on(False)  # Turn off frame since we have our custom rounded rectangle

# make sure it renders *under* everything else
cax_bg.set_zorder(ax.get_zorder() + 0.5)


# --- now the real colour‑bar axes, a touch smaller, on top ---
cax = inset_axes(ax,
                 width="9%", height="52%", loc='lower right',
                 bbox_to_anchor=(0.17, 0.05, 0.75, 0.75),   # centred in the white box
                 bbox_transform=ax.transAxes,
                 borderpad=3)

cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_title('Image Count', fontsize=12, weight='bold', pad=12)  # Add the title
# Set integer ticks
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_visible(False)        # we already have the outer border
cax.set_zorder(ax.get_zorder() + 1)    # make sure it's above the white box

# 7. Finish up
plt.show()
# %%

# 1. Set up figure/projection
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# 2. Draw basemap features
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

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
custom_colors = [
    '#87CEEB',  # Light Blue
    '#FFE4B5',  # Soft Orange
    '#20B2AA',  # Teal
    '#FF6961',  # Tomato
    '#E6E6FA',  # Soft Purple
    '#DAA520',  # Goldenrod
]
color_map = dict(zip(unique_rois, custom_colors))

# 4. Plot the main ROIs with colors and add them to the legend
for roi, col in color_map.items():
    subset = main_rois[main_rois['roi'] == roi]
    subset.plot(
        ax=ax,
        facecolor=col,  # Fill the polygons with the color
        edgecolor='navy',  # Keep the boundary color
        linewidth=1.5,
        label=str(roi)  # Add label for legend
    )

# 5. Add a legend for the main ROIs

# Create a font dictionary with bold weight
title_font = fm.FontProperties(weight='bold', size=18)
handles = [mpatches.Patch(facecolor=col, edgecolor='navy', label=roi) for roi, col in color_map.items()]
legend = ax.legend(handles=handles,
                   loc='lower right',
                   title='Hydrographic Regions',
                   fontsize=12,               # Specific point size instead of 'x-large'
                   title_fontproperties=title_font,         # Specific point size instead of 'xx-large'
                   facecolor='white',
                   edgecolor='navy',
                   framealpha=1,
                   frameon=True,
                   handlelength=3.0,          # Increase handle length (default is 2.0)
                   handleheight=1.5,          # Increase handle height (default is 0.8)
                   labelspacing=1.2,          # Increase space between legend entries
                   borderpad=1.0,             # Padding inside legend border
                   bbox_to_anchor=(0.95, 0.0),# Position it higher and more right
                   handletextpad=1.0)         # Space between handle and text

legend.get_frame().set_linewidth(2.5)



plt.show()

# %%



# %%
