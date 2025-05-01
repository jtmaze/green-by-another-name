# %% 1.0 Libraries and directories

import glob
import geopandas as gpd
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
valid_roi_dates = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch2.csv')
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
    figsize=(12, 10),
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
cmap = mpl.colors.LinearSegmentedColormap.from_list('RYG', ['red','yellow','green'])
footprints.plot(
    ax=ax,
    column='roi_img_count',
    cmap=cmap,
    alpha=1,
    edgecolor='black',
    linewidth=0.1,
    legend=False
)

# 6. Manually add a vertical colorbar for roi_img_count
norm = mpl.colors.Normalize(
    vmin=footprints['roi_img_count'].min(),
    vmax=footprints['roi_img_count'].max()
)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []  # dummy array for the ScalarMappable
cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='vertical',    # vertical bar on the right
    fraction=0.04,             # size of the colorbar
    pad=0.02                   # distance from the plot
)
cbar.set_label('Coincident Image Count')

# 7. Finish up
plt.title('Sub Regions Colored by Image Count with Main Boundaries')
plt.tight_layout()
plt.show()