# %% 1.0 Libraries and directories
import os
import pprint as pp
import numpy as np
import geopandas as gpd

import rasterio as rio
from rasterio import features, warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds

import skimage as ski
from skimage.morphology import binary_dilation

roi_name = 'AKCP_sub1'
out_res = 30 # ensure this matches the resolution of your analysis (e.g. 30m or 60m)
buffer_dist = 180 # The distance in meters to dilate/buffer the rivers

roi_prefix = roi_name.split('_')[0]
rivers_dir = f'./data/river_files/{roi_prefix}_river.shp'
roi_dir = f'./data/roi_shapes/{roi_name}_shape.shp'
out_dir = f'./data/river_files/'
out_path = f'{out_dir}/{roi_name}_binary_rivers_dilated{buffer_dist}.tif'

rivers = gpd.read_file(rivers_dir)
roi = gpd.read_file(roi_dir)

# %% 2.0 Generate a mask image from the River shapefiles

def clip_dilate_rivers(rivers: gpd.GeoDataFrame, roi: gpd.GeoDataFrame, buffer_dist: int):
    """"""
    # Conversion to ROI's local UTM zone
    roi_est_utm = roi.estimate_utm_crs()
    roi_utm = roi.to_crs(roi_est_utm)
    rivers_utm = rivers.to_crs(roi_est_utm)
    # Clip the rivers to roi and buffer thier shapes
    clipped_rivers = gpd.clip(rivers_utm, roi_utm)
    dilated_rivers = clipped_rivers.buffer(buffer_dist)
    # Bounds for the ROI to use in writing rivers raster
    roi_bounds = roi_utm.geometry.total_bounds

    return dilated_rivers, roi_bounds

def make_river_mask(bounds: tuple, 
                    rivers_dilated: gpd.GeoDataFrame, 
                    out_res: int,
                    out_path: str):
    """
    Makes a binary mask from rivers shapefile and clips the mask to the roi
    """
    # Get metadata from the roi's bounds (local UTM)
    width = int((bounds[2] - bounds[0]) / out_res)
    height = int((bounds[3] - bounds [1]) / out_res)
    transform = from_bounds(
        bounds[0], bounds[1],
        bounds[2], bounds[3],
        width, height
    )

    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 2,
        'width': width,
        'height': height,
        'count': 1,
        'crs': rivers_dilated.crs, #crs from dataframe (local UTM)
        'transform': transform,
        'compress': 'lzw'
    }

    mask = features.rasterize(
        shapes=rivers_dilated.geometry,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    )

    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, indexes=1)

    print(f'Processed {out_path}')

# %% 3.0 Run the functions

rivers_dilated, bounds = clip_dilate_rivers(rivers, roi, buffer_dist)
make_river_mask(bounds, rivers_dilated, out_res, out_path)

# %%
