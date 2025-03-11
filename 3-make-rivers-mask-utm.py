# %% 1.0 Libraries and directories
import os
import pprint as pp
import numpy as np
import geopandas as gpd

import rasterio as rio
from rasterio import features, warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds



roi_name = 'YKF_sub3'
out_res = 60 # ensure this matches the resolution of your analysis (e.g. 30m or 60m)
buffer_dist = 180 # The distance in meters to dilate/buffer the rivers

# These regions (MRD, TUK, AND) have the same river file
roi_prefix = roi_name.split('_')[0]
if roi_prefix in ['AND', 'TUK', 'MRD']:
    rivers_prefix = 'MRD'
else:
    rivers_prefix = roi_prefix

rivers_dir = f'./data/river_files/{rivers_prefix}_river.shp'
roi_dir = f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp'
out_dir = f'./data/river_files/'
out_path = f'{out_dir}/{roi_name}_binary_rivers_dilated{buffer_dist}_res{out_res}.tif'

rivers = gpd.read_file(rivers_dir)
if rivers.crs is None:
    rivers.set_crs('EPSG:4326', inplace=True)
    print('Rivers were missing crs')
else:
    print(rivers.crs)


sub_rois = gpd.read_file(roi_dir)
roi = sub_rois[sub_rois['sub_name'] == roi_name]

# %% 2.0 Functions

def clip_dilate_rivers(
    rivers: gpd.GeoDataFrame, 
    roi: gpd.GeoDataFrame, 
    buffer_dist: int
):
    """
    Reprojects the rivers and roi shapefile to local UTM
    Clips the Rivers to the roi extent
    Dilates rivers by the buffer distance
    """
    # Conversion to ROI's local UTM zone
    est_utm = rivers.estimate_utm_crs()
    est_utm_roi = roi.estimate_utm_crs()
    roi_utm = roi.to_crs(est_utm_roi)
    rivers_utm = rivers.to_crs(est_utm_roi)
    # Clip the rivers to roi and buffer thier shapes
    clipped_rivers = gpd.clip(rivers_utm, roi_utm)
    dilated_rivers = clipped_rivers.buffer(buffer_dist)
    # Bounds for the ROI to use in writing rivers raster
    roi_bounds = roi_utm.geometry.total_bounds

    return dilated_rivers

def make_river_mask(
    rivers_dilated: gpd.GeoDataFrame, 
    out_res: int,
    roi_mask_fp: str,
    out_path: str,
):
    """
    Makes a binary mask from rivers shapefile and clips the mask to the roi
    """

    with rio.open(roi_mask_fp) as ref:
        meta = ref.meta.copy()

    mask = features.rasterize(
        shapes=rivers_dilated.geometry,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    )

    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, indexes=1)

    print(f'Processed {out_path}')

# %% 3.0 Run the functions

rivers_dilated = clip_dilate_rivers(rivers, roi, buffer_dist)
roi_mask_fp = f'./data/roi_shapes/rois/rasterized_{roi_name}_shape_res{out_res}.tif'
make_river_mask(rivers_dilated, out_res, roi_mask_fp, out_path)

# %%
