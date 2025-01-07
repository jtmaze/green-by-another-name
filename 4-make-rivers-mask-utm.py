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

roi_name = 'YKF_sub1'
out_res = 30 # ensure this matches the resolution of your analysis
dilation_dist = 150 #meters

roi_prefix = roi_name.split('_')[0]
rivers_dir = f'./data/river_files/{roi_prefix}_river.shp'
roi_dir = f'./data/roi_shapes/{roi_name}_shape.shp'
out_dir = f'./data/river_files/'
rivers = gpd.read_file(rivers_dir)
roi = gpd.read_file(roi_dir)

# %% 2.0 Generate a mask image from the River shapefiles

def make_river_mask(roi: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, out_res: int):
    """
    Makes a binary mask from rivers shapefile and clips the mask to the roi
    """
    est_utm = roi.estimate_utm_crs()
    roi = roi.to_crs(est_utm)
    rivers = rivers.to_crs(est_utm)
    clipped_rivers = gpd.clip(rivers, roi)

    bounds = roi.geometry.total_bounds # Tuple with bounding box
    print(bounds)
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
        'crs': est_utm,
        'transform': transform,
        'compress': 'lzw'
    }

    mask = features.rasterize(
        shapes=clipped_rivers.geometry,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    )

    return mask, meta

def dilate_rivers(mask: np.array, 
                  meta: dict, 
                  dilation_dist: int, 
                  out_res: int):
    """
    Dilates a the river mask by a specified amount in meters.
    """
    buffer_amnt_pixels = int(dilation_dist / out_res)
    structuring_element = np.ones((buffer_amnt_pixels, buffer_amnt_pixels), dtype=bool)
    pp.pp(structuring_element)
    dilated = binary_dilation(mask, footprint=structuring_element)

    return dilated

def write(mask: np.array, meta: dict, out_path: str):
    """
    Writes binary mask to out_dir
    """
    if mask.dtype == bool:
        mask = mask.astype('uint8')
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)

    
    final_meta = meta.copy()

    pp.pp(meta)

    with rio.open(out_path, 'w', **final_meta) as dst:
        dst.write(mask)



# %% 3.0 Run the functions

out_path = f'{out_dir}/{roi_name}_binary_rivers_dilated{dilation_dist}_UTMretest.tif'
mask, meta = make_river_mask(roi, rivers, out_res=out_res)
dilated_mask = dilate_rivers(mask, meta, dilation_dist=dilation_dist, out_res=out_res)
write(mask=dilated_mask, meta=meta, out_path=out_path)

# %%
