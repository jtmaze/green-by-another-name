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

def reproject_write(final_crs: str, mask: np.array, meta: dict, out_path: str):
    """
    Reprojects binary mask to new CRS and writes to file
    
    Args:
        final_crs (str): Target CRS (e.g. 'EPSG:4326')
        mask (np.array): Input binary mask
        meta (dict): Original raster metadata
        out_path (str): Output file path
    """
    if mask.dtype == bool:
        mask = mask.astype('uint8')

    
    final_crs_obj = rio.crs.CRS.from_string(final_crs)
    final_meta = meta.copy()
    final_meta.update({'crs': final_crs})

    pp.pp(meta)

    dst_affine, dst_width, dst_height = warp.calculate_default_transform(
        src_crs=meta['crs'],
        dst_crs=final_crs_obj,
        width=meta['width'],
        height=meta['height'],
        left=meta['transform'][2],
        bottom=meta['transform'][5],
        right=meta['transform'][2] + meta['transform'][0] * meta['width'],
        top=meta['transform'][5] + meta['transform'][4] * meta['height']
    )

    final_meta.update({
        'transform': dst_affine,
        'width': dst_width,
        'height': dst_height
    })

    with rio.open(out_path, 'w', **final_meta) as dst:
        reproject(
            source=mask,
            destination=rio.band(dst, 1),
            src_transform=meta['transform'],
            src_crs=meta['crs'],
            dst_transform=dst_affine,
            dst_crs=final_crs,
            resampling=Resampling.nearest
        )

    print(f'River mask reprojected and written to {out_path}')


# %% 3.0 Run the functions

out_path = f'{out_dir}/{roi_name}_binary_rivers_dilated{dilation_dist}.tif'
mask, meta = make_river_mask(roi, rivers, out_res=out_res)
dilated_mask = dilate_rivers(mask, meta, dilation_dist=dilation_dist, out_res=out_res)
reproject_write(final_crs='EPSG:4326', mask=dilated_mask, meta=meta, out_path=out_path)

# %%
