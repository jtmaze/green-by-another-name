# %% 1.0 Libraries and directories
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio import features
from rasterio.transform import from_bounds

roi_name = 'YKF_sub1'
out_res = 30 # ensure this matches the resolution of your analysis


roi_prefix = roi_name.split('_')[0]
rivers_dir = f'./data/river_files/{roi_prefix}_river.shp'
roi_dir = f'./data/roi_shapes/{roi_name}_shape.shp'
rivers = gpd.read_file(rivers_dir)
roi = gpd.read_file(roi_dir)

# %% 2.0 Generate a mask image from the River shapefiles

def make_river_mask(roi: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, out_res: int, out_path: str):
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
        fill=1,
        default_value=0,
        dtype='uint8'
    )

    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, indexes=1)

# %% 3.0

make_river_mask(roi, rivers, out_res=30, out_path='./data/test.tiff')
# %%
