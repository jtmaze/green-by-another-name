# %% 1.0 Libraries and directories

import geopandas as gpd
import numpy as np

import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

roi_prefix = 'YKF'
data_dir = './data/roi_shapes/rois/'
roi_shapes = gpd.read_file(f'{data_dir}{roi_prefix}_sub_rois.shp')

# %% 3.0 

resolution = 30 # meters
original_crs = roi_shapes.crs
print(original_crs)

for idx, roi in roi_shapes.iterrows():

    geometry = roi['geometry']
    roi_name = roi['sub_name']

    # Need to convert the roi's crs to local utm
    single_gdf = gpd.GeoDataFrame(geometry=[geometry], crs=original_crs)
    est_utm = single_gdf.estimate_utm_crs()
    if est_utm != roi['utm_epsg']:
        print("ERRORR UTMS aren't matching")

    reproj_gdf = single_gdf.to_crs(est_utm)
    mask_geom = reproj_gdf.geometry.iloc[0]

    # Get bounds in UTM
    minx, miny, maxx, maxy = mask_geom.bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    mask = rasterize(
        [(mask_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    out_path = f'{data_dir}rasterized_{roi_name}_shape_res{resolution}.tif'
    with rio.open(
        out_path,
        'w', 
        driver='GTiff', 
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=est_utm,
        transform=transform
    ) as dst:
        dst.write(mask, 1)

    print(f"Wrote {out_path}")


# %%
