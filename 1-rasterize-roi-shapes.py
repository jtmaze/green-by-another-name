"""
Rasterize ROI Shapes

This script produces binary rasters based on the image target area's width and height.
The projection is the master projection that the Sentinel-2 and Landsat-8 images are
projected to before analysis.

Input:
- Shapefile containing ROIs with 'sub_name' and 'utm_epsg' attributes
- Each ROI will be rasterized separately

Output:
- GeoTIFF binary masks (1=inside ROI, 0=outside) at specified resolution (30m 0r 60m for this project)
- Files are named with ROI name and resolution in the filename
- Projected to appropriate UTM coordinate reference system
"""
# %% 1.0 Libraries and directories

import geopandas as gpd
import numpy as np

import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ROI configuration - prefix identifies the shapefile
roi_prefix = 'YKF'
data_dir = './data/roi_shapes/rois/'
roi_shapes = gpd.read_file(f'{data_dir}{roi_prefix}_sub_rois.shp')

# %% 3.0 Rasterization process

resolution = 30 # meters - defines the pixel size of output rasters
original_crs = roi_shapes.crs
print(original_crs)

# Process each ROI polygon individually
for idx, roi in roi_shapes.iterrows():

    geometry = roi['geometry']  # Extract the polygon geometry
    roi_name = roi['sub_name']  # Get the ROI name for filename

    # Convert ROI's CRS to local UTM for proper distance measurements
    single_gdf = gpd.GeoDataFrame(geometry=[geometry], crs=original_crs)
    est_utm = single_gdf.estimate_utm_crs()
    if est_utm != roi['utm_epsg']:
        print("ERRORR UTMS aren't matching")

    reproj_gdf = single_gdf.to_crs(est_utm)
    mask_geom = reproj_gdf.geometry.iloc[0]

    # Calculate raster dimensions based on geometry bounds and resolution
    minx, miny, maxx, maxy = mask_geom.bounds
    width = int((maxx - minx) / resolution)  # Number of pixels in X direction
    height = int((maxy - miny) / resolution)  # Number of pixels in Y direction

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create binary mask: 1 for area inside ROI, 0 for outside
    mask = rasterize(
        [(mask_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Save the binary mask as a GeoTIFF file
    out_path = f'{data_dir}rasterized_{roi_name}_shape_res{resolution}.tif'
    with rio.open(
        out_path,
        'w', 
        driver='GTiff', 
        height=height,
        width=width,
        count=1,      # Single band binary raster
        dtype=np.uint8,
        crs=est_utm,  # Using appropriate UTM projection
        transform=transform
    ) as dst:
        dst.write(mask, 1)

    print(f"Wrote {out_path}")

