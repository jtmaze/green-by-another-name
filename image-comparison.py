# %% 
import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds
import rioxarray as rxr
import geopandas as gpd

# %% 1. Mask the LandSat and Sentinel2 Images 
# B) ?? Resample the Sentinel-2 image to 30 meter resolution
# C) Apply the common mask to both images
# D) Write to file


ls_green_band = 2 # Green is the 2nd band in export
s2_green_band = 2 # Green is the 2nd band in the export, even though its Band#3 in Sentinel-2

with rio.open('./data/sentinel2-images/sentinel2-images-merged.tif') as s2_img_src, \
    rio.open('./data/landsat-images/landsat-images-merged.tif') as ls_img_src:
    s2_bounds = s2_img_src.bounds
    s2_meta = s2_img_src.meta
    ls_bounds = ls_img_src.bounds
    ls_meta = ls_img_src.meta

    # Bounds are slightly different, make common bounds for reading the data
    # !!! This depends on hemisphere and CRS !!!
    left = max(ls_bounds.left, s2_bounds.left)
    right = min(ls_bounds.right, s2_bounds.right)
    top = min(ls_bounds.top, s2_bounds.top)
    bottom = max(ls_bounds.bottom, s2_bounds.bottom)
    intersection_bounds = (left, bottom, right, top)

    window_s2 = from_bounds(*intersection_bounds, s2_img_src.transform)
    window_ls = from_bounds(*intersection_bounds, ls_img_src.transform) 

    s2_data = s2_img_src.read(s2_green_band, window=window_s2)
    ls_data = ls_img_src.read(ls_green_band, window=window_ls)

    print(s2_data.shape)
    print(ls_data.shape)

# with rio.open('./data/common_mask.tif') as cmask_src:
#     cmask = cmask_src.read(1)
#     cmask_meta = cmask.meta





# %%
