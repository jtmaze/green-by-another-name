# %% 
import numpy as np
import rasterio as rio
import geopandas as gpd

# %% 1. Mask the LandSat and Sentinel2 Images 
# A) Clip the LandSat and Sentinel-2 Images to the combined boundary
# B) ?? Resample the Sentinel-2 image to 30 meter resolution
# C) Apply the common mask to both images
# D) Write to file

s2_boundary = gpd.read_file('./data/s2_boundary.shp')
ls_boundary = gpd.read_file('./data/ls_boundary.shp')

with rio.open('./data/sentinel2-images/s2-images-merged.tif') as s2_img_src:
    s2_img = read() # Just do green band for now!
    s2_meta = s2_img_src.meta

s2_img_resample = np.empty_like()

with rio.open('./data/landsat-images/landsat-images-merged.tif') as ls_img_src:
    ls_img = read() # start with green!
    ls_meta = ls_img_src.meta

s2_img_resample = np.empty_like(ls_img)

with rio.open('./data/common_mask.tif') as cmask_src:
    cmask = read(1)
    cmask_meta = cmask.meta()

