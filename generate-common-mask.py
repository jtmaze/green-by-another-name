# %%

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# %% 1. Get extent for landsat image

with rasterio.open('./data/landsat-masks/ls-masks-test.tif') as ls_src:
    ls_mask = ls_src.read(1)
    ls_meta = ls_src.meta
    

with rasterio.open('./data/sentinel2-masks/s2-masks-merged.tif') as s2_src:
    s2_mask = s2_src.read(1)
    s2_meta = s2_src.meta

s2_mask_resampled = np.empty_like(ls_mask)
reproject(
    source=s2_mask,
    destination=s2_mask_resampled,
    src_transform=s2_src.transform,
    src_crs=s2_src.crs,
    dst_transform=ls_src.transform,
    dst_crs=ls_src.crs,
    resampling=Resampling.nearest #nearest is best for catagorical data
)

common_mask = np.where((s2_mask_resampled == 1) | (ls_mask == 1), 1, 0)
out_meta = ls_meta.copy()


with rasterio.open('./data/common_mask.tif', 'w', **out_meta) as dst:
    dst.write(common_mask.astype(rasterio.uint8), 1)







# %%
