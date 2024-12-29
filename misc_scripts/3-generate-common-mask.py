# %%

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import sieve
from skimage.morphology import binary_dilation, remove_small_objects, disk

# %% 1. Get extent for landsat image

with rasterio.open('./data/landsat-masks/ls-masks-test-v2.tif') as ls_src:
    ls_mask = ls_src.read(1)
    ls_meta = ls_src.meta
    

with rasterio.open('./data/sentinel2-masks/sentinel2-masks-v2-merged.tif') as s2_src:
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

with rasterio.open('./data/common_mask_v2.tif', 'w', **out_meta) as dst:
    dst.write(common_mask.astype(rasterio.uint8), 1)

# %% 2.0 Make a more conservative version of the mask

with rasterio.open('./data/common_mask_v2.tif') as src:
    common_mask = src.read(1)
    out_meta = src.meta

    common_mask_sieved = sieve(common_mask, size=100, connectivity=8)
    sieved_mask_bool = common_mask_sieved.astype(bool)

    dilation_size = 200
    structuring = np.ones((dilation_size, dilation_size), dtype=bool)

    mask_dilated = binary_dilation(common_mask_sieved, footprint=structuring)

with rasterio.open('./data/common_mask_sieved_dilated.tif', 'w', **out_meta) as dst:
    dst.write(mask_dilated, 1)


# %%
