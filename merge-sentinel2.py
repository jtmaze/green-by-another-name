# %%
import pprint as pp

import os
import glob 
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

s2_data_dir = './data/sentinel2-images'
s2_data = glob.glob(s2_data_dir + '/*.tif')

# %%
merged, out_transform = merge(s2_data, resampling=Resampling.average)

with rio.open(s2_data[0]) as src: 
    out_meta = src.meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": merged.shape[1],
    "width": merged.shape[2],
    "transform": out_transform,
    "crs": out_meta['crs']
})

output_path = './data/sentinel2-images/sentinel2-images-merged.tif'
with rio.open(output_path, "w", **out_meta) as dest:
    dest.write(merged)


# %%
