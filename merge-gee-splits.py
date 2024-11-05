# %%
import os
import glob 
import rasterio as rio
from rasterio.merge import merge


import xarray as xr
import rioxarray as rxr

s2_masks_dir = './data/sentinel2-masks'
s2_masks = glob.glob(s2_masks_dir + '/*')

s2_data_dir = './data/sentinel2-images'
s2_data = glob.glob(s2_data_dir + '/*')

ls_data_dir = './data/landsat-images'
ls_data = glob.glob(ls_data_dir + '/*')
# %%

rasters = []

for p in ls_data:
    print(p)
    r = rio.open(p)
    rasters.append(r)

merged, out_trans = merge(rasters)
out_meta = r.meta.copy()
out_meta.update({"driver": "GTiff",
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "transform":out_trans,
                    'crs': r.crs
                })

output_path = './data/landsat-images/landsat-images-merged.tif'
with rio.open(output_path, "w", **out_meta) as dest:
    dest.write(merged)


# %%
