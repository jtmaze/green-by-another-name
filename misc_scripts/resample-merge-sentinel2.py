# %%
import pprint as pp

import os
import glob 
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge

s2_data_dir = './data/sentinel2-images'
s2_data = glob.glob(s2_data_dir + '/*.tif')

# %% For Sentinel-2 Images (not masks), resample to larger resolution (30m)

resample_res = 30 #meters
resampled_files = []

roi = gpd.read_file('./data/YKflats_roi_shape.shp')
est_utm = roi.estimate_utm_crs()
pp.pp(est_utm)

rasters = []

for i, p in enumerate(s2_data):
    out_path = f'./data/sentinel2-images/s2-resample-image-number{i}.tif'
    print(p)
    with rio.open(p) as src:
        src_crs = src.crs
        src_meta = src.meta.copy()
        src_trans = src_meta['transform']
        src_width = src.width
        src_height = src.height
        src_dtype = src.dtypes[0]
        print(src_dtype)
        bounds = src.bounds


        new_trans, new_width, new_height = calculate_default_transform(
            src_crs=src_crs,
            dst_crs=est_utm, # Shouldn't need to reproject, but arg is here incase
            width=src_width, 
            height=src_height,
            left=bounds.left,
            bottom=bounds.bottom,
            right=bounds.right,
            top=bounds.top,
            resolution=resample_res
        )
        pp.pp(new_trans)

        out_meta = src_meta.copy()
        out_meta.update({
            'transform': new_trans,
            'height': new_height,
            'width': new_width
        })

        with rio.open(out_path, 'w', **out_meta) as dst:
            for band in range(1, src.count + 1):
                data = src.read(band)

                out_data = np.empty((new_height, new_width), dtype=src_dtype)
                reproject(
                    source=data,
                    destination=out_data,
                    src_transform=src_trans,
                    src_crs=src_crs,
                    dst_transform=new_trans, 
                    dst_crs=est_utm,
                    resampling=Resampling.average
                )

                dst.write(out_data, band)

    print('Resample a split file')
    resampled_files.append(out_path)


# %%
merged, out_transform = merge(resampled_files, resampling=Resampling.average)

with rio.open(resampled_files[0]) as src: 
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
