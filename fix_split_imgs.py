# %% 1.0 

import glob
import re
import numpy as np
from itertools import product

import rasterio as rio
from rasterio.merge import merge
from rasterio.coords import BoundingBox
from rasterio.warp import reproject, Resampling

from image_analysis_functions import extract_unique

import pprint as pp

level = 'sr' #level should be 'sr' or 'toa'
roi_name = 'YKF_sub3'


data_dir = './data/new_test/'
#roi_prefix = roi_name.split('_')[0]
river_dir = './data/river_files/'
river_path = f'{river_dir}/{roi_name}_binary_rivers_dilated180.tif'
all_files = glob.glob(f'{data_dir}/*.tif')
all_s2_files = glob.glob(f'{data_dir}/Sentinel2*.tif')
all_ls8_files = glob.glob(f'{data_dir}/Landsat8*.tif')

roi_pattern = r'roi_(.*?)_resampled'
date_pattern = r'date_(.*?)_roi'
resamp_pattern = r'resampled_(.*?)_date'
unique_rois = extract_unique(all_files, roi_pattern)
unique_dates = extract_unique(all_files, date_pattern)

batches = list(product(unique_dates, unique_rois))
print(batches)



# %% 2.0 

# def get_full_bounds(bounds_list: list):

#     left = min(bb.left for bb in bounds_list)
#     right = max(bb.right for bb in bounds_list)
#     bottom = min(bb.bottom for bb in bounds_list)
#     top = max(bb.top for bb in bounds_list)

#     full_bounds = BoundingBox(left=left, right=right, bottom=bottom, top=top)

#     return full_bounds


######################################################


ref_raster = rio.open(river_path)
ref_trans = ref_raster.transform
ref_crs = ref_raster.crs
ref_bounds = ref_raster.bounds


src_files = []
dst_raster_list = []

for idx, path in enumerate(batch):
    print(path)
    with rio.open(path) as src:
        # Create destination array with the shape (# bands, height, width)
        src_data = src.read()
        dst_data = np.empty((src.count, ref_raster.height, ref_raster.width), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_trans,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )
        out_meta = ref_raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count": src.count,
            "height": dst_data.shape[1],
            "width": dst_data.shape[2],
            "transform": ref_trans,
            "dtype": 'float32',
            "crs": ref_crs,
            "bounds": ref_bounds
        })

        out_path = f'{data_dir}/reproj_{sat}_test_{idx}.tif'
        dst_raster_list.append(dst_data)
    # with rio.open(out_path, 'w', **out_meta) as dst:
    #     dst.write(dst_data, indexes=list(range(1, dst_data.shape[0] + 1)))

# %%

stacked_rasters = np.stack(dst_raster_list)
print(stacked_rasters.shape)
imgs_mean = np.nanmean(stacked_rasters, axis=0)
print(imgs_mean.shape)
zeros_count = (imgs_mean == 0).sum()
nan_count = np.isnan(imgs_mean).sum()
valid_count = (imgs_mean > 0).sum()
print(valid_count)
print(nan_count)
print(zeros_count)
imgs_mean[imgs_mean == 0] = np.nan
mean_meta = out_meta.copy()
mean_out_path = f'{data_dir}/mean_{sat}_composite.tif'
with rio.open(mean_out_path, 'w', **mean_meta) as dst:
    dst.write(imgs_mean, indexes=list(range(1, imgs_mean.shape[0] + 1)))

print(f"Mean composite saved to: {mean_out_path}")

# %%

