# %% 1.0 

import glob
import re
import rasterio as rio
from rasterio.merge import merge
from rasterio.coords import BoundingBox
from shapely.geometry import box
from functools import reduce

import pprint as pp

data_dir = './data/new_test_v2/'

s2_files = glob.glob(f'{data_dir}/Sentinel2*.tif')
print(s2_files)

batch = s2_files

# %% 2.0 

def get_full_bounds(bounds_list: list):

    left = min(bb.left for bb in bounds_list)
    right = max(bb.right for bb in bounds_list)
    bottom = min(bb.bottom for bb in bounds_list)
    top = max(bb.top for bb in bounds_list)

    full_bounds = BoundingBox(left=left, right=right, bottom=bottom, top=top)

    return full_bounds

src_files = []

for path in batch:
    
    src = rio.open(path)
    src_files.append(src)
    print(src.profile['nodata'])
    bounds_list = [src.bounds for src in src_files]


full_bounds = get_full_bounds(bounds_list)
pp.pp(full_bounds)

merged, out_transform = merge(
    src_files, 
    bounds=full_bounds,
    method='max',
    nodata=None,
    masked=True
)

pp.pp(out_transform)

out_meta = src_files[2].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": merged.shape[1],
    "width": merged.shape[2],
    "transform": out_transform,
    "crs": src_files[0].crs
})

out_path = f'{data_dir}/merged_s2_test.tif'

with rio.open(out_path, 'w', **out_meta) as dest:
    dest.write(merged)

for src in src_files:
    src.close()

# %%
