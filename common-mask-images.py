# %% 1.0 Libraries
import pprint as pp

import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds
import rioxarray as rxr
import geopandas as gpd

# %% 2. Use common cloud mask on the LandSat and Sentinel2 Images 

band_color = 'green'

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

with rio.open('./data/common_mask.tif') as cmask_src:
    window_cmask = from_bounds(*intersection_bounds, cmask_src.transform)
    cmask = cmask_src.read(1, window=window_cmask)
    cmask_meta = cmask_src.meta

    print(cmask.shape)

s2_data = np.where(cmask != 1, s2_data, np.nan)
ls_data = np.where(cmask != 1, ls_data, np.nan)
negative_ls_pix = np.sum(ls_data < 0) / ls_data.size * 100
print(f'Negative Landsat pixels: {negative_ls_pix:.2f}%')
# Convert negative Landsat values to 0
#ls_data = np.where(ls_data < 0, 0, np.nan)

# Save the masked images
out_meta = s2_meta.copy()
out_meta.update({
    'count': 2,
    'dtype': 'float64',
    'height': s2_data.shape[0],
    'width': s2_data.shape[1]
})

with rio.open(f'./data/masked_images/{band_color}-masked.tif', 'w', **out_meta) as dst:
    dst.write(ls_data, 1)
    dst.set_band_description(1, f'landsat-{band_color}')
    dst.write(s2_data, 2)
    dst.set_band_description(2, f'sentinel2-{band_color}')






