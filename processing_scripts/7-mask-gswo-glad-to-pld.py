# %%

import os
import glob
import re

import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

gswo_glad_dir = './data/gswo_glad_raw/'
out_dir = './data/gswo_glad_pld/'
files = glob.glob(f'{gswo_glad_dir}/*.tif')


# %% Reproject and mask the GSWO and PLD data

for f in files:

    # Get the info from file paths
    roi_match = re.search(r"_roi_(.*?)\.tif", f)
    ds_match = re.search(r"dataset_(.*?)_month", f)
    month_match = re.search(r'month_(.*?)_roi', f)

    if roi_match and ds_match and month_match:
        roi_name = roi_match.group(1)
        ds = ds_match.group(1)
        month = month_match.group(1)
    
    # Fetch the PLD data for masking
    pld_path = f'./data/pld_rasterized/{roi_name}_lake_masks_res30.tif'
    if not os.path.exists(pld_path):
        continue

    with rio.open(pld_path) as pld, rio.open(f) as src:

        dst_meta = pld.meta.copy()
        dst_meta.update({
            'nodata': -1,
            'dtype': 'float32',
            'count': 1
        })

        out_data = np.full(
            (src.count, pld.height, pld.width),
            fill_value=-1,
            dtype=np.float32
        )

        src_data = src.read(1)
        if ds == "GSWO": # NOTE: Need to rescale the GSWO data
            src_data = src_data * 100

        pld_data = pld.read(6)
        # Reproject occurence rasters to UTM, same as PLD
        reproject(
            source=src_data,
            destination=out_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=pld.transform,
            dst_crs=pld.crs,
            resampling=Resampling.bilinear
        )
    
        # mask the occurance rasters to match PLD
        out_data = np.where(pld_data == 1, out_data, -1)
        out_data = np.clip(out_data, -1, 100)

    out_fp = f'{out_dir}/dataset_{ds}_month_{month}_roi_{roi_name}.tif'

    with rio.open(out_fp, 'w', **dst_meta) as dst:
        dst.write(out_data)


# %%
