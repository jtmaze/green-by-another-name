# 1.0 Libraries and directories

import glob
import rasterio as rio
from rasterio.merge import merge

ls_data_dir = './data/landsat-images'
ls_data = glob.glob(ls_data_dir + '/*.tif')

s2_masks_dir = './data/sentinel2-masks'
s2_masks = glob.glob(s2_masks_dir + '/s2-mask-t*.tif')

# %% 2.0 Merge Landsat images

def merge_raster_list(raster_list, dataset):

    rasters = []

    for p in raster_list:
        print(p)
        r = rio.open(p)
        rasters.append(r)

    merged, out_trans = merge(rasters)
    out_meta = r.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": out_trans,
        'crs': r.crs
    })

    output_path = f'./data/{dataset}/{dataset}-merged.tif'
    with rio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged)

    print('Merged raster saved to:', output_path)

    return

#merge_raster_list(ls_data, 'landsat-images')
merge_raster_list(s2_masks, 'sentinel2-masks-v2')


# %% 3.0 Merge Sentinel-2 masks

