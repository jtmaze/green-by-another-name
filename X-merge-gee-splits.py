# 1.0 Libraries and directories

import glob
import re
import rasterio as rio
from rasterio.merge import merge

ls_raw_dir = './data/landsat-gee'
ls_data = glob.glob(ls_raw_dir + '/*.tif')

s2_raw_dir = './data/s2-gee'
s2_masks = glob.glob(s2_raw_dir + '/*.tif')

roi_pattern = r'.*_(.*?)_resolution.*\.tif'
s2_date_pattern = r'.*Sentinel2_(.*?)_.*\.tif'
ls_date_pattern = r'.*Landsat8_(.*?)_.*\.tif'
resolution_pattern = r'_resolution(\d+)-\d+.*\.tif'

# %% 2.0 Merge functions

def extract_unique(raster_list, pattern):

    unique_items = set()
    for f in raster_list: 
        match = re.search(pattern, f)
        if match:
            unique_items.add(match.group(1))

    return list(unique_items)


def merge_mosiac(raster_list, satellite, date, roi, resolution):

    mosaic_list = [f for f in raster_list if satellite in f and date in f and roi in f and resolution in f]
    print(mosaic_list)
    img_list = []

    for p in mosaic_list:
        r = rio.open(p)
        img_list.append(r)

    merged, out_trans = merge(img_list)
    out_meta = r.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": out_trans,
        'crs': r.crs
    })

    output_path = f'./data/merged/{satellite}_date_{date}_roi_{roi}_resolution{resolution}.tif'
    with rio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged)

    print('Merged raster saved to:', output_path)

def merge_raster_list(raster_list, satellite, roi_pattern, date_pattern, resolution_pattern):

    dates = extract_unique(raster_list, date_pattern)
    rois = extract_unique(raster_list, roi_pattern)
    resolutions = extract_unique(raster_list, resolution_pattern)

    for date in dates:
        for roi in rois:
            for resolution in resolutions:
                merge_mosiac(raster_list, satellite, date, roi, resolution)



# %% 3.0 Merge the d
merge_raster_list(ls_data, 'Landsat8', roi_pattern, ls_date_pattern, resolution_pattern)
merge_raster_list(s2_masks, 'Sentinel2', roi_pattern, s2_date_pattern, resolution_pattern)




# %%
