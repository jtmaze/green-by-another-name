"""
This script dilates and erodes the PLD lakes and writes them to a raster file
"""
# %% 1.0 Libraries and directories

import pprint as pp
from math import inf
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio

from rasterio.features import rasterize

out_dir = './data/pld_rasterized/'
roi_name = 'YKD_sub2'
roi_prefix = roi_name.split('_')[0]
res = 30 # 30 or 60 meters
buffers = [-120, -60, -30, 0, 30, 60, 120] # Buffer sizes in meters to dilate and erode the PLD lakes

if res == 60:
    buffers = [-120, -60, 0, 60, 120] # Change buffers to increments matching the resolution

# Dictionary of lake size bins to make PLD masks corresponding to lake sizes. 
lake_size_bins = {
    'smallest': (0, 0.05),
    'small': (0.05, 0.5),
    'medium': (0.5, 1),
    'large': (1, inf)
}

pld = gpd.read_file(f'./data/pld_shapes/{roi_prefix}_pld_clipped.shp')
sub_rois = gpd.read_file(f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp')
roi = sub_rois[sub_rois['sub_name'] == roi_name]
est_utm = pld.estimate_utm_crs()
est_utm_roi = roi.estimate_utm_crs()

"""
Worth adding check to ensure est_utm is same for lakes and boundary
"""
print(est_utm, est_utm_roi)

pld_utm = pld.to_crs(est_utm)
roi_utm = roi.to_crs(est_utm) 


def pld_buffer_img_clip(
    gdf_utm: gpd.GeoDataFrame, 
    buffer_size: int, 
    clip_bounds: gpd.GeoDataFrame
):
    """
    Clips the PLD lakes to the roi boudary
    Returns a tupple of dilated/eroded lakes and a band name for writing to raster
    """
    # Clip the lakes to the common image boundary
    out_gdf = gdf_utm.copy()
    out_gdf = gpd.clip(out_gdf, clip_bounds)

    # Buffer the lakes
    out_gdf = out_gdf.buffer(buffer_size)

    # Calculate number of small lakes removed by errosion opperation
    total_lakes = len(out_gdf)
    out_gdf = out_gdf[~out_gdf.is_empty]
    out_lakes = len(out_gdf)
    print(f'{total_lakes - out_lakes} of {total_lakes} lakes were removed due to buffer size')

    band_name = f'buffered_{buffer_size}m'

    return (out_gdf, band_name)


def rasterize_buffers(
    gdf: gpd.GeoDataFrame, 
    band_name: str, 
    img_path: str,
    out_path: str,
    band_idx: int
):
    """
    Rasterize the dilated/eroded lake shapes and write them to disk
    """
    # Get meta data data from rivers mask
    with rio.open(img_path) as src_img:
        img_meta = src_img.meta
    out_meta = img_meta.copy()
    #print(out_meta['crs'])

    # For first band (i.e, no file yet), mode is write
    if band_idx == 1:
        mode = 'w'
        out_meta.update({
            'count': len(buffers)
        })
    # For subsequent bands, r+ mode on existing file
    else:
        mode = 'r+'

    # Convert the lakes the local UTM from the image
    gdf = gdf.to_crs(img_meta['crs'])
    # Rasterize the the lake shapes to a binary raster
    # The image path to roi shape raster, becuase satellite images have variable footprints by date.
    lake_raster = rasterize(
        shapes=[(geom, 1) for geom in gdf.geometry],
        out_shape=(img_meta['height'], img_meta['width']),
        transform=img_meta['transform'],
        all_touched=True,
        fill=0,
        default_value=1,
        dtype=rio.uint8 
    )
    # Write the output
    with rio.open(out_path, mode, **out_meta) as dst:
        dst.write(lake_raster, band_idx)
        dst.set_band_description(band_idx, band_name)

    #print(f'{band_name} rasterized to {img_meta['crs']}')
    

img_path = f'./data/roi_shapes/rois/rasterized_{roi_name}_shape_res{res}.tif'

lake_size_summaries = []

total_lake_area = pld_utm.area.sum() / 1_000_000
print(f'Total lake area: {total_lake_area:.2f} km^2')

for size, (min_area, max_area) in lake_size_bins.items(): 
    # Designate out_path for the lake
    out_path = f'{out_dir}/{roi_name}_lake_masks_res{res}_{size}.tif'
    # Filter lakes by size category
    # caluculate the total lake_area
    total_lake_area = pld_utm.area.sum() / 1_000_000
    print(f'Total lake area: {total_lake_area:.2f} km^2')
    if max_area != inf:
        lakes_in_category = pld_utm[
            (pld_utm.area/1_000_000 >= min_area) &
            (pld_utm.area/1_000_000 < max_area)
        ].copy()
    else:
        lakes_in_category = pld_utm[pld_utm.area/1_000_000 >= min_area]

    # Calculate the size categories total lake area    
    category_lake_area = lakes_in_category.area.sum() / 1_000_000

    # Summary dataframe to track roi's 
    summary = {
        'roi_name': roi_name,
        'lake_size': size,
        'count': len(lakes_in_category),
        'area_proportion': (category_lake_area / total_lake_area * 100)
    }
    lake_size_summaries.append(summary)

    for i, buffer in enumerate(buffers):
        pld_buffered, band_name = pld_buffer_img_clip(lakes_in_category, buffer, roi_utm)
        rasterize_buffers(pld_buffered, band_name, img_path, out_path, band_idx=i+1)


summary_df = pd.DataFrame(lake_size_summaries)

lake_summary_path = './data/lake_size_summaries.csv'

if not os.path.exists(lake_summary_path):
    summary_df.to_csv(lake_summary_path, index=False)
    print("Adding first roi's data to lake_size_summaries.csv")
else:
    existing_df = pd.read_csv(lake_summary_path)
    new_data = summary_df[~summary_df['roi_name'].isin(existing_df['roi_name'])]

    if not new_data.empty:
        combined = pd.concat([existing_df, new_data], ignore_index=True)
        combined.to_csv(lake_summary_path, index=False)
    else:
        print("ROI Already Processed")


# %%
