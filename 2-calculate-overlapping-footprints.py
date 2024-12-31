"""
After exporting the coincident satellite footprints for a given date from GEE,
use this script to calculate the overlapping footprint area for each date.
"""
# %% 1.0 Libraries and directories

import pandas as pd
import geopandas as gpd
import os
import glob
import re

proj_dir = '/Users/jtmaz/Documents/projects/green-by-another-name'
os.chdir(proj_dir)

footprint_file_list = glob.glob('./data/image_footprints/*.shp')

roi_pattern = r'roi_(.*?)_years'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

unique_rois = extract_unique(footprint_file_list, roi_pattern)

# %% 2.0 Find overlap as % of roi for each image date. 

for r in unique_rois:
    # Read footprints and rename the column
    ls8_path = [f for f in footprint_file_list if re.search(fr'footprints_ls8_roi_{r}.*\.shp$', f)][0]
    s2_path = [f for f in footprint_file_list if re.search(fr'footprints_s2_roi_{r}.*\.shp$', f)][0]
    ls8_footprints = gpd.read_file(ls8_path).rename(
        columns={'formatted_': 'date'})
    s2_footprints = gpd.read_file(s2_path).rename(
        columns={'formatted_': 'date'})
    
    # Read the original roi to find the % coincident overlap on a given date.
    roi_files_path = glob.glob('./data/roi_shapes/*.shp')
    roi_file = [f for f in roi_files_path if re.search(fr'{r}.*\_shape.shp$', f)][0]
    roi_shape = gpd.read_file(roi_file)
    roi_est_crs = roi_shape.estimate_utm_crs()
    roi_shape = roi_shape.to_crs(roi_est_crs)
    roi_area = (roi_shape.geometry.iloc[0].area / 1_000_000)
    print(f'{roi_area} in square meters')

    # Convert to local UTM for area opperations
    ls8_footprints = ls8_footprints.to_crs(roi_est_crs)
    s2_footprints = s2_footprints.to_crs(roi_est_crs)
    merged = ls8_footprints.merge(s2_footprints,
                                  on='date',
                                  suffixes=('_ls8', '_s2'))
    
    # Merge the satellite footprints for calculations
    merged['int'] = merged.apply(
        lambda row: row['geometry_ls8'].intersection(row['geometry_s2']),
        axis=1
    )
    merged = merged.set_geometry('int')
    merged.crs = roi_est_crs
    
    merged['int_sqkm'] = (merged['int'].area / 1_000_000).round(2)

    # Select the dates with best coverage in the roi
    top_dates = merged.sort_values(by='int_sqkm', 
                                   ascending=False
                                   ).head(10)
    top_dates.drop(columns=['geometry_ls8', 'geometry_s2'], inplace=True)
    top_dates['per_cover'] = (top_dates['int_sqkm'] / roi_area * 100).round(0)

    top_dates.to_crs('EPSG: 4326')
    top_dates.to_file(f'./data/overlap_dates_for_roi/{r}_overlap_dates.shp')




# %%
