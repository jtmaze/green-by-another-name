"""
After exporting the coincident satellite footprints for a given date from GEE,
use this script to calculate the overlapping footprint area for each date.
"""
# %% 1.0 Libraries and directories

import pandas as pd
import geopandas as gpd
import glob
import re

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
    # Read footprints and rename the date column
    ls8_path = [f for f in footprint_file_list if re.search(fr'footprints_ls8_roi_{r}.*\.shp$', f)][0]
    s2_path = [f for f in footprint_file_list if re.search(fr'footprints_s2_roi_{r}.*\.shp$', f)][0]
    ls8_footprints = gpd.read_file(ls8_path).rename(
        columns={'formatted_': 'date'})
    s2_footprints = gpd.read_file(s2_path).rename(
        columns={'formatted_': 'date'})
    
    # Read the original roi to find the % coincident overlap on a given date.
    roi_prefix = r.split('_')[0]
    file = f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp'
    sub_rois = gpd.read_file(file)
    roi = sub_rois[sub_rois['sub_name'] == r]
    roi_epsg = roi['utm_epsg'].iloc[0]
    est_utm_crs = roi.estimate_utm_crs()
    print(f'string epsg -- {roi_epsg} || estimated -- {est_utm_crs}')

    roi_shape = roi.geometry
    roi_shape = roi_shape.to_crs(est_utm_crs)
    roi_area = (roi_shape.geometry.iloc[0].area / 1_000_000)
    print(f'{roi_area} in square kilometers')

    # Convert to local UTM for area opperations
    ls8_footprints = ls8_footprints.to_crs(est_utm_crs)
    s2_footprints = s2_footprints.to_crs(est_utm_crs)
    merged = ls8_footprints.merge(s2_footprints,
                                  on='date',
                                  suffixes=('_ls8', '_s2'))
    
    # Merge the satellite footprints for calculations
    merged['inter'] = merged.apply(
        lambda row: row['geometry_ls8'].intersection(row['geometry_s2']),
        axis=1
    )
    merged = merged.set_geometry('inter')
    merged.crs = est_utm_crs
    
    merged['int_sqkm'] = (merged['inter'].area / 1_000_000).round(2)

    # Select the dates with best coverage in the roi
    top_dates = merged.sort_values(by='int_sqkm', 
                                   ascending=False)
    
    top_dates.drop(columns=['geometry_ls8', 'geometry_s2'], inplace=True)
    top_dates['per_cover'] = (top_dates['int_sqkm'] / roi_area * 100).round(0)
    top_dates = top_dates[top_dates['per_cover'] > 25]
    top_dates['date'] = top_dates['date'].astype(str)
    top_dates['date'] = pd.to_datetime(top_dates['date']).dt.date

    top_dates.to_crs('EPSG: 4326', inplace=True)
    # A few of the intersections yield LINESTRING geoms, filter these out for writing
    top_dates = top_dates[top_dates.geometry.type == 'Polygon']
    top_dates.to_file(f'./data/overlap_dates_for_roi/{r}_overlap_dates.shp')




# %%
