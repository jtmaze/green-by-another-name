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

# %%
for r in unique_rois:
    ls8_path = [f for f in footprint_file_list if re.search(fr'footprints_ls8_roi_{r}.*\.shp$', f)][0]
    s2_path = [f for f in footprint_file_list if re.search(fr'footprints_s2_roi_{r}.*\.shp$', f)][0]
    ls8_footprints = gpd.read_file(ls8_path).rename(
        columns={'formatted_': 'date'}).drop(
        columns=['label', 'count']
    )
    print(ls8_footprints.crs)

    s2_footprints = gpd.read_file(s2_path).rename(
        columns={'formatted_': 'date'}).drop(
        columns=['label', 'count']
    )
    print(s2_footprints.crs)

    merged = ls8_footprints.merge(s2_footprints,
                                  on='date',
                                  suffixes=('_ls8', '_s2'))
    
    merged['int'] = merged.apply(
        lambda row: row['geometry_ls8'].intersection(row['geometry_s2']),
        axis=1
    )
    merged = merged.set_geometry('int')
    merged.crs = ls8_footprints.crs
    

    merged['int_area_m2'] = merged['int'].area

    top_dates = merged.sort_values(by='int_area_m2', 
                                   ascending=False
                                   ).head(10)
    
    top_dates.drop(columns=['geometry_ls8', 'geometry_s2'], inplace=True)
    
    top_dates.to_file('test.shp')




# %%
