# %% 1.0 Libraries and file paths

import glob
import re
import pprint as pp
import ee.batch
import ee.data
import geopandas as gpd
import geemap
import ee
import pandas as pd

ee.Authenticate()
ee.Initialize()

roi_shape_dir = './data/roi_shapes/rois/'
s2_dir = 'projects/alpod-412314/assets'
files = glob.glob(roi_shape_dir + '*.shp')

rois_list = []
for f in files:
    gdf = gpd.read_file(f)
    rois_list.append(gdf)

rois = pd.concat(rois_list, ignore_index=True)
unique_rois = rois['sub_name'].unique()

names_to_ee = {
    'YKF': 'YKF',
    'AKCP': 'AKCP',
    'YKD': 'YKD',
    'TUK': 'TUK_MRD',
    'MRD': 'TUK_MRD',
    'AND': 'TUK_MRD'
}

# ee.data.listAssets(f'{s2_dir}/region_weekly')

# %% Auxillary Functions 

def convert_gpd_geom_to_ee(geom, est_utm):
    """
    Takes a geopandas geom object and coverts it to an Earth Engine polygon
    """
    if est_utm is None:
        out_crs = 'EPSG:4326'
    else:
        out_crs = est_utm

    coords = list(geom.exterior.coords)
    coords_list = [[x, y] for x, y in coords]

    return ee.Geometry.Polygon(coords_list, proj=out_crs)

def calc_total_roi_pixels(
    roi_ee: ee.Geometry
):
    """
    Calculate the total number of pixels in the ROI
    """
    roi_binary = (ee.Image.constant(1)
                  .clip(roi_ee)
                  .unmask()
                  .rename('roi_binary'))

    total_roi_pixels = roi_binary.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi_ee,
        scale=10,
        maxPixels=1e13
    ).get('roi_binary')

    return total_roi_pixels

def classify_lake_observations(
    img: ee.Image,
    lakes_binary: ee.FeatureCollection
):
    
    """
    From the weekly mosaic AND the original buffered lake polygons, produce a new image.
    The new image has one band with integers for all conditions.
    0 = Never observed, outside of ROI OR not a prior buffered lake
    1 = A valid observation of land on a buffered lake polygon
    2 = A valid observation of water on a buffered lake polygon
    3 = An invalid observatoin where could or ice cover obscured a lake polygon
    """
    
    # 1) Make a mask for valid water observations
    img_wtr = ee.Image.constant(0).where(img.select('water_occurance_max').eq(1), 2)
    img_wtr = img_wtr.rename('wtr')

    # 2) Make a mask of valid land observations
    img_land = ee.Image.constant(0).where(img.select('water_occurance_max').eq(0), 1)
    img_land = img_land.rename('land')

    # 3) Combine the masks with the following expression
    # 0 = No water observed OR invalid observation (cloudy, outside ALPOD)
    # 1 = Valid land observation
    # 2 = Valid water observation
    # 3 = Cloud or ice obscured lake (lakes_binary == 1, but no valid water or land observation)

    classified = lakes_binary.expression(
        """
        (wtr == 2) ? 2 :
        (land == 1) ? 1 :
        ((lakes_binary == 1) && (wtr == 0)) ? 3 :
        0
        """,
        {
            'wtr': img_wtr.select('wtr'),
            'land': img_land.select('land'),
            'lakes_binary': lakes_binary.select('lakes_binary'),
        }
    ).rename('classified')

    # 4) Tag the classified image with the mosaic_id
    mosaic_id = img.get('system:id')
    classified = classified.set('mosaic_id', mosaic_id)

    return classified

def export_timeseries(
    classified_ic: ee.ImageCollection,
    total_lake_pixels: ee.Number,
    total_roi_pixels: ee.Number,
    roi_name: str,
    roi_ee: ee.Geometry,
):
    
    """
    Takes a classified image collection and calculates the invlaid, water, and land pixels
    for each image in the collection. The results are exported to a CSV file.
    """
    
    def calc_invalid_wtr_land_pixels(
        img: ee.Image,
        roi_ee: ee.Geometry,
        total_roi_pixels: ee.Number,
        total_lake_pixels: ee.Number,
        roi_name: str
    ):
        """
        Calculate the proportion of valid area to total ALPOD area
        """
        
        invalid_img = img.eq(3)
        invalid_pixels = invalid_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_ee,
            scale=10,
            maxPixels=1e13
        ).get('classified')

        wtr_img = img.eq(2)
        wtr_pixels = wtr_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_ee,
            scale=10,
            maxPixels=1e13
        ).get('classified')

        land_img = img.eq(1)
        land_pixels = land_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_ee,
            scale=10,
            maxPixels=1e13
        ).get('classified')


        return ee.Feature(
            None,
            {
                'mosaic_id': img.get('mosaic_id'),
                'invalid_pixels': invalid_pixels,
                'water_pixels': wtr_pixels,
                'land_pixels': land_pixels,
                'total_roi_pixels': total_roi_pixels,
                'total_lake_pixels': total_lake_pixels,
                'roi_name': roi_name,
            }
        )

    roi_name_ee = ee.String(roi_name)
    calc_fc = classified_ic.map(
        lambda im: calc_invalid_wtr_land_pixels(
            img=im,
            roi_ee=roi_ee,
            total_roi_pixels=total_roi_pixels,
            total_lake_pixels=total_lake_pixels,
            roi_name=roi_name_ee
        )
    )

    task = ee.batch.Export.table.toDrive(
        collection=calc_fc,
        description=f'roi_{roi_name}_s2_timeseries',
        folder='s2_weekly_timeseries',
        fileFormat='CSV'
    )
    task.start()
    print(f'Exporting {roi_name} to Google Drive')

def make_water_fraction_timeseries(
    roi_name: str,
    rois: gpd.GeoDataFrame,
    names_to_ee: dict,
):
    
    roi = rois[rois['sub_name'] == roi_name]
    roi_main = roi_name.split('_')[0]

    roi_ee = convert_gpd_geom_to_ee(roi['geometry'].values[0], None)
    total_roi_pixels = calc_total_roi_pixels(roi_ee)
    # Becuase Earth Engine asset names are different from the shapefile names
    # Use dictionary to convert
    roi_name_ee = names_to_ee[roi_main]
    print(roi_name, roi_main, roi_name_ee)
    # Get the EE assets
    roi_lakes = ee.FeatureCollection(f'{s2_dir}/Lake_extractions/{roi_main}_extraction')
    lakes_binary = roi_lakes.reduceToImage(
        properties=['n_lakes'],
        reducer=ee.Reducer.sum(),
    ).neq(0).rename('lakes_binary').clip(roi_ee)


    asset_list = ee.data.listAssets(f'{s2_dir}/region_weekly')
    # Need to dynamically create feature collections, becuase data stored as indexed list
    weekly_mosaic_list = []
    for i in asset_list['assets']:
        asset_id = i['id']
        if re.search(roi_name_ee, asset_id): 
            weekly_mosaic_list.append(ee.Image(asset_id))
    roi_ic = ee.ImageCollection.fromImages(weekly_mosaic_list)
    clipped_roi_ic = roi_ic.map(lambda img: img.clip(roi_ee))  

    classified_ic = clipped_roi_ic.map(
        lambda img: classify_lake_observations(
            img=img,
            lakes_binary=lakes_binary,
        )
    )

    lakes_binary_pixels = lakes_binary.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi_ee,
        scale=10,
        maxPixels=1e13
    ).get('lakes_binary')


    lakes_binary_pixels = lakes_binary.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi_ee,
        scale=10,
        maxPixels=1e13
    ).get('lakes_binary')

    export_timeseries(
        classified_ic=classified_ic,
        total_lake_pixels=lakes_binary_pixels,
        total_roi_pixels=total_roi_pixels,
        roi_name=roi_name,
        roi_ee=roi_ee
    )


# %% 3.0 Run for all ROIs

for r in unique_rois:

    make_water_fraction_timeseries(
        roi_name=r,
        rois=rois,
        names_to_ee=names_to_ee,
    )








# %% Visualize some stuff


# # Grab a random, classified image to inspect
# test_list = classified_ic.toList(classified_ic.size())
# test_img = ee.Image(test_list.get(6))



# # Define a new discrete visualization dictionary.
# # 0 -> brown, 1 -> blue, 2 -> red, 3 -> grey.
# conditional_viz = {
#     'min': 0,
#     'max': 3,
#     'palette': ['grey', 'brown', 'blue', 'red']
# }

# # Create a map to display the result
# Map = geemap.Map(center=(66, -149), zoom=6)
# Map.add_basemap('SATELLITE')
# Map.addLayer(test_img, conditional_viz, 'Conditional Test')

# Map
