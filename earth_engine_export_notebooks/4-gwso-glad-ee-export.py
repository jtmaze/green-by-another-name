# %% 1.0 Libraries and file paths 
import os
import glob
import time
import pprint as pp

import ee
import pandas as pd
import geopandas as gpd

ee.Authenticate()
ee.Initialize()

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')
roi_shape_dir = './data/roi_shapes/rois'
files = glob.glob(f'{roi_shape_dir}/*.shp')
rois_list = []
for f in files:
    gdf = gpd.read_file(f)
    rois_list.append(gdf)

rois = pd.concat(rois_list, ignore_index=True)
unique_rois = rois['sub_name'].unique()


# %% 2.0 Combine all the GLAD observations for June & August for 20 years

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

def select_invalid(img: ee.Image, mask_val: int):
    """
    For GSWO nodata = 0
    For GLAD nodata = 255
    Finds the invalid pixels and returns a binary image with..
    1 = invalid
    0 = valid
    """
    img_unmasked = img.unmask(mask_val)
    inval_pixels = img_unmasked.eq(mask_val)
    inval_pixels = inval_pixels.updateMask(ee.Image.constant(1))

    return inval_pixels

def sepperate_mask_from_data(
    img: ee.Image,
    mask_val: int
):
    """
    For GSWO nodata = 0
    For GLAD nodata = 255
    Prepares a new image where only valid pixels are retained
    """
    data_binary = img.neq(mask_val)
    data = img.updateMask(data_binary)

    return data

def select_water_pixels(
    img: ee.Image,
    water_val: int
):
    """
    Only used for GSWO, becuase water values are equal == 2
    This makes binary mask (water == 1) for summing water occurence
    """
    water_pixels = img.eq(water_val)
    return water_pixels

def calc_valid_occurence(
        img_collection: ee.ImageCollection,
        max_obs: int,
        mask_val: int,
        dataset: str
):
    """
    Creates a new image with the sum of valid water observations across years
    """
    # Make an image with the sum of invalid observations across years
    inval_masks = img_collection.map(lambda img: select_invalid(img, mask_val))
    inval_sum = inval_masks.reduce(ee.Reducer.sum())

    # Sum the valid water observations in each image. 
    data = img_collection.map(lambda img: sepperate_mask_from_data(img, mask_val))
    if dataset == 'GSWO': # GSWO has water values = 2
        data = data.map(lambda img: select_water_pixels(img, water_val=2))
    
    sum_water = data.reduce(ee.Reducer.sum())

    # Calculate the % of valid occurance
    max_obs_img = ee.Image.constant(max_obs)
    total_valid_img = max_obs_img.subtract(inval_sum)
    valid_occurence = sum_water.divide(total_valid_img)

    return valid_occurence

def export_dataset(
    image: ee.Image,
    roi_name: str,
    dataset_name: str,
    month: str,
    polygon: ee.Geometry
):
    export_name = f'dataset_{dataset_name}_month_{month}_roi_{roi_name}'
    folder = f'global_datasets'
    print(f"batching {export_name}")

    clipped_image = image.clip(polygon)
    export_task = ee.batch.Export.image.toDrive(
        image=clipped_image,
        description=export_name,
        fileNamePrefix=export_name,
        folder=folder,
        scale=30,
        crs='EPSG:4326',
        region=polygon,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )

    export_task.start()

def gswo_glad_pipeline(
    roi_name: str,
    rois: gpd.GeoDataFrame,
    years: tuple
):
    
    roi = rois[rois['sub_name'] == roi_name]
    years_range = range(years[0], years[1])
    roi_ee = convert_gpd_geom_to_ee(roi['geometry'].values[0], None)

    june_sys_idx = [f'{year}_06' for year in years_range]
    aug_sys_idx = [f'{year}_08' for year in years_range]

    # ------------- GLAD Export ----------------- #
    glad = ee.ImageCollection('projects/glad/water/individualMonths')

    june_imgs = glad.filter(ee.Filter.inList('system:index', june_sys_idx))
    june_max_obs = june_imgs.size().getInfo()

    aug_imgs = glad.filter(ee.Filter.inList('system:index', aug_sys_idx))
    aug_max_obs = aug_imgs.size().getInfo()

    glad_monthly_datasets = [
        (june_imgs, june_max_obs, 'june'),
        (aug_imgs, aug_max_obs, 'aug')
    ]

    for ds in glad_monthly_datasets:
        valid_occurence = calc_valid_occurence(ds[0], ds[1], mask_val=255, dataset='GLAD')
        month_name = ds[2]
        export_dataset(
            image=valid_occurence,
            roi_name=roi_name,
            dataset_name='GLAD',
            month=month_name,
            polygon=roi_ee
        )

    # ------------- GSWO Export ----------------- #
    gswo = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')

    gswo_june_imgs = gswo.filter(ee.Filter.inList('system:index', june_sys_idx))
    gswo_june_max_obs = gswo_june_imgs.size().getInfo()
    gswo_aug_imgs = gswo.filter(ee.Filter.inList('system:index', aug_sys_idx))
    gswo_aug_max_obs = gswo_aug_imgs.size().getInfo()


    gswo_monthly_datasets = [
        (gswo_june_imgs, gswo_june_max_obs, 'june'),
        (gswo_aug_imgs, gswo_aug_max_obs, 'aug')
    ]

    for ds in gswo_monthly_datasets:
        valid_occurence = calc_valid_occurence(ds[0], ds[1], mask_val=0, dataset='GSWO')
        month_name = ds[2]
        export_dataset(
            image=valid_occurence,
            roi_name=roi_name,
            dataset_name='GSWO',
            month=month_name,
            polygon=roi_ee
        )

# %%

for r in unique_rois:

    gswo_glad_pipeline(
        roi_name=r,
        rois=rois,
        years=(1999, 2021)
    )

# %% 

