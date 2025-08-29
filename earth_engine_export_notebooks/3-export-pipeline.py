"""
This script iterates through the potential overlapping dates for a image targe area. The high-level steps are:
1. Spatially and temporally filter the Sentinel-2 and Landsat 8 image collections by the overlap footprint and date
2. Becuase the overlap footprint was generated using mosaics (i.e., multiple image tiles), we need to find the best image pair for each date.
   - The main criteria are lowest masked fraction and highest overlap area
   - First, we find the best Sentinel-2 image and its mask
   - Then, we find the Landsat 8 image with high overlap that also meets the masked fraction threshold.
3. We gernerate a common LS8 and S2 mask, which is seived (to remove noise) and dilated (for conservative masking of clouds, snow/ice, and cloud shaddows)
4. We export the images and common mask to Google Drive
5. We also save each images metadata (sun angle, cloud percetage, etc.) to a CSV file
"""

# %% 1.0 Libraries and directories

import ee
import re
import geopandas as gpd
import pandas as pd
import pprint as pp

ee.Authenticate()
ee.Initialize(project='ee-green-by-another-name')

roi_name = 'TUK_sub4'
level = 'sr'

image_footprints_path = f'./data/overlap_dates_for_roi/{roi_name}_overlap_dates.shp'
best_image_dates = gpd.read_file(image_footprints_path) 
est_utm = f'EPSG:{best_image_dates.estimate_utm_crs().to_epsg()}'
roi_prefix = roi_name.split('_')[0]
#region_shapes = gpd.read_file(f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp')
#full_roi_shape = region_shapes[region_shapes['sub_name'] == roi_name].iloc[0]

# 2.0 Functions for Earth Engine API

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

def get_mask_frac(
    mask: ee.Image,
    band_name: str,
    polygon: ee.Geometry,
    scale: int
):
    """
    Finds the proportion of masked pixels in an image. 
    """
    stats = mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        maxPixels=1e13,
        scale=scale
    ).getInfo()

    fraction = stats.get(band_name, -1)

    return fraction


def add_scl_join_key(img):
    """
    For joining Sentinel-2's Scene Classification Layer (SCL) to the Cloud Probability dataset
    Turns the SCL collection's PRODUCT_ID attribute into key that can be joined 
    with the cloud collection's system:index attribute
    """
    product_id = ee.String(img.get('PRODUCT_ID'))
    parts = product_id.split('_')
    join_key = ee.String(parts.get(2)).cat('_').cat(ee.String(parts.get(5)))

    return img.set('join_key', join_key)

def add_cloud_join_key(img):
    """
    For joining Sentinel-2's Cloud Probability dataset to the Scene Classifacation Layer
    Turns the cloud collection's system:index attribute into a key that can be joined
    with the SCL collection's PRODUCT_ID attribute
    """
    sys_idx = ee.String(img.get('system:index'))
    parts = sys_idx.split('_')
    join_key = ee.String(parts.get(0)).cat('_').cat(ee.String(parts.get(2)))

    return img.set('join_key', join_key)

def mask_full_s2_mask(feature):
    """
    Produces a comprehensive Sentinel-2 data mask removing clouds, ice, etc. 
    Requires joining the Cloud Probability Dataset to the Scene classifacation layer. 
    """
    scl_img = ee.Image(feature.get('primary'))
    cloud_img = ee.Image(feature.get('secondary'))
    
    # Create binary masks:
    # Cloud mask: probability > 70.
    cloud_mask = cloud_img.gt(70)
    # Shadow mask: SCL equals 3.
    shadow_mask = scl_img.eq(3)
    # Cirrus mask: SCL equals 10.
    cirrus_mask = scl_img.eq(10)
    # Snow/Ice mask: SCL equals 11.
    snowice_mask = scl_img.eq(11)
    
    # Combine the masks with a logical OR.
    combined_mask = cloud_mask.Or(shadow_mask).Or(cirrus_mask).Or(snowice_mask)
    combined_mask = combined_mask.rename('s2_mask')
    
    # Copy the 'PRODUCT_ID' property from the SCL image to the mask image.
    mask_with_id = combined_mask.copyProperties(scl_img, ['PRODUCT_ID'])
    
    return mask_with_id

def calc_s2_mask_stats(
    s2_scl: ee.ImageCollection,
    s2_clouds: ee.ImageCollection,
    polygon: ee.Geometry,
):
    """
    Calculates the proportion of masked Sentinel-2 pixels for each criteria. For
    example, X % snow & ice, and Y % cirrus clouds.
    """
    mask_attrs_list = []
    col_size = s2_scl.size().getInfo()
    s2_scl_list = s2_scl.toList(col_size)
    cloud_col_size = s2_clouds.size().getInfo()
    s2_cloud_list = s2_clouds.toList(col_size)

    for i in range(col_size):

        # print('CHECKING JOIN KEYS FOR S2 MASK...')
        scl_img = ee.Image(s2_scl_list.get(i))
        scl_key = scl_img.get('join_key').getInfo()  # get join_key as a client-side string
        # print("SCL join_key:", scl_key)
        shaddows = scl_img.eq(3).rename('s2_shaddows')
        shaddow_frac = get_mask_frac(shaddows, 's2_shaddows', polygon, 10)
        cirrus = scl_img.eq(10).rename('s2_cirrus')
        cirrus_frac = get_mask_frac(cirrus, 's2_cirrus', polygon, 10)
        snowice = scl_img.eq(11).rename('s2_snowice')
        snowice_frac = get_mask_frac(snowice, 's2_snowice', polygon, 10)

        cloud_img = ee.Image(s2_cloud_list.get(i))
        cloud_binary = cloud_img.gte(70).rename('s2_opaque_clouds')
        cloud_frac = get_mask_frac(cloud_binary, 's2_opaque_clouds', polygon, 10)
        cloud_key = cloud_img.get('join_key').getInfo()

        attrs = {
            's2_shaddows': shaddow_frac,
            's2_cirrus': cirrus_frac,
            's2_snowice': snowice_frac,
            's2_clouds': cloud_frac
        }
        mask_attrs_list.append(attrs)

    return mask_attrs_list

def make_s2_mask_col(
    polygon: ee.Geometry,
    date: str,
    date_plus1d: str,
):
    """
    Uses Sentinel-2 SCL band and Cloud Probability dataset to produce a collection of pixel masks.
    Eliminates: Clouds, Cloud Shaddows, Snow/Ice, cirrus clouds
    Returns: A collection of images for each tile's combined mask

    """
    
    s2_clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                 .filterBounds(polygon)
                 .filterDate(date, date_plus1d)
                 .select('probability')
    )

    s2_scl = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(polygon)
              .filterDate(date, date_plus1d)
              .select('SCL')
    )

    s2_scl = s2_scl.map(add_scl_join_key)
    s2_clouds = s2_clouds.map(add_cloud_join_key)

    mask_attrs_list = calc_s2_mask_stats(s2_scl, s2_clouds, polygon)

    join = ee.Join.inner()
    filter_join_key = ee.Filter.equals(leftField='join_key', rightField='join_key')

    joined = join.apply(s2_scl, s2_clouds, filter_join_key)
    
    s2_mask_col = joined.map(mask_full_s2_mask)

    return s2_mask_col, mask_attrs_list


def calc_ls8_mask_stats(
    ls8_qa: ee.ImageCollection,
    polygon: ee.Geometry
):
    """
    Generates a full data mask for Landsat 8 with all of the masked attributes (snow/ice, shaddows, etc.)
    """
    mask_attrs_list = []
    cloud_bit_mask = 1 << 3
    shaddow_bit_mask = 1 << 4
    snowice_bit_mask = 1 << 5
    cirrus_bit_mask = 1 << 2

    ls8_size = ls8_qa.size().getInfo()
    ls8_qa_list = ls8_qa.toList(ls8_size)

    for i in range(ls8_size):
        img = ee.Image(ls8_qa_list.get(i))
        shaddows = img.bitwiseAnd(shaddow_bit_mask).gt(0).rename('ls8_shaddows')
        shaddow_frac = get_mask_frac(shaddows, 'ls8_shaddows', polygon, 30)
        cirrus = img.bitwiseAnd(cirrus_bit_mask).gt(0).rename('ls8_cirrus')
        cirrus_frac = get_mask_frac(cirrus, 'ls8_cirrus', polygon, 30)
        snowice = img.bitwiseAnd(snowice_bit_mask).gt(0).rename('ls8_snowice')
        snowice_frac = get_mask_frac(snowice, 'ls8_snowice', polygon, 30)
        clouds = img.bitwiseAnd(cloud_bit_mask).gt(0).rename('ls8_clouds')
        cloud_frac = get_mask_frac(clouds, 'ls8_clouds', polygon, 30)

        mask_attrs = {
            'ls8_shaddows': shaddow_frac,
            'ls8_cirrus': cirrus_frac,
            'ls8_snowice': snowice_frac,
            'cloud_frac': cloud_frac
        }

        mask_attrs_list.append(mask_attrs)

    return mask_attrs_list


def make_ls8_mask_col(
    polygon: ee.Geometry,
    date: str,
    date_plus1d: str,
    level: str
): 
    """
    Generates the LS8 data mask from the QA_PIXEL band applies the mask conditions across the
    whole collection (snow/ice, shaddows, ect.)
    """
    if level == 'sr':
        asset_string = "LANDSAT/LC08/C02/T1_L2"
    elif level == 'toa':
        asset_string = "LANDSAT/LC08/C02/T1_TOA"
    
    ls8_qa = (ee.ImageCollection(asset_string)
              .filterBounds(polygon)
              .filterDate(date, date_plus1d)
              .select('QA_PIXEL')
    )

    ls8_attrs_list = calc_ls8_mask_stats(ls8_qa, polygon)
    # Define the cloud, cloud shadow, snow/ice, and cirrus bitmasks
    cloud_bit_mask = 1 << 3
    shaddow_bit_mask = 1 << 4
    snowice_bit_mask = 1 << 5
    cirrus_bit_mask = 1 << 2
    full_bitmask = (cloud_bit_mask | shaddow_bit_mask | snowice_bit_mask | cirrus_bit_mask)

    def apply_bitmask(image):
        # Compute a binary image where:
        # - 1 indicates that one or more flagged bits are set in QA_PIXEL,
        # - 0 indicates that none of the flags are set.
        flagged = image.bitwiseAnd(full_bitmask).gt(0)
        binary_mask = flagged.toInt().rename('ls8_mask')
        # Copy the LANDSAT_PRODUCT_ID property from the original image.
        return binary_mask.copyProperties(image, ['LANDSAT_PRODUCT_ID'])
    
    ls8_mask_col = ls8_qa.map(apply_bitmask)

    return ls8_mask_col, ls8_attrs_list

def compute_valid_pixel_coverage(
    mask_image: ee.Image, 
    band_name: str,
    polygon: ee.Geometry,
    scale: int
):
    """
    For a given tile in the mask collection...
    Calculates the fraction of the ROI covered by valid pixels
    """
    total_roi_pixels_dict = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13
    )
    total_roi_pixels = ee.Number(total_roi_pixels_dict.get('constant'))

    valid_img = mask_image.eq(0)
    unmasked_pixels_dict = valid_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13
    )

    unmasked_pixels = ee.Number(unmasked_pixels_dict.get(band_name))
    frac_unmasked = unmasked_pixels.divide(total_roi_pixels)
    frac_unmasked_float = frac_unmasked.getInfo()

    #print('Total pixels in ROI:', total_roi_pixels.getInfo())
    #print('Unmasked pixels in ROI:', unmasked_pixels.getInfo())
    print(f'ROI fraction covered by unmasked tile: {frac_unmasked_float:.2f}')

    return frac_unmasked_float

def determine_best_img(
    mask_col: ee.ImageCollection,
    mask_attrs_list: list, 
    polygon: ee.Geometry,
    satellite: str
):
    """
    Iterates through the valid masks for each tile in the collection. 
    Finds the image tile with best roi coverage. 
    """
    if satellite == 'S2':
        band_name = 's2_mask'
        scale=10
        img_id_str = 'PRODUCT_ID'

    elif satellite == 'LS8':  
        band_name = "ls8_mask"
        scale=30
        img_id_str = "LANDSAT_PRODUCT_ID" 
    else: 
        print("Specify satellite as S2 or LS8")

    col_len = mask_col.size().getInfo()
    col_list = mask_col.toList(col_len)

    ranked = []
    print("***********************************************")
    print(f'Satellite: {satellite} tiles={col_len} masked frations...')
    for i in range(col_len):
        img = ee.Image(col_list.get(i))
        img_id = img.get(img_id_str).getInfo()
        print("----------------------")
        frac_unmasked = compute_valid_pixel_coverage(img, band_name, polygon, scale)

        ranked.append({
            'idx': i,
            'img': img,
            'img_id': img_id,
            'frac_unmasked': frac_unmasked,
            'attrs': mask_attrs_list[i] if i < len(mask_attrs_list) else None
        })

    ranked.sort(key=lambda dct: dct["frac_unmasked"], reverse=True)
    return ranked
        
def calculate_tile_overlap(
    s2_mask: ee.Image,
    ls8_mask: ee.Image,
    polygon: ee.Geometry
):
    """
    Calculates the area of overlap between two tiles inside the roi
    NOTE: Shouldn't need to worry about projections/scale differences between images
          This is because geometry operations are applied to a consistent internal coordinate system
    """
    
    s2_geom = s2_mask.geometry()
    ls8_geom = ls8_mask.geometry()

    # First find the intersection of both image footprints
    intersection_tiles = s2_geom.intersection(ls8_geom, maxError=1)
    # Then find where this intersection overlaps with the ROI polygon
    roi_intersection = intersection_tiles.intersection(polygon, maxError=1)

    # Calculate area in square meters and kilometers
    intersection_m2 = roi_intersection.area(maxError=1)
    intersection_km2 = intersection_m2.divide(1_000_000).getInfo()

    # Also get the total ROI area for context
    roi_area_km2 = polygon.area(maxError=1).divide(1_000_000).getInfo()

    overlap_percentage = (intersection_km2 / roi_area_km2) * 100

    print(f"Tile overlap area = {intersection_km2:.2f} sqkm")
    print(f"This is {overlap_percentage}% of roi area")

    return overlap_percentage



def find_pairs_and_masks(
    polygon: ee.Geometry,
    date: str, 
    date_plus1d: str,
    level: str
):
    """
    Identifies the best Sentinel-2 and Landsat 8 image pairs and their cloud masks
    for a given region and date.
    
    Args:
        polygon: Earth Engine geometry defining the region of interest
        date: Start date for image search (YYYY-MM-DD)
        date_plus1d: End date for image search (YYYY-MM-DD)
        level: Processing level ('sr' or 'toa')
        
    Returns:
        Best S2 mask, S2 image ID, best LS8 mask, LS8 image ID, and mask attributes
        Returns None if tile overlap is insufficient
    """
    # Step 1: Find the best Sentinel-2 image and its mask
    s2_mask_col, s2_mask_attrs = make_s2_mask_col(polygon, date, date_plus1d)
    s2_ranked = determine_best_img(
        s2_mask_col, 
        s2_mask_attrs, 
        polygon=polygon, 
        satellite="S2"
    )
    best_s2 = s2_ranked[0]
    UNMASKED_THRESHOLD = 0.25
     # Check if Sentinel-2 too much cloud/mask coverage
    # (unmasked fraction < 25% means >75% of image is masked/cloudy)
    if best_s2["frac_unmasked"] < UNMASKED_THRESHOLD:
        print(f"Skiping Export: best S2 tile only {best_s2['frac_unmasked']:.2f} un-masked")
        print(f'No need to compute LS8 tile(s) masked fractions')
        return None
    
    # Step 2: Find the best Landsat 8 image and its mask
    ls8_mask_col, ls8_mask_attrs = make_ls8_mask_col(polygon, date, date_plus1d, level)
    ls8_ranked = determine_best_img(
        ls8_mask_col, 
        ls8_mask_attrs, 
        polygon=polygon, 
        satellite="LS8"
    )
    OVERLAP_THRESHOLD = 0.4
    
    for cand in ls8_ranked:
        if cand['frac_unmasked'] < UNMASKED_THRESHOLD:
            if cand['idx'] == 0:
                print(f'WARNING: best LS8 tile only {cand["frac_unmasked"]:.2f} un-masked')
                continue
            else:
                continue
        
        overlap_percentage = calculate_tile_overlap(best_s2['img'], cand['img'], polygon)
        if overlap_percentage >= 0.75:
            print("Found LS8 tile with high overlap")
            return (
                best_s2['img'], best_s2['img_id'],
                cand['img'], cand['img_id'],
                best_s2['attrs'], cand['attrs']
            )
        
        elif cand['ixd'] == ls8_ranked[-1]['idx'] and overlap_percentage < OVERLAP_THRESHOLD:
            print("No LS8 tile with overlap > 75%")
            print(f"Using best ranked LS8 image with overlap > 40%")
            first_ls8 = ls8_ranked[0]
            return (
                best_s2['img'], best_s2['img_id'],
                first_ls8['img'], first_ls8['img_id'],
                best_s2['attrs'], first_ls8['attrs']
            )
    # If we get here, no LS-8 tile met the overlap rule
    print("SKIPPING EXPORT.. No Landsat-8 candidate meets overlap and/or masked threshold")
    return None

def fetch_imgs_from_ids(
    s2_id: str,
    ls8_id: str,
    polygon: ee.Geometry,
    date: str,
    date_plus1d: str,
    level: str
):  
    """
    Finds a single specific Sentinel-2 and LandSat8 image based on PRODUCT_ID
    Returns each image
    """
 
    if level == 'sr':
        ls8_asset_string = 'LANDSAT/LC08/C02/T1_L2'
        s2_asset_string = 'COPERNICUS/S2_SR_HARMONIZED'
        ls8_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']
        s2_bands = ['B2', 'B3', 'B4', 'B8']

        ls8_img = (ee.ImageCollection(ls8_asset_string)
               .filterDate(date, date_plus1d)
               .filterBounds(polygon)
               .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', ls8_id))
               .select(ls8_bands)
        )

        s2_img = (ee.ImageCollection(s2_asset_string)
            .filterDate(date, date_plus1d)
            .filterBounds(polygon)
            .filter(ee.Filter.eq('PRODUCT_ID', s2_id))
            .select(s2_bands)
        )

    elif level == 'toa':
        ls8_asset_string = 'LANDSAT/LC08/C02/T1_TOA'
        s2_asset_string = 'COPERNICUS/S2_HARMONIZED'
        ls8_bands = ['B2', 'B3', 'B4', 'B5']
        s2_bands = ['B2', 'B3', 'B4', 'B8']

        # NOTE the PRODUCT_IDs were fetched on SR Images. 
        # This means we need to modify strings to fetch toa images. 
        def change_ls8_collection_num(id):
            parts = id.split('_')
            if parts[1] == 'L2SP': 
                parts[1] = 'L1TP'
            else:
                print("WARNING UNSUSPECTED STRING PATTERN IN LS8 ID")
                return None
            return '_'.join(parts)
        
        def extract_s2_id(s2_id):
            parts = s2_id.split('_')
            # Extract relative orbit number from product_id string
            relative_orbit = parts[4]
            relative_orbit_number = str(re.search(r"R(\d{3})", relative_orbit).group(1))
            if relative_orbit_number[0] == 0: #begins with zero
                relative_orbit_number = relative_orbit_number[-2:]

            # Extract the MGRS tile from product_id string
            tile_num = parts[5]
            tile = re.search(r"T([A-Z0-9]+)", tile_num).group(1) # Any uppercase characters and digits following T
            return int(relative_orbit_number), str(tile)
        
        #ls8_id = change_ls8_collection_num(ls8_id) 
        ls8_img = (ee.ImageCollection(ls8_asset_string)
               .filterDate(date, date_plus1d)
               .filterBounds(polygon)
               .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', ls8_id))
               .select(ls8_bands)
        )
        # Fetch Sentinel-2 Image with filtering
        s2_relative_orbit_number, s2_tile_num = extract_s2_id(s2_id) 

        s2_img = (ee.ImageCollection(s2_asset_string)
                  .filterDate(date, date_plus1d)
                  .filterBounds(polygon)
                  .filter(ee.Filter.eq('MGRS_TILE', s2_tile_num))
                  .filter(ee.Filter.eq('SENSING_ORBIT_NUMBER', s2_relative_orbit_number))
                  .select(s2_bands)
        )
    else:
        print("ERROR: Specify level as sr or toa")


    # Check that we found only 1 exact image. 
    s2_size = s2_img.size().getInfo()
    ls8_size = ls8_img.size().getInfo()
    print(f"LS8 size = {ls8_size}")
    print(f"S2 size = {s2_size}")

    if ls8_size != 1 and s2_size != 1:
        print("ERROR in fetch_imgs_from_ids, collection size != 1")
        return None, None

    return s2_img.first(), ls8_img.first()

def rescale_imgs(
        img: ee.Image, 
        satellite: str,
        level: str
):

    if satellite == "LS8" and level == "toa":
        rescaled = img
    elif satellite == "LS8" and level == "sr":
        rescaled = img.multiply(0.0000275).add(-0.2)
    elif satellite == "S2":
        rescaled = img.divide(10_000)
    else:
        print("ERROR in rescale_imgs arguments")

    return rescaled

def find_ls8_img_attrs(
    ls8_img: ee.Image,
    level: str
):
    
    "Finds specific attributes for a give LandSat8 image"
    
    if level == 'toa':
        attrs_list = [
            "LANDSAT_PRODUCT_ID",
            "ORIENTATION",
            "ROLL_ANGLE",
            "SCENE_CENTER_TIME",
            "SUN_AZIMUTH",
            "SUN_ELEVATION"
        ]

    elif level == "sr":
        attrs_list = [
            "ALGORITHM_SOURCE_SURFACE_REFLECTANCE",
            "LANDSAT_PRODUCT_ID",
            "ORIENTATION", # This may not be avialable for TOA data.
            "ROLL_ANGLE",
            "SCENE_CENTER_TIME",
            "SUN_AZIMUTH", 
            "SUN_ELEVATION"
        ]
    else:
        print("Invalid level")

    all_attrs = ls8_img.toDictionary().getInfo()
    attrs_out = {}
    for attr_name in attrs_list:
        if attr_name in all_attrs:
            attrs_out[attr_name] = all_attrs[attr_name]
        else:
            attrs_out[attr_name] = None

    return attrs_out

def find_s2_img_attrs(
    s2_img: ee.Image,
    level: str
):
    """
    Finds the specific attributes for a given Sentinel-2 images
    """
    # NOTE: Because Sentinel-2 is a push-broom sensor, each band could have slightly different mean
    # incidence angles. For our purposes, let's just use NIR and GREEN.
    if level == 'toa':
        attrs_list = [
            "GENERATION_TIME",
            "PRODUCT_ID",
            "MEAN_INCIDENCE_AZIMUTH_ANGLE_B3",
            "MEAN_INCIDENCE_ZENITH_ANGLE_B3",
            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8',
            "MEAN_INCIDENCE_ZENITH_ANGLE_B8",
            "SOLAR_IRRADIANCE_B3",
            "SOLAR_IRRADIANCE_B8",
            "MEAN_SOLAR_AZIMUTH_ANGLE",
            "MEAN_SOLAR_ZENITH_ANGLE",
            "SPACECRAFT_NAME",
            "SENSING_ORBIT_DIRECTION"
        ]

    elif level == 'sr':
        attrs_list = [
            "GENERATION_TIME",
            "PRODUCT_ID",
            "MEAN_INCIDENCE_AZIMUTH_ANGLE_B3",
            "MEAN_INCIDENCE_ZENITH_ANGLE_B3",
            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8',
            "MEAN_INCIDENCE_ZENITH_ANGLE_B8",
            "SOLAR_IRRADIANCE_B3",
            "SOLAR_IRRADIANCE_B8",
            "MEAN_SOLAR_AZIMUTH_ANGLE",
            "MEAN_SOLAR_ZENITH_ANGLE",
            "SPACECRAFT_NAME",
            "SENSING_ORBIT_DIRECTION",
            "AOT_RETRIVAL_ACCURACY",
            "RADIATIVE_TRANSFER_ACCURACY",
            "WATER_VAPOUR_RETRIEVAL_ACCURACY"
        ]
        
    else:
        print("Invalid level")
    
    all_attrs = s2_img.toDictionary().getInfo()
    attrs_out = {}
    for attr_name in attrs_list:
        if attr_name in all_attrs:
            attrs_out[attr_name] = all_attrs[attr_name]
        else:
            attrs_out[attr_name] = None

    return attrs_out



def calc_common_mask_frac(
    common_mask: ee.Image,
    polygon: ee.Geometry
):
    """
    Need to reproject the common mask from UTM to EPSG:4326 to find masked fraction in image footprint
    Converting the image footprint to UTM causes some geometry errors in Earth Engine
    """
    common_mask_reproj = common_mask.reproject(
        crs="EPSG:4326",
        scale=30
    ).resample('bilinear')
    # Get the number of pixels for the dilated mask
    # NOTE: Due to timeout errors, I had to adjust a few arguments sacrificing precision
    common_mask_stats = common_mask_reproj.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        maxPixels=1e12,
        bestEffort=True,
        scale=30,
        #tileScale=2 # Adjust this if time-out errors persist. Splits work into smaller chunks. 
        # The downside is it increases overall workload due to splitting load into smaller chunks.
    ).getInfo()

    common_mask_fraction = (common_mask_stats.get('common_mask', -1) * 100)

    return common_mask_fraction

def export_undilated_common_mask(
    mask: ee.Image,
    polygon: ee.Geometry,
    date: str,
    roi_name:str
):  
    export_name = f'CommonMask_RAW_date_{date}_roi_{roi_name}'
    # mask_proj = mask.projection().getInfo()
    # task = ee.batch.Export.image.toDrive(
    #     image=mask,
    #     description=export_name,
    #     fileNamePrefix=export_name,
    #     folder='raw_masks',
    #     region=polygon,
    #     crs=mask_proj['crs'],
    #     crsTransform=mask_proj['transform'],
    #     maxPixels=1e13
    # )
    # task.start()
    # print("EXPORTING raw common mask")
    pass


def generate_common_mask(
    s2_mask: ee.Image,
    ls8_mask: ee.Image,
    polygon: ee.Geometry,
    date: str,
    roi_name: str,
    roi_est_crs: str # The EPSG code assocaited with the Region's UTM zone
):
    """
    Generates a common mask
    """
    
    s2_mask_reproj = (s2_mask.reproject(
        scale=30,
        crs=roi_est_crs)
        .resample('bilinear')
    )

    ls8_mask_reproj = (ls8_mask.reproject(
        scale=30,
        crs=roi_est_crs)
        .resample('bilinear')
    )

    ls8_mask_reproj = ls8_mask_reproj.rename('common_mask')
    s2_mask_reproj = s2_mask_reproj.rename('common_mask')

    combined_mask = s2_mask_reproj.Or(ls8_mask_reproj)

    # I don't trust Google Earth Engine anymore (want to double check sieving and dilating)
    export_undilated_common_mask(
        combined_mask,
        polygon,
        date,
        roi_name
    )

    connected_pixels = combined_mask.connectedPixelCount(maxSize=1_000, eightConnected=True)
    sieved_mask = combined_mask.updateMask(connected_pixels.gte(50))
    dilation_kernal = ee.Kernel.circle(radius=500, units='meters', normalize=False)
    dilated_mask = sieved_mask.focal_max(kernel=dilation_kernal, iterations=1)

    common_mask_fraction = calc_common_mask_frac(dilated_mask, polygon)

    return dilated_mask, common_mask_fraction

# def apply_common_mask(
#     img: ee.Image,
#     mask: ee.Image
# ):
#     """
#     Brings the common mask into the image's projection. 
#     Then, applies mask to the image
#     """
#     img_proj = img.projection().getInfo()
#     mask_repoj = mask.reproject(
#         crs=img_proj['crs'], 
#         crsTransform=img_proj['transform']
#     ).resample('bilinear') # Default is nearest neighbor

#     masked_img = img.updateMask(mask_repoj.eq(0))

#     return masked_img



def export_imgs_to_drive(
    s2_out_img: ee.Image,
    ls8_out_img: ee.Image,
    common_mask: ee.Image,
    polygon: ee.Geometry, 
    roi_name: str,
    date: str,
    level: str,
):
    if level == 'sr':
        folder = 'sr_images'
        mask_folder = 'sr_masks'
    elif level == 'toa':
        folder = 'toa_images'
        mask_folder = 'toa_masks'
    else:
        print("ERROR: specify level as sr or toa")

    s2_export_name = f'Sentinel2_{level}_date_{date}_roi_{roi_name}'
    s2_proj = s2_out_img.projection().getInfo()
    ls8_export_name = f'Landsat8_{level}_date_{date}_roi_{roi_name}'
    ls8_proj = ls8_out_img.projection().getInfo()
    mask_export_name = f'CommonMask_date_{date}_roi_{roi_name}'
    mask_proj = common_mask.projection().getInfo()

    s2_task = ee.batch.Export.image.toDrive(
        image=s2_out_img,
        description=s2_export_name,
        fileNamePrefix=s2_export_name,
        folder=folder,
        region=polygon,
        crs=s2_proj['crs'],
        crsTransform=s2_proj['transform'],
        maxPixels=1e13
    )
    s2_task.start()
    
    ls8_task = ee.batch.Export.image.toDrive(
        image=ls8_out_img,
        description=ls8_export_name,
        fileNamePrefix=ls8_export_name,
        folder=folder,
        region=polygon,
        crs=ls8_proj['crs'],
        crsTransform=ls8_proj['transform'],
        maxPixels=1e13
    )
    ls8_task.start()

    mask_task = ee.batch.Export.image.toDrive(
        image=common_mask,
        description=mask_export_name,
        fileNamePrefix=mask_export_name,
        folder=mask_folder,
        region=polygon,
        crs=mask_proj['crs'],
        crsTransform=mask_proj['transform'],
        maxPixels=1e13
    )
    mask_task.start()

    return s2_export_name, ls8_export_name, mask_export_name


# %% Full Function

def pair_processor(
    footprint: gpd.GeoSeries,
    level: str,
    roi_name: str,
    roi_est_crs: str
):
    
    date = footprint.date 
    geom = footprint.geometry

    date_plus1d = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    polygon = convert_gpd_geom_to_ee(geom, None)
    print("=======================================")
    print(f"Processing {roi_name} for date {date}")
    print("=======================================")
###############################################################################
# 1.0 Find the optimal image pair (i.e. lowest masked portion and high overlap)
##############################################################################

    pairs_and_masks = find_pairs_and_masks(polygon, date, date_plus1d, level)
    if pairs_and_masks is None: # Don't bother exporting non-overlapping tiles
        return None 

######################################################################
# 2.0 Fetch the best image pair and their attributes
######################################################################

    s2_id =pairs_and_masks[1]
    ls8_id = pairs_and_masks[3]

    s2_img, ls8_img = fetch_imgs_from_ids(
        s2_id=s2_id, 
        ls8_id=ls8_id, 
        polygon=polygon,
        date=date,
        date_plus1d=date_plus1d,
        level=level
    )

    s2_attrs = find_s2_img_attrs(s2_img, level)
    ls8_attrs = find_ls8_img_attrs(ls8_img, level)
    s2_img = rescale_imgs(s2_img, satellite="S2", level=level)
    ls8_img = rescale_imgs(ls8_img, satellite="LS8", level=level)


#########################################################################
# Produce the common mask
#########################################################################
    s2_mask = pairs_and_masks[0]
    ls8_mask = pairs_and_masks[2]
    common_mask, masked_frac = generate_common_mask(
        s2_mask, 
        ls8_mask, 
        polygon, 
        date,
        roi_name,
        roi_est_crs
    )

#########################################################################
# Check the common mask's fraction, Export images and attributes
#########################################################################

    print(f'The common mask covers {masked_frac:.2f}% of the region')
    if masked_frac > 55:
        print("Bad image -- skipping export")
        mask_export_name = 'bad_img'
        s2_export_name = 'bad_img'
        ls8_export_name = 'bad_img'

    else:
        s2_export_name, ls8_export_name, mask_export_name = export_imgs_to_drive(
            s2_img,
            ls8_img,
            common_mask,
            polygon,
            roi_name,
            date,
            level
        )
    print(".................")
    print("Sentinel-2 Mask Attributes")
    pp.pp(pairs_and_masks[4])
    print("Landsat-8 Mask Attributes")
    pp.pp(pairs_and_masks[5])
    print(".................")

    mask_attrs = pairs_and_masks[4] | pairs_and_masks[5]
    mask_attrs['s2_export_name'] = s2_export_name
    mask_attrs['ls8_export_name'] = ls8_export_name
    mask_attrs['mask_export_name'] = mask_export_name
    mask_attrs['roi_name'] = roi_name
    mask_attrs['date'] = date

    s2_attrs['s2_export_name'] = s2_export_name
    s2_attrs['mask_export_name'] = mask_export_name
    s2_attrs['roi_name'] = roi_name
    s2_attrs['date'] = date

    ls8_attrs['ls8_export_name'] = ls8_export_name
    ls8_attrs['mask_export_name'] = mask_export_name
    ls8_attrs['roi_name'] = roi_name
    ls8_attrs['date'] = date

    return s2_attrs, ls8_attrs, mask_attrs

# %% Run the script

s2_attrs_list = []
ls8_attrs_list = []
mask_attrs_list = []

for idx, row in best_image_dates.iterrows():

    result = pair_processor(
        row,
        roi_name=roi_name,
        roi_est_crs=est_utm, 
        level=level
    )
    if result is not None:
        s2_attrs, ls8_attrs, mask_attrs = result
        mask_attrs_list.append(mask_attrs)
        s2_attrs_list.append(s2_attrs)
        ls8_attrs_list.append(ls8_attrs)
    else:
        print(f"Skipping row {idx} - no valid image pair found")

# %% Write output to a dataframe
mask_batch_summary = pd.DataFrame(mask_attrs_list)
mask_batch_summary.to_csv(
    f'./data/img_mask_solar_stats/{roi_name}_{level}_mask_attrs.csv',
    index=False
)
s2_batch_summary = pd.DataFrame(s2_attrs_list)
s2_batch_summary.to_csv(
    f'./data/img_mask_solar_stats/{roi_name}_{level}_Sentinel2_attrs.csv',
    index=False
)
ls8_batch_summary = pd.DataFrame(ls8_attrs_list)
ls8_batch_summary.to_csv(
    f'./data/img_mask_solar_stats/{roi_name}_{level}_Landsat8_attrs.csv',
    index=False
)

# %%
