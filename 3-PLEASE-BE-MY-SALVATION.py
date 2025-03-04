# %% Libraries and directories

import ee
import geopandas as gpd
import pandas as pd
import pprint as pp

import datetime #??

ee.Authenticate()
ee.Initialize(project='ee-green-by-another-name')

roi_name = 'YKF_sub1'
resamp_method = 'bilinear'
resamp_res = 30
level = 'sr'

image_footprints_path = f'./data/overlap_dates_for_roi/{roi_name}_overlap_dates.shp'
best_image_dates = gpd.read_file(image_footprints_path) 
est_utm = f'EPSG:{best_image_dates.estimate_utm_crs().to_epsg()}'
roi_prefix = roi_name.split('_')[0]
region_shapes = gpd.read_file(f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp')
full_roi_shape = region_shapes[region_shapes['sub_name'] == roi_name].iloc[0]

# %% Functions for the pipeline

test = best_image_dates.iloc[0]

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

def print_collection_times(collection, collection_name="Collection"):
    """
    Prints the system:time_start property for each image in an EE ImageCollection
    in HH:MM format.
    
    Parameters:
      collection (ee.ImageCollection): The collection whose times are to be printed.
      collection_name (str): A label for the collection.
    """
    # Convert the collection to a list so we can iterate over it.
    image_list = collection.toList(collection.size())
    count = image_list.size().getInfo()
    
    for i in range(count):
        img = ee.Image(image_list.get(i))
        # Get the timestamp in milliseconds.
        time_ms = img.get('system:time_start').getInfo()
        # Convert milliseconds to a Python datetime object.
        dt = datetime.datetime.fromtimestamp(time_ms / 1000.0)
        mill_sec = int(time_ms % 1000)
        # Format the time as HH:MM.
        time_str = dt.strftime("%H:%M:%S") + f'{mill_sec:03d}'
        print(f"{collection_name} Image {i}: {time_str}")

# %% Cloud mask functions

def find_product_ids(
    col: ee.ImageCollection,
    satellite: str # S2 or LS8
):
    """
    Returns a list of "PRODUCT_IDs" for a given Earth Engine Collctions
    """
    if satellite == "S2":
        prod_string = "PRODUCT_ID"
    elif satellite == "LS8":
        prod_string = "LANDSAT_PRODUCT_ID"
    else:
        print("Error: please specify the satellite as S2 or LS8")

    # Use aggregate_array to extract all product IDs at once.
    product_ids = col.aggregate_array(prod_string).getInfo()
    
    return product_ids

def add_scl_join_key(img):
    """
    Turns the SCL collection's PRODUCT_ID attribute into key that can be joined 
    with the cloud collection's system:index attribute
    """
    product_id = ee.String(img.get('PRODUCT_ID'))
    parts = product_id.split('_')
    join_key = ee.String(parts.get(2)).cat('_').cat(ee.String(parts.get(5)))

    return img.set('join_key', join_key)

def add_cloud_join_key(img):
    """
    Turns the cloud collection's system:index attribute into a key that can be joined
    with the SCL collection's PRODUCT_ID attribute
    """
    sys_idx = ee.String(img.get('system:index'))
    parts = sys_idx.split('_')
    join_key = ee.String(parts.get(0)).cat('_').cat(ee.String(parts.get(2)))

    return img.set('join_key', join_key)


def make_s2_mask_col(
    polygon: ee.Geometry,
    date: str,
    date_plus1d: str,
):
    
    "Returns a collection of Binary Masks "
    
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

    sr_product_ids = find_product_ids(s2_scl, "S2")

    # Ignore this!! I'm just testing whether toa product ids can map to SR images. 
    # s2_toa = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(polygon).filterDate(date, date_plus1d))
    # toa_product_ids = find_product_ids(s2_toa, "S2")

    # print_collection_times(s2_clouds, "S2 Clouds")
    # print_collection_times(s2_scl, "S2 SCL")

    s2_scl = s2_scl.map(add_scl_join_key)
    s2_clouds = s2_clouds.map(add_cloud_join_key)

    col_size = s2_scl.size().getInfo()
    s2_scl_list = s2_scl.toList(col_size)
    s2_cloud_list = s2_clouds.toList(col_size)

    # TODO: Remove this sanity-check later
    # for i in range(col_size):
    #     print('---------------------')
    #     print('CHECKING JOIN KEYS FOR S2 MASK...')
    #     scl_img = ee.Image(s2_scl_list.get(i))
    #     scl_key = scl_img.get('join_key').getInfo()  # get join_key as a client-side string
    #     print("SCL join_key:", scl_key)
        
    #     cloud_img = ee.Image(s2_cloud_list.get(i))
    #     cloud_key = cloud_img.get('join_key').getInfo()
    #     print("Cloud join_key:", cloud_key)
    #     print('----------------------')

    join = ee.Join.inner()
    filter_join_key = ee.Filter.equals(leftField='join_key', rightField='join_key')

    joined = join.apply(s2_scl, s2_clouds, filter_join_key)

    def full_s2_mask(feature):

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
    
    s2_mask_col = joined.map(full_s2_mask)

    return s2_mask_col

# %% Landsat Cloud Masking Functions

def make_ls8_mask_col(
    polygon: ee.Geometry,
    date: str,
    date_plus1d: str
): 
    """
    Worth noting that the Landsat TOA and SR data have the same QA_Pixel band, so we can use either.
    """
    
    ls8_qa = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(polygon)
              .filterDate(date, date_plus1d)
              .select('QA_PIXEL')
    )
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

    return ls8_mask_col

# %% Functions evaluating pixel coverage and 

def compute_valid_pixel_coverage(
    mask_image: ee.Image, 
    band_name: str,
    polygon: ee.Geometry,
    scale: int
):
    
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

    #print('Total pixels in ROI:', total_roi_pixels.getInfo())
    #print('Unmasked pixels in ROI:', unmasked_pixels.getInfo())
    # print('Fraction unmasked:', frac_unmasked.getInfo())

    return frac_unmasked.getInfo()

def determine_best_img(
    mask_col: ee.ImageCollection,
    polygon: ee.Geometry,
    satellite: str
):
    if satellite == 'S2':
        band_name = 's2_mask'
        scale=10
        img_id_str = 'PRODUCT_ID'

    elif satellite == 'LS8': # TODO: update these params. 
        band_name = "ls8_mask"
        scale=30
        img_id_str = "LANDSAT_PRODUCT_ID" 
    else: 
        print("Specify Satellite")

    col_len = mask_col.size().getInfo()
    col_list = mask_col.toList(col_len)
    best_img_id = None
    best_img_mask = None
    highest_frac_unmasked = float(0)

    print(satellite)

    for i in range(col_len):
        img = ee.Image(col_list.get(i))
        img_id = img.get(img_id_str).getInfo()
        frac_unmasked = compute_valid_pixel_coverage(img, band_name, polygon, scale)

        if frac_unmasked > highest_frac_unmasked:
            highest_frac_unmasked = frac_unmasked
            best_img_id = img_id
            best_img_mask = img

    
    return (best_img_mask, best_img_id, highest_frac_unmasked) 
        
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

    return intersection_km2

    


# %% Full Pipeline

def pair_processor(
    footprint: gpd.GeoSeries,
    level: str
):
    
    date = footprint.date 
    geom = footprint.geometry

    date_plus1d = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    polygon = convert_gpd_geom_to_ee(geom, None)

    s2_mask_col = make_s2_mask_col(polygon, date, date_plus1d)
    best_s2_mask, best_s2_id, s2_unmasked_frac = determine_best_img(s2_mask_col, polygon=polygon, satellite="S2")

    ls8_mask_col = make_ls8_mask_col(polygon, date, date_plus1d)
    best_ls8_mask, best_ls8_id, ls8_unmasked_frac = determine_best_img(ls8_mask_col, polygon=polygon, satellite="LS8")

    intersection_km2 = calculate_tile_overlap(best_s2_mask, best_ls8_mask, polygon)


    return None
# %% Run tests

items = pair_processor(test, level=level)


# %%
# if level == 'toa':
#     best_ls8_id = 'LANDSAT/LC08/C02/T1_TOA/' + best_ls8_id
#     best_s2_id = "COPERNICUS/S2_HARMONIZED/" + best_s2_id
# elif level == 'sr':
#     best_ls8_id = 'LANDSAT/LC08/C02/T1_L2/' + best_ls8_id
#     best_s2_id = "COPERNICUS/S2_SR_HARMONIZED/" + best_s2_id
# else:
#     print("Error: Specify sr or toa for level")
# %%
