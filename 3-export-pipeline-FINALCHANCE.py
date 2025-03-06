# %% 1.0 Libraries and directories

import ee
import re
import geopandas as gpd
import pandas as pd
import pprint as pp

import datetime #??

ee.Authenticate()
ee.Initialize(project='ee-green-by-another-name')

roi_name = 'YKF_sub1'
level = 'toa'

image_footprints_path = f'./data/overlap_dates_for_roi/{roi_name}_overlap_dates.shp'
best_image_dates = gpd.read_file(image_footprints_path) 
est_utm = f'EPSG:{best_image_dates.estimate_utm_crs().to_epsg()}'
roi_prefix = roi_name.split('_')[0]
#region_shapes = gpd.read_file(f'./data/roi_shapes/rois/{roi_prefix}_sub_rois.shp')
#full_roi_shape = region_shapes[region_shapes['sub_name'] == roi_name].iloc[0]
test = best_image_dates.iloc[0]

# 2.0 %% Helper Functions for the pipeline

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



# %% Cloud mask functions 

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
    """
    Uses Sentinel-2 SCL and Cloud Probability Bands to produce a collection of cloud masks.
    Eliminates:
        1) Clouds
        2) Cloud Shaddows
        3) Snow/ICE
        4) Cirrus clouds
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

    col_size = s2_scl.size().getInfo()
    # s2_scl_list = s2_scl.toList(col_size)
    # s2_cloud_list = s2_clouds.toList(col_size)

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
    print(f'Tile fraction unmasked: {frac_unmasked_float:.2f}')

    return frac_unmasked_float

def determine_best_img(
    mask_col: ee.ImageCollection,
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

    elif satellite == 'LS8': # TODO: update these params. 
        band_name = "ls8_mask"
        scale=30
        img_id_str = "LANDSAT_PRODUCT_ID" 
    else: 
        print("Specify satellite as S2 or LS8")

    col_len = mask_col.size().getInfo()
    col_list = mask_col.toList(col_len)
    best_img_id = None
    best_img_mask = None
    highest_frac_unmasked = float(0)

    for i in range(col_len):
        img = ee.Image(col_list.get(i))
        img_id = img.get(img_id_str).getInfo()
        frac_unmasked = compute_valid_pixel_coverage(img, band_name, polygon, scale)

        if frac_unmasked > highest_frac_unmasked:
            highest_frac_unmasked = frac_unmasked
            best_img_id = img_id
            best_img_mask = img

    print(f'{satellite} Highest Frac Unmasked = {highest_frac_unmasked:.2f}')
    return best_img_mask, best_img_id, highest_frac_unmasked
        
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

    


# %% Full Pipeline

def find_pairs_and_masks(
    polygon: ee.Geometry,
    date: str, 
    date_plus1d: str,
):

    s2_mask_col = make_s2_mask_col(polygon, date, date_plus1d)
    best_s2_mask, best_s2_id, s2_unmasked_frac = determine_best_img(s2_mask_col, polygon=polygon, satellite="S2")

    ls8_mask_col = make_ls8_mask_col(polygon, date, date_plus1d)
    best_ls8_mask, best_ls8_id, ls8_unmasked_frac = determine_best_img(ls8_mask_col, polygon=polygon, satellite="LS8")

    if s2_unmasked_frac < 0.25 or ls8_unmasked_frac < 0.25: # TODO: think more about this threshold
        print(f'SKIPPING EXPORT: Bad ROI coverage with tiles')
        return None

    overlap_percentage = calculate_tile_overlap(best_s2_mask, best_ls8_mask, polygon)

    if overlap_percentage < 40:
        print("SKIPPING EXPORT: LOW TILE OVERLAP")
        return None

    return best_s2_mask, best_s2_id, best_ls8_mask, best_ls8_id


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
               .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', ls8_id))
               .select(ls8_bands)
        )

        s2_img = (ee.ImageCollection(s2_asset_string)
            .filterDate(date, date_plus1d)
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
        
        ls8_id = change_ls8_collection_num(ls8_id) 
        ls8_img = (ee.ImageCollection(ls8_asset_string)
               .filterDate(date, date_plus1d)
               .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', ls8_id))
               .select(ls8_bands)
        )
        # Fetch Sentinel-2 Image with filtering
        s2_relative_orbit_number, s2_tile_num = extract_s2_id(s2_id) 

        s2_img = (ee.ImageCollection(s2_asset_string)
                  .filterDate(date, date_plus1d)
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

# %% 

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
    )
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

    common_mask_fraction = common_mask_stats.get('combined_mask', -1)

    return common_mask_fraction


def generate_common_mask(
    s2_mask: ee.Image,
    ls8_mask: ee.Image,
    polygon: ee.Geometry,
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
    print("Check the fucking band names")
    print(s2_mask_reproj.bandNames().getInfo())
    ls8_mask_reproj = (ls8_mask.reproject(
        scale=30,
        crs=roi_est_crs)
        .resample('bilinear')
    )
    print(ls8_mask_reproj.bandNames().getInfo())

    ls8_mask_reproj = ls8_mask_reproj.rename('common_mask')
    s2_mask_reproj = s2_mask_reproj.rename('common_mask')
    print(ls8_mask_reproj.projection().getInfo())
    print(s2_mask_reproj.projection().getInfo())
    print(ls8_mask_reproj.bandNames().getInfo())
    print(s2_mask_reproj.bandNames().getInfo())

    combined_mask = s2_mask_reproj.Or(ls8_mask_reproj)
    connected_pixels = combined_mask.connectedPixelCount(maxSize=1_000, eightConnected=True)
    sieved_mask = combined_mask.updateMask(connected_pixels.gte(50))
    dilation_kernal = ee.Kernel.circle(radius=500, units='meters', normalize=False)
    dilated_mask = sieved_mask.focal_max(kernel=dilation_kernal, iterations=1)

    common_mask_fraction = calc_common_mask_frac(dilated_mask, polygon)

    return dilated_mask, common_mask_fraction

def apply_common_mask(
    img: ee.Image,
    mask: ee.Image
):
    """
    Brings the common mask into the image's projection. 
    Then, applies mask to the image
    """
    img_proj = img.projection().getInfo()
    mask_repoj = mask.reproject(
        crs=img_proj['crs'], 
        crsTransform=img_proj['transform']
    ).resample('bilinear') # Default is nearest neighbor

    masked_img = img.updateMask(mask_repoj.eq(0))

    return masked_img

# %% Export Function

def export_imgs_to_drive(
    s2_out_img: ee.Image,
    ls8_out_img: ee.Image,
    polygon: ee.Geometry, # TODO: see if this arg breaks shit
    roi_name: str,
    date: str,
    level: str,
):
    if level == 'sr':
        folder = 'sr_images'
    elif level == 'toa':
        folder = 'toa_images'
    else:
        print("ERROR: specify level as sr or toa")

    s2_export_name = f'Sentinel2_{level}_date_{date}_roi_{roi_name}'
    s2_proj = s2_out_img.projection().getInfo()
    ls8_export_name = f'Landsat8_{level}_date_{date}_roi_{roi_name}'
    ls8_proj = ls8_out_img.projection().getInfo()
    print("---- checking projections -------")
    print(s2_proj)
    print(ls8_proj)

    s2_task = ee.batch.Export.image.toDrive(
        image=s2_out_img,
        description=s2_export_name,
        fileNamePrefix=s2_export_name,
        folder='test',
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
        folder='test',
        region=polygon,
        crs=ls8_proj['crs'],
        crsTransform=ls8_proj['transform'],
        maxPixels=1e13
    )
    ls8_task.start()

    return s2_export_name, ls8_export_name


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

    pairs_and_masks = find_pairs_and_masks(polygon, date, date_plus1d)
    if pairs_and_masks is None: # Don't bother exporting bad tiles. 
        return None 
    
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

    # Make the common mask
    s2_mask = pairs_and_masks[0]
    ls8_mask = pairs_and_masks[2]
    common_mask, masked_frac = generate_common_mask(s2_mask, ls8_mask, polygon, roi_est_crs)

    print(f'The common mask covers {masked_frac:.2f}% of the region')

    # Apply the common mask to the images
    out_s2_img = apply_common_mask(s2_img, common_mask)
    out_ls8_img = apply_common_mask(ls8_img, common_mask)

    s2_export_name, ls8_export_name = export_imgs_to_drive(
        out_s2_img,
        out_ls8_img,
        polygon,
        roi_name,
        date,
        level
    )

    s2_attrs['s2_export_name'] = s2_export_name
    s2_attrs['roi_name'] = roi_name
    s2_attrs['date'] = date
    ls8_attrs['ls8_export_name'] = ls8_export_name
    ls8_attrs['roi_name'] = roi_name
    ls8_attrs['date'] = date

    return s2_attrs, ls8_attrs

# %% Run tests

s2_attrs, ls8_attrs = pair_processor(
    test,
    roi_name=roi_name,
    roi_est_crs=est_utm, 
    level=level
)


# %%

s2_attrs_list = []
ls8_attrs_list = []
# %%
