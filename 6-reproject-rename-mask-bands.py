# %% 1.0 Libraries and directories

# TODO: Some images are entirely nan 
# (issue with ee cloud masks). Don't write these images

import glob
import os
import re
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling

level = 'toa' #level should be 'sr' or 'toa'
roi_name = 'TUK_sub1'
utm_epsg = 'EPSG:32609' 

"""
EPSG Codes for Sites:
YKF_sub1: EPSG:32606
AKCP_sub1: EPSG:32605
YKD_sub1: EPSG:32603
MRD_sub1: EPSG: 32608
AKCP_sub2: EPSG: 32604
TUK_sub1: EPSG: 32609
"""
# Dictionary to add description to bands based on the
band_desc = {
    1: 'Blue',
    2: 'Green',
    3: 'Red',
    4: 'NIR'
}

out_dir = f'./data/{level}_images/'
river_dir = './data/river_files/'
img_download_dir = f'./data/{level}_image_downloads/*.tif'
img_list = glob.glob(img_download_dir)
roi_img_list = [img for img in img_list if re.search(roi_name, img)]

river_mask_path = f'{river_dir}/{roi_name}_binary_rivers_dilated180.tif'
print(img_list)
print(roi_img_list)

# %% 2.0 Function to mask rivers from images

def reproject_img(img_path: str, utm_epsg: str):
    """
    Reprojects the google drive downloads from EPSG 4326 into local UTM
    This matches to the image to river mask
    Reprojected images are written to 'temp' folder, and deleted after masking
    """
    # Only selects the file name ignores rest of path with directory
    file_name = os.path.basename(img_path)
    out_path = os.path.join('./data/temp/', file_name)
    dst_crs = utm_epsg

    with rio.open(img_path) as src:
        out_meta = src.meta.copy()
        transform, width, height = calculate_default_transform(
            src.crs, #src_crs
            dst_crs, #dst_crs
            src.width, #width
            src.height, #height
            *src.bounds
        )
        out_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                # Reproject each band to the new coordinate system
                reproject(
                    source=band,
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
            

        print(f"Reprojected {file_name} to UTM")

def apply_river_mask(img_path: str, river_path: str, out_dir: str, band_descript: dict):
    """
    Reads the image data and mask with a common intersecting window.
    This common window ensures identical shapes for numpy.where()
    band_descript adds band descriptions (e.g. "NIR") to match the band index. 
    """
    file_name = os.path.basename(img_path)
    in_path = os.path.join('./data/temp/', file_name)
    out_path = os.path.join(out_dir, file_name)

    with rio.open(in_path) as img, rio.open(river_path) as mask:

        img_meta = img.meta
        mask_meta = mask.meta
        out_meta = img_meta.copy()

        img_bounds = img.bounds
        mask_bounds = mask.bounds
        left = max(img_bounds.left, mask_bounds.left)
        right = min(img_bounds.right, mask_bounds.right)
        top = min(img_bounds.top, mask_bounds.top)
        bottom = max(img_bounds.bottom, mask_bounds.bottom)
        intersection_bounds = (left, bottom, right, top) 

        if out_meta['nodata'] is None:
            out_meta['nodata'] = -1

        window_img = from_bounds(*intersection_bounds, img.transform)
        window_mask = from_bounds(*intersection_bounds, mask.transform)

        band_count = img_meta['count']
        for i in range(1, band_count + 1):
            img_data = img.read(i, window=window_img)
            
            # Without resampling, there's a slight mismatch between the image and mask dimensions. 
            mask_data = mask.read(
                1,
                window=window_mask,
                out_shape=img_data.shape,
                resampling=Resampling.nearest
            )

            # For first band mode should be write
            if i == 1:
                mode = 'w'
                out_meta.update({
                    'height': img_data.shape[0],
                    'width': img_data.shape[1],
                    'transform': img.window_transform(window_img)
                })
            # Use the same file for subsequent bands
            else:
                mode = 'r+'

            # Apply the river mask to the data
            out_img = np.where(mask_data == 1, out_meta['nodata'], img_data)

            # Write the output
            with rio.open(out_path, mode, **out_meta) as dst:
                dst.write(out_img, i)
                dst.set_band_description(i, f'{band_descript[i]}')

        print(f'Processed {out_path}')

def clean_temp_folder(img):
        """
        Once the masked image is written, clean out the temp folder
        Prevents storage from getting bogged down
        """
        file_name = os.path.basename(img)
        temp_path = os.path.join('./data/temp/', file_name)
        print(f"Deleting temporary file {temp_path}")
        os.remove(temp_path)


# %% 3.0 Apply the river masking function

for img in roi_img_list:
    reproject_img(img, utm_epsg)
    apply_river_mask(img, river_mask_path, out_dir, band_desc)
    clean_temp_folder(img)


# %%
