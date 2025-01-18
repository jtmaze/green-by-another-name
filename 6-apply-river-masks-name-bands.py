# %% 1.0 Libraries and directories
import glob
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling

level = 'toa' #level should be 'sr' or 'toa'
roi_name = 'YKF_sub1'
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

river_mask_path = f'{river_dir}/{roi_name}_binary_rivers_dilated180.tif'


# %% 2.0 Function to mask rivers from images

def apply_river_mask(img_path: str, river_path: str, out_dir: str, band_descript: dict):
    """
    Applies the river mask to all the bands in the images.
    Any pixel where the mask is 1 will become nodata in the output.
    """
    # Only selects the file name ignores rest of path with directory
    file_name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, file_name)

    with rio.open(img_path) as src, rio.open(river_path) as mask:

        img_meta = src.meta
        mask_meta = mask.meta
        out_meta = img_meta.copy()

        # Set no data value to -1 instead of None
        if out_meta['nodata'] is None:
            out_meta['nodata'] = -1

        dst_transform, width, height = calculate_default_transform(
            src.crs, 
            mask.crs,
            src.width,
            src.height,
            *src.bounds
        )

        out_meta.update({
            'crs': mask.crs,
            'transform': dst_transform,
            'width': width,
            'height': height
        })

        # Determine the bounds of image in UTM coordinates
        src_bounds_utm = transform_bounds(
            src.crs,
            mask.crs,
            src.bounds.left,
            src.bounds.bottom, 
            src.bounds.right,
            src.bounds.top
        )

        # Find the common intersecting bounds for mask and image
        left = max(src_bounds_utm[0], mask.bounds.left)
        right = min(src_bounds_utm[2], mask.bounds.right)
        top = min(src_bounds_utm[3], mask.bounds.top)
        bottom = max(src_bounds_utm[1], mask.bounds.bottom)
        intersection_bounds = (left, bottom, right, top)
        # Read the image and mask data
        #window_img = from_bounds(*intersection_bounds, src.transform)
        window_mask = from_bounds(*intersection_bounds, mask.transform)

        # Apply the mask to each band
        band_count = img_meta['count']
        for i in range(1, band_count + 1):
            # np.array for the reprojected out image. 
            reproj_img = np.empty(
                (out_meta['height'], out_meta['width']), 
                dtype=img_meta['dtype']
            )
            reproject(
                source=rio.band(src, i),
                destination=reproj_img,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=mask.crs,
                resampling=Resampling.bilinear
            )

            # Without resampling, there's a slight mismatch between the image and mask dimensions. 
            mask_data = mask.read(
                1, 
                window=window_mask,
                out_shape=reproj_img.shape,
                resampling=Resampling.nearest
            )

            out_img = np.where(mask_data == 1, out_meta['nodata'], reproj_img)
            # For first band mode should be write
            mode = 'w' if i == 1 else 'r+'
            with rio.open(out_path, mode, **out_meta) as dst:
                dst.write(out_img, i)
                dst.set_band_description(i, f'{band_descript[i]}')

        print(f'Processed {out_path}')

# %% 3.0 Apply the river masking function
for img in img_list:
    apply_river_mask(img, river_mask_path, out_dir, band_desc)


# %%
