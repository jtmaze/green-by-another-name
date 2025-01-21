# %% 1.0 Libraries and directories
import glob
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling

level = 'sr' #level should be 'sr' or 'toa'
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

        print(src.crs)
        print(mask.crs)

        img_meta = src.meta
        mask_meta = mask.meta
        out_meta = img_meta.copy()

        # Set no data value to -1 instead of None
        if out_meta['nodata'] is None:
            out_meta['nodata'] = -1

        # Find the common intersecting bounds for mask and image
        left = max(src.bounds.left, mask.bounds.left)
        right = min(src.bounds.right, mask.bounds.right)
        top = min(src.bounds.top, mask.bounds.top)
        bottom = max(src.bounds.bottom, mask.bounds.bottom)
        intersection_bounds = (left, bottom, right, top)
        # Read the image and mask data
        window_img = from_bounds(*intersection_bounds, src.transform)
        window_mask = from_bounds(*intersection_bounds, mask.transform)

        # Apply the mask to each band
        band_count = img_meta['count']
        for i in range(1, band_count + 1):
   
            img = src.read(i)
            # Without resampling, there's a slight mismatch between the image and mask dimensions. 
            mask_data = mask.read(
                1, 
                window=window_mask,
                out_shape=img.shape,
                resampling=Resampling.nearest
            )

            out_img = np.where(mask_data == 1, out_meta['nodata'], img)
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
