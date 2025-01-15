# %% 1.0 Libraries and directories
import glob
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

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
img_download_dir = './data/sr_images_downloads/*.tif'
img_list = glob.glob(img_download_dir)

river_mask_path = f'{river_dir}/{roi_name}_binary_rivers_dilated150_UTMretest.tif'


# %% 2.0 Function to mask rivers from images

def apply_river_mask(img_path: str, river_path: str, out_dir: str, band_descript: dict):
    """
    Applies the river mask to all the bands in the images.
    Any pixel where the mask is 1 will become nodata in the output.
    """
    # Only selects the file name ignores rest of path with directory
    file_name = img_path.split('images_downloads')[1] 
    out_path = out_dir + file_name

    with rio.open(img_path) as src, rio.open(river_path) as mask:

        img_meta = src.meta
        mask_meta = mask.meta
        out_meta = img_meta.copy()

        # Set no data value to -1 instead of None
        if out_meta['nodata'] is None:
            out_meta['nodata'] = -1

        # Find the common intersecting bounds for mask and image
        img_bounds = src.bounds
        mask_bounds = mask.bounds
        left = max(img_bounds.left, mask_bounds.left)
        right = min(img_bounds.right, mask_bounds.right)
        top = min(img_bounds.top, mask_bounds.top)
        bottom = max(img_bounds.bottom, mask_bounds.bottom)
        intersection_bounds = (left, bottom, right, top)

        # Read the image and mask data
        window_img = from_bounds(*intersection_bounds, src.transform)
        window_mask = from_bounds(*intersection_bounds, mask.transform)

        # Apply the mask to each band
        band_count = img_meta['count']
        print(band_count)
        for i in range(1, band_count + 1):
            img_data = src.read(i, window=window_img)

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
                    'transform': src.window_transform(window_img)
                })
            # Use the same file for subsequent bands
            else:
                mode = 'r+'

            # Apply the river mask to the data
            out_img = np.where(mask_data == 1, out_meta['nodata'], img_data)
            print(out_img.shape)

            # Write the output
            with rio.open(out_path, mode, **out_meta) as dst:
                dst.write(out_img, i)
                dst.set_band_description(i, f'{band_descript[i]}')
                
        print(out_img.shape)
        print(f'Processed {out_path}')

# %% 3.0 Apply the river masking function
for img in img_list:
    apply_river_mask(img, river_mask_path, out_dir, band_desc)
# %%
