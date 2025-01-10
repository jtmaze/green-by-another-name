# %% 1.0 Libraries and directories
import glob
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

out_dir = './data/image_downloads_masked/'
img_dir = './data/image_downloads/*.tif'
img_list = glob.glob(img_dir)
roi_name = 'YKF_sub1'
roi_path = f'./data/roi_shapes/{roi_name}_shape.shp'
roi = gpd.read_file(roi_path)
est_utm = roi.estimate_utm_crs()
print(est_utm)

river_path = f'./data/river_files/{roi_name}_binary_rivers_dilated150_UTMretest.tif'

# %% 2.0 Function to mask rivers from images

def reproj_img_to_utm(src: rio.DatasetReader, utm_crs: str):
    transform, width, height = calculate_default_transform(
        src.crs,
        utm_crs,
        src.width,
        src.height,
        *src.bounds
    )
    out_array = np.empty((src.count, height, width), dtype=src.dtype)
    reproject(
        source=src.read(),
        destination=out_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=utm_crs,
        resampling=Resampling.nearest
    )

def apply_river_mask(img_path: str, river_path: str, out_dir: str):
    """
    Applies the river mask to all the bands in the images.
    Any pixel where the mask is 1 will become nodata in the output.
    """
    file_name = img_path.split('image_downloads')[1]
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

        band_count = img_meta['count']
        for i in range(1, band_count + 1):
            img_data = src.read(i, window=window_img)
            mask_data = mask.read(1, window=window_mask)

            if i == 1:
                mode = 'w'
                out_meta.update({
                    'height': img_data.shape[0],
                    'width': img_data.shape[1],
                    'transform': src.window_transform(window_img)
                })
            else:
                mode = 'r+'

            # Apply the mask
            out_img = np.where(mask_data == 1, out_meta['nodata'], img_data)

            # Write the output
            with rio.open(out_path, mode, **out_meta) as dst:
                dst.write(out_img, i)
                dst.set_band_description(i, f'Band {i}')

        print(f'Processed {out_path}')

# %% 3.0 Apply the river masking function

for img in img_list:
    apply_river_mask(img, river_path, out_dir)