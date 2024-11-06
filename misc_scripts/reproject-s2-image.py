# %%

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# %%

merged_path = './data/sentinel2-images/sentinel2-images-merged.tif'
merged_reprojected_path = './data/sentinel2-images/sentinel2-images-clean.tif'
target_crs = 'EPSG:4326'

with rio.open(merged_path) as src:
    src_crs = src.crs
    src_transform = src.transform
    src_width = src.width
    src_height = src.height
    src_dtype = src.dtypes[0]
    src_count = src.count
    src_nodata = src.nodata
    src_bounds = src.bounds

    out_transform, width, height = calculate_default_transform(
        src_crs=src_crs,
        dst_crs=target_crs,
        width=src_width,
        height=src_height,
        left=src_bounds.left,
        bottom=src_bounds.bottom,
        right=src_bounds.right,
        top=src_bounds.top
    )

    out_meta = src.meta.copy()
    out_meta.update({
        'crs': target_crs,
        'transform': out_transform,
        'width': width,
        'height': height
    })

    with rio.open(merged_reprojected_path, 'w', **out_meta) as dst:
        for band in range(1, src.count + 1):
            reproject(
                source=rio.band(src, band),
                destination=rio.band(dst, band),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_crs=target_crs,
                resampling=Resampling.average,
                src_nodata=src_nodata,
                dst_nodata=src_nodata
            )


# %%
