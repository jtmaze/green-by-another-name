# %% 1.0 Libraries
import pprint as pp

import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize

# %% 2.0 

pld = gpd.read_file('./data/pld_shapes/YKflats_pld_clipped.shp')
boundary = gpd.read_file('./data/roi_and_img_bounds/ls-boundary.shp')
est_utm = pld.estimate_utm_crs()
est_utm_boudary = boundary.estimate_utm_crs()

"""
!!! Worth adding check to ensure est_utm is same for lakes and boundary
"""
print(est_utm, est_utm_boudary)

pld_utm = pld.to_crs(est_utm)
boundary_utm = boundary.to_crs(est_utm) 

# %% 3.0 Define functions

def pld_buffer_img_clip(gdf_utm, buffer_size, clip_bounds):
    """Returns a tupple of new lakes and a band name for writing to raster"""
    # Clip the lakes to the common image boundary
    out_gdf = gdf_utm.copy()
    out_gdf = gpd.clip(out_gdf, clip_bounds)

    # Buffer the lakes
    out_gdf = out_gdf.buffer(buffer_size)

    # Calculate number of lakes removed by errosion
    total_lakes = len(out_gdf)
    out_gdf = out_gdf[~out_gdf.is_empty]
    out_lakes = len(out_gdf)
    print(f'{total_lakes - out_lakes} of {total_lakes} lakes were removed due to buffer size')

    band_name = f'buffered_{buffer_size}m'

    return (out_gdf, band_name)


def rasterize_buffers(gdf, band_name, common_mask_path, band_number, mode):
    """
    Rasterize the dilated/eroded shapes and write them to disk
    """

    with rio.open(common_mask_path) as cmask_src:
        cmask_meta = cmask_src.meta

    out_meta = cmask_meta.copy()
    if mode == 'w':
        out_meta.update({
            'count': len(buffers)  # Ensure all bands are accounted for
        })

    # Convert the lakes the common mask crs (EPSG:4326)
    gdf = gdf.to_crs(cmask_meta['crs'])

    lake_raster = rasterize(
        shapes=[(geom, 1) for geom in gdf.geometry],
        out_shape=(cmask_meta['height'], cmask_meta['width']),
        transform=cmask_meta['transform'],
        all_touched=True,
        fill=0,
        default_value=1,
        dtype=rio.uint8 
    )

    with rio.open(f'./data/pld_shapes/pld_masks.tif', mode, **out_meta) as dst:
        dst.write(lake_raster, band_number)
        dst.set_band_description(band_number, band_name)

    print(f'{band_name} rasterized')
    

# %% Rasterize the new shapes as bands

buffers = [-60, -30, 0, 30, 60, 120]

for i, buffer in enumerate(buffers):
    pld_buffered, band_name = pld_buffer_img_clip(pld_utm, buffer, boundary_utm)
    mode = 'w' if i == 0 else 'r+'
    rasterize_buffers(pld_buffered, band_name, './data/common_mask_v2.tif', i+1, mode)

# %%
