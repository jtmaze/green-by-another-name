# %% 1.0 Libaries and Directories
import pprint as pp
import rasterio as rio
import numpy as np

# %% 2.0 Subtract green band and write file

masked_green_path = './data/masked_images/green-masked.tif'

with rio.open(masked_green_path) as src:
    ls_green = src.read(1)
    ls_green = ls_green * 10_000
    s2_green = src.read(2)
    pp.pp(src.meta)

    green_diff = ls_green - s2_green
    green_diff = np.where((green_diff < 500) & (green_diff > -500), green_diff, np.nan)

    out_meta = src.meta.copy()
    out_meta.update(
        dtype=green_diff.dtype,
        count=1
    )
    pp.pp(out_meta)

with rio.open('./data/masked_images/green-diff.tif', 'w', **out_meta) as dst:
    dst.write(green_diff, 1)



# %% Subtract the NIR band and write to file 

masked_nir_path = './data/masked_images/nir-masked-conservative.tif'

with rio.open(masked_nir_path) as src:
    ls_data = src.read(1)
    ls_data = ls_data * 10_000
    s2_data = src.read(2)
    pp.pp(src.meta)

    diff = ls_data - s2_data
    diff = np.where((diff < 500) & (diff > -500), diff, np.nan)

    out_meta = src.meta.copy()
    out_meta.update(
        dtype=diff.dtype,
        count=1
    )
    pp.pp(out_meta)

with rio.open('./data/masked_images/nir-conservative-diff.tif', 'w', **out_meta) as dst:
    dst.write(green_diff, 1)


# %% Difference between the NDWI values
masked_green_path = './data/masked_images/green-masked-conservative.tif'
masked_nir_path = './data/masked_images/nir-masked-conservative.tif'

with rio.open(masked_nir_path) as nir_src, rio.open(masked_green_path) as green_src:
    ls_nir = nir_src.read(1)
    ls_green = green_src.read(1)
    ls_nir = ls_nir * 10_000
    ls_green = ls_green * 10_000

    s2_nir = nir_src.read(2)
    s2_green = green_src.read(2)

    ndwi_ls = (ls_green - ls_nir) / (ls_green + ls_nir)
    ndwi_ls = np.where((ndwi_ls < 1) & (ndwi_ls > -1), ndwi_ls, np.nan)
    ndwi_s2 = (s2_green - s2_nir) / (s2_green + s2_nir)
    ndwi_s2 = np.where((ndwi_s2 < 1) & (ndwi_s2 > -1), ndwi_s2, np.nan)

    ndwi_diff = ndwi_ls - ndwi_s2

    out_meta = nir_src.meta.copy()
    out_meta.update(
        dtype=ndwi_diff.dtype,
        count=1
    )

with rio.open('./data/masked_images/ndwi-diff.tif', 'w', **out_meta) as dst:
    dst.write(ndwi_diff, 1)
    

# %%
