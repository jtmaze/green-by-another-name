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



# %%
