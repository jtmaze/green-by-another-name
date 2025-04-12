# %% 1.0 Libraries and file paths

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ts_dir = './data/s2_timeseries'

ts_files = glob.glob(f'{ts_dir}/*.csv')

dfs = []
for f in ts_files:
    df = pd.read_csv(f)
    dfs.append(df)

timeseries = pd.concat(dfs, ignore_index=True)
# %% 2.0 Format the Sentinel-2 timeseries

timeseries = timeseries[
    ['invalid_pixels', 'water_pixels', 'land_pixels', 
     'total_roi_pixels', 'total_lake_pixels', 'roi_name', 'mosaic_id']
].copy()

timeseries['lake_valid_frac'] = ((timeseries['total_lake_pixels'] - timeseries['invalid_pixels'])
                                  / timeseries['total_lake_pixels'] * 100)

timeseries.sort_values(by='lake_valid_frac', ascending=False, inplace=True)

clean_timeseries = timeseries[
    timeseries['lake_valid_frac'] > 50
].copy()
# %% 

clean_timeseries['lake_water_frac'] = (
    clean_timeseries['water_pixels'] / (clean_timeseries['total_lake_pixels'] - clean_timeseries['invalid_pixels']) * 100
)

# %%

plt.figure(figsize=(10, 6))
sns.histplot(
    data=clean_timeseries,
    x='lake_water_frac',
    bins=30,
    color='#6a9ecf',
    edgecolor='black'
)

plt.legend()
plt.tight_layout()
plt.show()