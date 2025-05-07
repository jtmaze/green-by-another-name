# %% 1.0 Libraries and file paths
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')
lake_area_dir = './data/lake_area_results/'
lake_area_files = glob.glob(lake_area_dir + '*_bilinear30_area_summaries_batch3.csv')
print(lake_area_files)

# %% 2.0 Read CSV files into DataFrames, combine into one
lake_area_dfs = []
for f in lake_area_files:
    df = pd.read_csv(f)
    lake_area_dfs.append(df)
    print(len(df))

area_data = pd.concat(lake_area_dfs, ignore_index=True)

count_data = area_data[area_data['level'] == 'toa']

unique_target_areas = count_data['roi'].unique()
print(unique_target_areas)

# %% 3.0 Plot the distribution of PLD+60m pixels that were masked

count_data['pld_plus_valid_frac'].hist(
    bins=10,
    edgecolor='black',
    alpha=0.7,
    grid=False,
    density=False
)

plt.title('Proportion of Lake Coverage Across Image Sets')
plt.xlabel('Image sets valid (%) of PLD + 60m pixels')
plt.ylabel('Number of Image Sets')
plt.show()

# %% 3.0 Make a summary of observations by roi

count_data = count_data[['date', 'roi', 'pld_plus_valid_frac']].copy()
count_data['roi_main'] = count_data['roi'].str.split('_').str[0]
count_data['unique_roi_date'] = count_data['roi'].astype(str) + count_data['date'].astype(str)
count_data['above_75'] = np.where(count_data['pld_plus_valid_frac'] >= 75, True, False)
count_data['month'] = pd.to_datetime(count_data['date']).dt.month.astype(str)

# %% 

roi_summary = count_data.groupby(['roi_main']).agg(
    total_obs_count=('unique_roi_date', 'count'),
    above_75_count=('above_75', 'sum')
)

# %% 


# %%
month_summary = count_data.groupby(['month']).agg(
    total_obs_count=('unique_roi_date', 'count'),
    above_75_count=('above_75', 'sum')
)
# %%

sub_roi_summary = area_data.groupby(['roi']).agg(
    unique_dt_count=('date', 'nunique')
).sort_values(by='unique_dt_count', ascending=False).reset_index()