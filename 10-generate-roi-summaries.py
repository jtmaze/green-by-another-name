# %% 1.0 Libraries and file paths
import pandas as pd
import glob

lake_area_dir = './data/lake_area_results/'
lake_area_files = glob.glob(lake_area_dir + '*.csv')
print(lake_area_files)

# %% 2.0 Read CSV files into DataFrames, combine into one
lake_area_dfs = []
for f in lake_area_files:
    df = pd.read_csv(f)
    lake_area_dfs.append(df)

area_data = pd.concat(lake_area_dfs, ignore_index=True)
# %% 3.0 Make a summary of observations by roi

sub_roi_summary = area_data.groupby(['roi']).agg(
    unique_dt_count=('date', 'nunique')
).sort_values(by='unique_dt_count', ascending=False).reset_index()

sub_roi_summary['roi_main'] = sub_roi_summary['roi'].str.split('_').str[0]

roi_summary = sub_roi_summary.groupby(['roi_main']).agg(
    total_unique_dt_count=('unique_dt_count', 'sum')
).sort_values(by='total_unique_dt_count', ascending=False).reset_index()

# %%
