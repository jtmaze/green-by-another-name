# %% 1.0 Libraries and file paths

import glob
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

ts_dir = './data/s2_weekly_timeseries'

ts_files = glob.glob(f'{ts_dir}/*.csv')
# %%
dfs = []
for f in ts_files:
    df = pd.read_csv(f)
    df = df[
        ['invalid_pixels', 'water_pixels', 'land_pixels', 
        'total_roi_pixels', 'total_lake_pixels', 'roi_name', 'mosaic_id']
    ].copy()

    df['year'] = df['mosaic_id'].apply(lambda x: x.split('_')[-2])
    df['week'] = df['mosaic_id'].apply(lambda x: x.split('_')[-1])

    df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int64')
    df['valid_fraction'] = ((df['total_lake_pixels'] - df['invalid_pixels'])
                            / df['total_lake_pixels'] * 100)
    df['lake_water_fraction'] = (df['water_pixels'] / 
                                 (df['total_lake_pixels'] - df['invalid_pixels']) * 100)
    
    print(f'Total observations for {df['roi_name'].iloc[0]} is {len(df)}')

    # plt.figure(figsize=(12, 6))
    # scatter = plt.scatter(
    #     x=df['valid_fraction'],
    #     y=df['lake_water_fraction'],
    #     c=df['week'],
    #     cmap='RdYlBu'
    # )
    # plt.title(f'{df['roi_name'].iloc[0]} lake water % relative to valid obs %')
    # plt.colorbar(scatter, label='Week of year')
    # plt.xlabel('Valid Fraction of ALPOD pixels')
    # plt.ylabel('Water Fraction of ALPOD pixels')
    # plt.show()

    df_clean = df[(df['valid_fraction'] >= 40)]
    print("------------------------------------------------")
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        x=df_clean['valid_fraction'],
        y=df_clean['lake_water_fraction'],
        c=df_clean['week'],
        cmap='RdYlBu'
    )
    plt.title(f'{df_clean['roi_name'].iloc[0]} lake water % relative to valid obs %')
    plt.colorbar(scatter, label='Week of year')
    plt.xlabel('Valid Fraction of ALPOD pixels')
    plt.ylabel('Water Fraction of ALPOD pixels')
    plt.show()


    dfs.append(df_clean)

timeseries = pd.concat(dfs, ignore_index=True)
rois = timeseries['roi_name'].unique()

# %%


# %% 2.0 Read and format the lake area data

def add_year_week(df, datetime_col='datetime'):
    """
    Add a new series 'year_week' to the DataFrame with .isocalendar()
    """    
    iso_calendar = df[datetime_col].dt.isocalendar()
    # Create the year_week column by combining the year and week number as a string (e.g. "2025-14")
    df['year_week'] = iso_calendar['year'].astype(str) + '-' + iso_calendar['week'].astype(str).str.zfill(2)
    return df

def datetime_to_yrweek():
    pass



# %% 


# %%
