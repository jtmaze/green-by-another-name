# %% 1.0 Libraries and filepaths

import pandas as pd

lake_areas_dir = './data/lake_area_results/'
toa_data_resamp = pd.read_csv(f'{lake_areas_dir}/toa_resampled_bilinear30_area_summaries_batch2.csv')
sr_data_resamp = pd.read_csv(f'{lake_areas_dir}/sr_resampled_bilinear30_area_summaries_batch2.csv')

# 1.1 Select the relevant columns

water_frac_cols = [
    'smallest_buff_lake_ls_water_frac_adaptive', 'smallest_buff_lake_s2_water_frac_adaptive',
    'small_buff_lake_ls_water_frac_adaptive', 'small_buff_lake_s2_water_frac_adaptive', 'medium_buff_lake_ls_water_frac_adaptive',
    'medium_buff_lake_s2_water_frac_adaptive', 'large_buff_lake_ls_water_frac_adaptive', 'large_buff_lake_s2_water_frac_adaptive'
]

cols_to_keep = ['date', 'roi', 'level'] + water_frac_cols
toa_data = toa_data_resamp[cols_to_keep]
sr_data = sr_data_resamp[cols_to_keep]

# %% 2.0 

def filter_data_calc_satellite_differences(
    df: pd.DataFrame,
    water_frac_cols: list
):
    # If there's no lakes for a size bin within the image pair, the water fraction is marked 0
    # I replace these invalid zeros with 'na' values

    for idx, row in df.iterrows():
        for col in water_frac_cols:
            if row[col] == 0:
                print(f'No lakes for size = {col} for {row['roi']} on {row['date']}')

    print(f'Making missing size data as NA')

    out_df = df.copy()
    out_df[water_frac_cols] = df[water_frac_cols].replace(0, 'na')
    

    return out_df

# %%

toa_plot = filter_data_calc_satellite_differences(sr_data, water_frac_cols)
