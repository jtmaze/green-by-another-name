# %% 1.0 Libraries and filepaths

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

resample_method = 'bilinear30'
lake_areas_dir = './data/lake_area_results/'
toa_data_resamp = pd.read_csv(f'{lake_areas_dir}/toa_resampled_{resample_method}_area_summaries_batch2.csv')
sr_data_resamp = pd.read_csv(f'{lake_areas_dir}/sr_resampled_{resample_method}_area_summaries_batch2.csv')

# 1.1 Select the relevant columns

water_frac_cols = [
    'smallest_buff_lake_ls_water_frac_adaptive', 'smallest_buff_lake_s2_water_frac_adaptive',
    'small_buff_lake_ls_water_frac_adaptive', 'small_buff_lake_s2_water_frac_adaptive', 'medium_buff_lake_ls_water_frac_adaptive',
    'medium_buff_lake_s2_water_frac_adaptive', 'large_buff_lake_ls_water_frac_adaptive', 'large_buff_lake_s2_water_frac_adaptive'
]

cols_to_keep = ['date', 'roi', 'level'] + water_frac_cols
toa_data = toa_data_resamp[cols_to_keep]
sr_data = sr_data_resamp[cols_to_keep]


# %% 2.0 Define the function to filter and calculate satellite differences

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
    out_df[water_frac_cols] = df[water_frac_cols].replace(0, np.nan)

    out_df['rel_smallest_sat_diff'] = ((out_df['smallest_buff_lake_ls_water_frac_adaptive'] - out_df['smallest_buff_lake_s2_water_frac_adaptive']) 
                             / out_df['smallest_buff_lake_ls_water_frac_adaptive'])
    out_df['rel_small_sat_diff'] = ((out_df['small_buff_lake_ls_water_frac_adaptive'] - out_df['small_buff_lake_s2_water_frac_adaptive'])
                            / out_df['small_buff_lake_ls_water_frac_adaptive'])
    out_df['rel_medium_sat_diff'] = ((out_df['medium_buff_lake_ls_water_frac_adaptive'] - out_df['medium_buff_lake_s2_water_frac_adaptive'])
                            / out_df['medium_buff_lake_ls_water_frac_adaptive'])
    out_df['rel_large_sat_diff'] = ((out_df['large_buff_lake_ls_water_frac_adaptive'] - out_df['large_buff_lake_s2_water_frac_adaptive'])
                            / out_df['large_buff_lake_ls_water_frac_adaptive'])
    
    return out_df

# %% 3.0 Orgaized the data by lake size and satellite differences

toa_plot = filter_data_calc_satellite_differences(toa_data, water_frac_cols)
sr_plot = filter_data_calc_satellite_differences(sr_data, water_frac_cols)

plot_df = pd.concat([toa_plot, sr_plot])


plot_data = plot_df.melt(
    id_vars=['date', 'roi', 'level'],
    value_vars=[
        'rel_smallest_sat_diff', 'rel_small_sat_diff', 'rel_medium_sat_diff', 'rel_large_sat_diff'
    ],
    var_name='lake_size',
    value_name='satellite_difference'
)

plot_data['lake_size'] = plot_data['lake_size'].map({
    'rel_smallest_sat_diff': 'Smallest (0.01-0.5 km²)',
    'rel_small_sat_diff': 'Small (0.05-0.5 km²)',
    'rel_medium_sat_diff': 'Medium (0.5-10 km²)',
    'rel_large_sat_diff': 'Large (> 1 km²)'
})

plt.figure(figsize=(12, 5))
sns.boxplot(
    data=plot_data,
    x='lake_size',
    y='satellite_difference',
    hue='level',
    palette={'sr': '#4C72B0', 'toa': '#55A868'},  
    width=0.7
)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.title(f'Satellite Differences by Lake Size for {resample_method}', fontsize=14)
plt.xlabel('Lake Size Category', fontsize=12)
plt.ylabel('Relative Satellite Difference (LS8% - S2%) / LS8%', fontsize=12)
plt.ylim(-4, 1.25) # NOTE: Adjust this, because the plot is a little messy

plt.tight_layout()
plt.show()
# %%


