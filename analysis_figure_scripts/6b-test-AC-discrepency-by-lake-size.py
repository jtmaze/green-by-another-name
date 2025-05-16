# %% 1.0 Libraries and filepaths
import os
import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name')

resample_method = 'bilinear30'
lake_areas_dir = './data/lake_area_results/'
toa_data_resamp = pd.read_csv(f'{lake_areas_dir}/toa_resampled_{resample_method}_area_summaries_batch3.csv')
valids = toa_data_resamp[['roi', 'date']].agg('_'.join, axis=1).unique()


sr = pd.read_csv(f'{lake_areas_dir}/sr_resampled_{resample_method}_area_summaries_batch3.csv')
sr = sr[sr[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
toa = pd.read_csv(f'{lake_areas_dir}/toa_resampled_{resample_method}_area_summaries_batch3.csv')
toa = toa[toa[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

# 1.1 Select the relevant columns

water_frac_cols = [
    'smallest_lake_ls_water_frac_adaptive', 'smallest_lake_s2_water_frac_adaptive',
    'small_lake_ls_water_frac_adaptive', 'small_lake_s2_water_frac_adaptive', 'medium_lake_ls_water_frac_adaptive',
    'medium_lake_s2_water_frac_adaptive', 'large_lake_ls_water_frac_adaptive', 'large_lake_s2_water_frac_adaptive'
]

cols_to_keep = ['date', 'roi', 'level'] + water_frac_cols
toa_data = toa[cols_to_keep]
sr_data = sr[cols_to_keep]

# %% 2.0 Define the function to filter and calculate satellite differences

def filter_data_rename_cols(
    df: pd.DataFrame,
    water_frac_cols: list, 
    level: str
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

    # Rename the columns with level and to be more concise
    rename_dict = {}
    for col in water_frac_cols:
        split_col = col.split('_')
        new_string = f'{level}_{split_col[0]}_{split_col[2]}'
        rename_dict[col] = new_string
    
    # Rename the columns using the dictionary
    out_df = out_df.rename(columns=rename_dict)
    out_df =out_df.drop(columns=['level'])
    return out_df

# %% 3.0 Orgaized the data by lake size and satellite differences

toa_plot = filter_data_rename_cols(toa_data, water_frac_cols, 'toa')
sr_plot = filter_data_rename_cols(sr_data, water_frac_cols, 'sr')

combined = pd.merge(
    sr_plot, 
    toa_plot,
    how='inner',
    on=['date', 'roi']
)

combined['rel_ls_smallest_ac_diff'] = (
    (combined['toa_smallest_ls'] - combined['sr_smallest_ls']) 
        / ((combined['toa_smallest_ls'] + combined['sr_smallest_ls']) * 0.5) * 100
)

combined['rel_ls_small_ac_diff'] = (
    (combined['toa_small_ls'] - combined['sr_small_ls'])
        / ((combined['toa_small_ls'] + combined['sr_small_ls']) * 0.5 ) * 100
)


combined['rel_ls_medium_ac_diff'] = (
    (combined['toa_medium_ls'] - combined['sr_medium_ls'])
        / ((combined['toa_medium_ls'] + combined['sr_medium_ls']) * 0.5) * 100
)

combined['rel_ls_large_ac_diff'] = (
    (combined['toa_large_ls'] - combined['sr_large_ls'])
        / ((combined['toa_large_ls'] + combined['sr_large_ls']) * 0.5) * 100
)

combined['rel_s2_smallest_ac_diff'] = (
    (combined['toa_smallest_s2'] - combined['sr_smallest_s2'])
        / ((combined['toa_smallest_s2'] + combined['sr_smallest_s2']) * 0.5) * 100
)

combined['rel_s2_small_ac_diff'] = (
    (combined['toa_small_s2'] - combined['sr_small_s2'])
        / ((combined['toa_small_s2'] + combined['sr_small_s2']) * 0.5) * 100
)

combined['rel_s2_medium_ac_diff'] = (
    (combined['toa_medium_s2'] - combined['sr_medium_s2'])
        / ((combined['toa_medium_s2'] + combined['sr_medium_s2']) * 0.5) * 100
)

combined['rel_s2_large_ac_diff'] = (
    (combined['toa_large_s2'] - combined['sr_large_s2'])
        / ((combined['toa_large_s2'] + combined['sr_large_s2']) * 0.5) * 100
)

combined = combined[[
    'date', 'roi',
    'rel_ls_smallest_ac_diff', 'rel_ls_small_ac_diff', 'rel_ls_medium_ac_diff', 'rel_ls_large_ac_diff',
    'rel_s2_smallest_ac_diff', 'rel_s2_small_ac_diff', 'rel_s2_medium_ac_diff', 'rel_s2_large_ac_diff'
]]

# %% 2.0 Create the boxplot

plot_data = pd.melt(
    combined,
    id_vars=['date', 'roi'],
    value_vars=[
        'rel_ls_smallest_ac_diff', 'rel_ls_small_ac_diff', 'rel_ls_medium_ac_diff', 'rel_ls_large_ac_diff',
        'rel_s2_smallest_ac_diff', 'rel_s2_small_ac_diff', 'rel_s2_medium_ac_diff', 'rel_s2_large_ac_diff'
    ],
    var_name='metric',
    value_name='relative_difference'
)

# Extract satellite type and lake size from the metric column
plot_data['satellite'] = plot_data['metric'].apply(lambda x: 'Landsat 8' if 'ls_' in x else 'Sentinel-2')
plot_data['lake_size'] = plot_data['metric'].apply(lambda x: x.split('_')[2].capitalize())
plot_data = plot_data.drop(columns=['metric'])

# Order lake sizes appropriately
size_order = ['Smallest', 'Small', 'Medium', 'Large']
plot_data['lake_size'] = pd.Categorical(plot_data['lake_size'], categories=size_order, ordered=True)
size_labels = {
    'Smallest': 'Smallest (0.01-0.05 km²)',
    'Small': 'Small (0.05-0.5 km²)',
    'Medium': 'Medium (0.5-1 km²)',
    'Large': 'Large (> 1 km²)'
}

# Then apply it to your lake_size column:
plot_data['lake_size'] = plot_data['lake_size'].map(size_labels)

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=plot_data,
    x='lake_size',
    y='relative_difference',
    hue='satellite',
    palette={'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'},
    width=0.7
)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.title(f'AC Differences by Lake Size for {resample_method}', fontsize=14)
plt.xlabel('Lake Size Category', fontsize=12)
plt.ylabel('Relative AC Difference (%)', fontsize=12)
plt.ylim(-100, 210)
plt.tight_layout()
plt.show()

# %% Make a summary table for relative differences

relative_diff_summary = plot_data.groupby(['lake_size', 'satellite'], observed=True)['relative_difference'].agg(
    mean='mean',
    var='var',
    std='std',
    pvalue=lambda x: stats.ttest_1samp(x.dropna(), 0)[1],
    q25=lambda x: x.quantile(0.25),
    q50=lambda x: x.quantile(0.5),
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
).reset_index()

print(relative_diff_summary)

# %% HSD test for Landsat 8 AC differences
temp = plot_data[plot_data['satellite'] == 'Landsat 8']
print(temp.columns)

smallest = temp[temp['lake_size'] == 'Smallest (0.01-0.05 km²)']['relative_difference'].dropna()
small = temp[temp['lake_size'] == 'Small (0.05-0.5 km²)']['relative_difference'].dropna()
medium = temp[temp['lake_size'] == 'Medium (0.5-1 km²)']['relative_difference'].dropna()
large = temp[temp['lake_size'] == 'Large (> 1 km²)']['relative_difference'].dropna()

f_stat, p_val = stats.f_oneway(smallest, small, medium, large)
print(f'F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}')

ls8_tukey_data = []
for values, group_name in [
    (smallest, 'smallest'),
    (small, 'small'),
    (medium, 'medium'),
    (large, 'large')
]:
    for value in values:
        ls8_tukey_data.append((value, group_name))

ls8_tukey_df = pd.DataFrame(ls8_tukey_data, columns=['value', 'group'])
tukey_result = pairwise_tukeyhsd(
    endog=ls8_tukey_df['value'],
    groups=ls8_tukey_df['group'],
    alpha=0.01
)
print(tukey_result)

# %% Make the same plot for Sentinel-2 
temp = plot_data[plot_data['satellite'] == 'Sentinel-2']
print(temp.columns)

smallest = temp[temp['lake_size'] == 'Smallest (0.01-0.05 km²)']['relative_difference'].dropna()
small = temp[temp['lake_size'] == 'Small (0.05-0.5 km²)']['relative_difference'].dropna()
medium = temp[temp['lake_size'] == 'Medium (0.5-1 km²)']['relative_difference'].dropna()
large = temp[temp['lake_size'] == 'Large (> 1 km²)']['relative_difference'].dropna()

f_stat, p_val = stats.f_oneway(smallest, small, medium, large)
print(f'F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}')

s2_tukey_data = []
for values, group_name in [
    (smallest, 'smallest'),
    (small, 'small'),
    (medium, 'medium'),
    (large, 'large')
]:
    for value in values:
        s2_tukey_data.append((value, group_name))

s2_tukey_df = pd.DataFrame(s2_tukey_data, columns=['value', 'group'])
tukey_result = pairwise_tukeyhsd(
    endog=s2_tukey_df['value'],
    groups=s2_tukey_df['group'],
    alpha=0.01
)

print(tukey_result)

# %% 3.0 Create the boxplot for absolute differences


# combined = pd.merge(
#     sr_plot, 
#     toa_plot,
#     how='inner',
#     on=['date', 'roi']
# )

# # First, create absolute difference columns
# combined['abs_ls_smallest_ac_diff'] = combined['toa_smallest_ls'] - combined['sr_smallest_ls']
# combined['abs_ls_small_ac_diff'] = combined['toa_small_ls'] - combined['sr_small_ls']
# combined['abs_ls_medium_ac_diff'] = combined['toa_medium_ls'] - combined['sr_medium_ls']
# combined['abs_ls_large_ac_diff'] = combined['toa_large_ls'] - combined['sr_large_ls']
# combined['abs_s2_smallest_ac_diff'] = combined['toa_smallest_s2'] - combined['sr_smallest_s2']
# combined['abs_s2_small_ac_diff'] = combined['toa_small_s2'] - combined['sr_small_s2']
# combined['abs_s2_medium_ac_diff'] = combined['toa_medium_s2'] - combined['sr_medium_s2']
# combined['abs_s2_large_ac_diff'] = combined['toa_large_s2'] - combined['sr_large_s2']

# # Reshape data for plotting
# abs_plot_data = pd.melt(
#     combined,
#     id_vars=['date', 'roi'],
#     value_vars=[
#         'abs_ls_smallest_ac_diff', 'abs_ls_small_ac_diff', 'abs_ls_medium_ac_diff', 'abs_ls_large_ac_diff',
#         'abs_s2_smallest_ac_diff', 'abs_s2_small_ac_diff', 'abs_s2_medium_ac_diff', 'abs_s2_large_ac_diff'
#     ],
#     var_name='metric',
#     value_name='absolute_difference'
# )

# # Extract satellite type and lake size from the metric column
# abs_plot_data['satellite'] = abs_plot_data['metric'].apply(lambda x: 'Landsat 8' if 'ls_' in x else 'Sentinel-2')
# abs_plot_data['lake_size'] = abs_plot_data['metric'].apply(lambda x: x.split('_')[2].capitalize())

# # Order lake sizes appropriately
# size_order = ['Smallest', 'Small', 'Medium', 'Large']
# abs_plot_data['lake_size'] = pd.Categorical(abs_plot_data['lake_size'], categories=size_order, ordered=True)
# size_labels = {
#     'Smallest': 'Smallest (0.01-0.05 km²)',
#     'Small': 'Small (0.05-0.5 km²)',
#     'Medium': 'Medium (0.5-1 km²)',
#     'Large': 'Large (> 1 km²)'
# }

# # Apply size labels
# abs_plot_data['lake_size'] = abs_plot_data['lake_size'].map(size_labels)

# # Create the boxplot for absolute differences
# plt.figure(figsize=(12, 5))
# sns.boxplot(
#     data=abs_plot_data,
#     x='lake_size',
#     y='absolute_difference',
#     hue='satellite',
#     palette={'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'},
#     width=0.7
# )
# plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
# plt.title(f'Absolute AC Differences by Lake Size for {resample_method}', fontsize=14)
# plt.xlabel('Lake Size Category', fontsize=12)
# plt.ylabel('Absolute AC Difference (TOA% - SR%)', fontsize=12)
# plt.ylim(-15, 40)  # Adjust as needed for your data
# plt.tight_layout()
# plt.show()