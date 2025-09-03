# %% 1.0 Libraries and filepaths
import os
import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name')

resample_method = 'bilinear30'
lake_areas_dir = './data/lake_area_results/'
toa_data_resamp = pd.read_csv(f'{lake_areas_dir}/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = toa_data_resamp[['roi', 'date']].agg('_'.join, axis=1).unique()

sr = pd.read_csv(f'{lake_areas_dir}/sr_resampled_{resample_method}_area_summaries_batch3.csv')
sr = sr[sr[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

toa = pd.read_csv(f'{lake_areas_dir}/toa_resampled_{resample_method}_area_summaries_batch3.csv')
toa = toa[toa[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

# 1.1 Select the relevant columns

water_frac_cols = [
    'smallest_buff_lake_ls_water_frac_adaptive', 'smallest_buff_lake_s2_water_frac_adaptive',
    'small_buff_lake_ls_water_frac_adaptive', 'small_buff_lake_s2_water_frac_adaptive', 'medium_buff_lake_ls_water_frac_adaptive',
    'medium_buff_lake_s2_water_frac_adaptive', 'large_buff_lake_ls_water_frac_adaptive', 'large_buff_lake_s2_water_frac_adaptive'
]

cols_to_keep = ['date', 'roi', 'level'] + water_frac_cols
toa_data = toa[cols_to_keep]
sr_data = sr[cols_to_keep]

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
                print(f'No lakes for size = {col} for {row["roi"]} on {row["date"]}')

    print(f'Making missing size data as NA')

    out_df = df.copy()
    out_df[water_frac_cols] = df[water_frac_cols].replace(0, np.nan)

    out_df['rel_smallest_sat_diff'] = (
        (out_df['smallest_buff_lake_ls_water_frac_adaptive'] - out_df['smallest_buff_lake_s2_water_frac_adaptive']) 
            / ((out_df['smallest_buff_lake_ls_water_frac_adaptive'] + out_df['smallest_buff_lake_s2_water_frac_adaptive']) * 0.5) * 100
    )

    out_df['rel_small_sat_diff'] = (
        (out_df['small_buff_lake_ls_water_frac_adaptive'] - out_df['small_buff_lake_s2_water_frac_adaptive'])
            / ((out_df['small_buff_lake_ls_water_frac_adaptive'] + out_df['small_buff_lake_s2_water_frac_adaptive'])* 0.5 ) * 100
    )

    out_df['rel_medium_sat_diff'] = (
        (out_df['medium_buff_lake_ls_water_frac_adaptive'] - out_df['medium_buff_lake_s2_water_frac_adaptive'])
            / ((out_df['medium_buff_lake_ls_water_frac_adaptive'] + out_df['medium_buff_lake_s2_water_frac_adaptive']) * 0.5) * 100
    )
    

    out_df['rel_large_sat_diff'] = (
        (out_df['large_buff_lake_ls_water_frac_adaptive'] - out_df['large_buff_lake_s2_water_frac_adaptive'])
            / ((out_df['large_buff_lake_ls_water_frac_adaptive'] + out_df['large_buff_lake_s2_water_frac_adaptive']) * 0.5 ) * 100
    )
    
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
    'rel_smallest_sat_diff': 'Smallest (0.01-0.05 km²)',
    'rel_small_sat_diff': 'Small (0.05-0.5 km²)',
    'rel_medium_sat_diff': 'Medium (0.5-1 km²)',
    'rel_large_sat_diff': 'Large (> 1 km²)'
})

plot_data['level'] = plot_data['level'].map({
    'toa': 'TOA',
    'sr': 'SR'
})

plt.figure(figsize=(12, 9))  # Changed from (12, 5) to match 6b
ax = sns.boxplot(
    data=plot_data,
    x='lake_size',
    y='satellite_difference',
    hue='level',
    palette={'SR': '#88c999', 'TOA': '#6a9ecf'},  
    width=0.6,  # Changed from 0.7 to 0.6 to match 6b
    showfliers=False
)

for patch in ax.patches:
    fc = patch.get_facecolor()
    patch.set_facecolor(mpl.colors.to_rgba(fc, alpha=0.6))  # Changed from 0.7 to 0.6 to match 6b

plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
plt.xlabel("")
plt.ylabel('Relative Difference (%)', fontsize=20)  # Changed from 18 to 20
plt.xticks(fontsize=18, rotation=25)  # Changed from 14 to 18 and added rotation
plt.yticks(fontsize=16)  # Changed from 14 to 16
plt.legend(title='AC Level', title_fontsize=20, fontsize=16)  # Updated font sizes to match 6b

plt.tight_layout()
plt.show()

# %% Make a summary table

rel_diff_summary = plot_data.groupby(['level', 'lake_size'])['satellite_difference'].agg(
    mean='mean',
    var='var',
    std='std',
    p_val=lambda x: stats.ttest_1samp(x.dropna(), 0)[1],
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
).reset_index()

print(rel_diff_summary)

# %% Quick Tukey HSD across lake sizes for TOA satellite differences

temp = plot_data[plot_data['level'] == 'TOA']

smallest = temp[temp['lake_size'] == 'Smallest (0.01-0.05 km²)']['satellite_difference']
small = temp[temp['lake_size'] == 'Small (0.05-0.5 km²)']['satellite_difference']
medium = temp[temp['lake_size'] == 'Medium (0.5-1 km²)']['satellite_difference']
large = temp[temp['lake_size'] == 'Large (> 1 km²)']['satellite_difference']

f_stat, p_val = stats.f_oneway(smallest, small, medium, large)
print(f'F-statistic: {f_stat}, p-value: {p_val}')

toa_tukey_data = []

for values, group_name in [
    (smallest, 'smallest'),
    (small, 'small'),
    (medium, 'medium'),
    (large, 'large')
]:
    for val in values:
        toa_tukey_data.append((val, group_name))

toa_tukey_df = pd.DataFrame(toa_tukey_data, columns=['satellite_difference', 'lake_size'])
model = ols('satellite_difference ~ lake_size', data=toa_tukey_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

tukey = pairwise_tukeyhsd(
    endog=toa_tukey_df['satellite_difference'],
    groups=toa_tukey_df['lake_size'],
    alpha=0.001
)
print(tukey)


# %% 2.0 Define the function to filter and calculate satellite differences
def filter_data_calc_absolute_differences(
    df: pd.DataFrame,
    water_frac_cols: list
):
    # If there's no lakes for a size bin within the image pair, the water fraction is marked 0
    # I replace these invalid zeros with 'na' values

    for idx, row in df.iterrows():
        for col in water_frac_cols:
            if row[col] == 0:
                print(f'No lakes for size = {col} for {row["roi"]} on {row["date"]}')

    print(f'Making missing size data as NA')

    out_df = df.copy()
    out_df[water_frac_cols] = df[water_frac_cols].replace(0, np.nan)

    # Calculate absolute differences
    out_df['abs_smallest_sat_diff'] = (out_df['smallest_buff_lake_ls_water_frac_adaptive'] - 
                                          out_df['smallest_buff_lake_s2_water_frac_adaptive'])
    out_df['abs_small_sat_diff'] = (out_df['small_buff_lake_ls_water_frac_adaptive'] - 
                                       out_df['small_buff_lake_s2_water_frac_adaptive'])
    out_df['abs_medium_sat_diff'] = (out_df['medium_buff_lake_ls_water_frac_adaptive'] - 
                                        out_df['medium_buff_lake_s2_water_frac_adaptive'])
    out_df['abs_large_sat_diff'] = (out_df['large_buff_lake_ls_water_frac_adaptive'] - 
                                       out_df['large_buff_lake_s2_water_frac_adaptive'])
    
    return out_df

# Run the absolute difference analysis
toa_abs_plot = filter_data_calc_absolute_differences(toa_data, water_frac_cols)
sr_abs_plot = filter_data_calc_absolute_differences(sr_data, water_frac_cols)

abs_plot_df = pd.concat([toa_abs_plot, sr_abs_plot])

# %% Create plot data for absolute differences
abs_plot_data = abs_plot_df.melt(
    id_vars=['date', 'roi', 'level'],
    value_vars=[
        'abs_smallest_sat_diff', 'abs_small_sat_diff', 'abs_medium_sat_diff', 'abs_large_sat_diff'
    ],
    var_name='lake_size',
    value_name='absolute_satellite_difference'
)

abs_plot_data['lake_size'] = abs_plot_data['lake_size'].map({
    'abs_smallest_sat_diff': 'Smallest (0.01-0.05 km²)',
    'abs_small_sat_diff': 'Small (0.05-0.5 km²)',
    'abs_medium_sat_diff': 'Medium (0.5-1 km²)',
    'abs_large_sat_diff': 'Large (> 1 km²)'
})

abs_plot_data['level'] = abs_plot_data['level'].map({
    'toa': 'TOA',
    'sr': 'SR'
})

plt.figure(figsize=(12, 9))  # Changed from (12, 5) to match 6b
ax = sns.boxplot(
    data=abs_plot_data,
    x='lake_size',
    y='absolute_satellite_difference',
    hue='level',
    palette={'SR': '#88c999', 'TOA': '#6a9ecf'},  
    width=0.6,  # Changed from 0.7 to 0.6 to match 6b
    showfliers=False
)

for patch in ax.patches:
    fc = patch.get_facecolor()
    patch.set_facecolor(mpl.colors.to_rgba(fc, alpha=0.6))  # Changed from 0.7 to 0.6 to match 6b

plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
plt.xlabel("")
plt.ylabel('Absolute Difference (%)', fontsize=20)  # Changed from 18 to 20
plt.xticks(fontsize=18, rotation=25)  # Changed from 14 to 18 and added rotation
plt.yticks(fontsize=16)  # Changed from 14 to 16
plt.legend().set_visible(False)  # Keep legend removed

plt.tight_layout()
plt.show()
# %%

abs_diff_summary = abs_plot_data.groupby(['level', 'lake_size'])['absolute_satellite_difference'].agg(
    mean='mean',
    var='var',
    std='std',
    p_val=lambda x: stats.ttest_1samp(x.dropna(), 0)[1],
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
).reset_index()

print(abs_diff_summary)

# %%

temp = abs_plot_data[abs_plot_data['level'] == 'TOA']

smallest = temp[temp['lake_size'] == 'Smallest (0.01-0.05 km²)']['absolute_satellite_difference']
small = temp[temp['lake_size'] == 'Small (0.05-0.5 km²)']['absolute_satellite_difference']
medium = temp[temp['lake_size'] == 'Medium (0.5-1 km²)']['absolute_satellite_difference']
large = temp[temp['lake_size'] == 'Large (> 1 km²)']['absolute_satellite_difference']

f_stat, p_val = stats.f_oneway(smallest, small, medium, large)
print(f'F-statistic: {f_stat}, p-value: {p_val}')

toa_tukey_data = []

for values, group_name in [
    (smallest, 'smallest'),
    (small, 'small'),
    (medium, 'medium'),
    (large, 'large')
]:
    for val in values:
        toa_tukey_data.append((val, group_name))

toa_tukey_df = pd.DataFrame(toa_tukey_data, columns=['satellite_difference', 'lake_size'])
model = ols('satellite_difference ~ lake_size', data=toa_tukey_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

tukey = pairwise_tukeyhsd(
    endog=toa_tukey_df['satellite_difference'],
    groups=toa_tukey_df['lake_size'],
    alpha=0.001
)
print(tukey)