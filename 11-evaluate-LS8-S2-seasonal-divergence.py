# %% 1.0 Libraries and file paths

import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

toa = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch2.csv')
toa = toa[toa['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']
sr = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch2.csv')
sr = sr[sr['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']

# %% 2.0 Prepare the data

cols_to_make_float = [
        'total_ls_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_otsu', 'total_s2_water_frac_adaptive',
        'lake_ls_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_otsu', 'lake_s2_water_frac_adaptive',
        'shoreline_ls_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_otsu', 'shoreline_s2_water_frac_adaptive',
]

for col in cols_to_make_float:
    toa[col] = toa[col].astype(float)
    sr[col] = sr[col].astype(float)


toa.loc[:, 'main_roi'] = toa.apply(lambda row: row['roi'].split('_')[0], axis=1)
sr.loc[:, 'main_roi'] = sr.apply(lambda row: row['roi'].split('_')[0], axis=1)
# %% Define month orders and apply to data

month_order = ['May','Jun','Jul','Aug','Sep']
half_month_order = [f"{prefix}_{month}" for month in month_order for prefix in ['Early', 'Late']]
month_dtype = CategoricalDtype(categories=month_order, ordered=True)
half_month_dtype = CategoricalDtype(categories=half_month_order, ordered=True)

toa['date'] = pd.to_datetime(toa['date'])
toa['month'] = toa['date'].dt.strftime('%b').astype(month_dtype)
toa['day'] = toa['date'].dt.day
toa['half_month'] = toa.apply(
    lambda x: f"Early_{x['date'].strftime('%b')}" if x['day'] <= 15 else f"Late_{x['date'].strftime('%b')}", 
    axis=1
).astype(half_month_dtype)



# %% Calculate total and relative differences for TOA

toa['total_diff_toa'] = toa['total_ls_water_frac_adaptive'] - toa['total_s2_water_frac_adaptive']
toa['lake_diff_toa'] = toa['lake_ls_water_frac_adaptive'] - toa['lake_s2_water_frac_adaptive']
toa['shoreline_diff_toa'] = toa['shoreline_ls_water_frac_adaptive'] - toa['shoreline_s2_water_frac_adaptive']
toa['relative_total_diff_toa'] = toa['total_diff_toa'] / toa['total_ls_water_frac_adaptive']
toa['relative_lake_diff_toa'] = toa['lake_diff_toa'] / toa['lake_ls_water_frac_adaptive']
toa['relative_shoreline_diff_toa'] = toa['shoreline_diff_toa'] / toa['shoreline_ls_water_frac_adaptive']

# %%
plot_df = toa[['date', 'month', 'half_month', 'roi', 'main_roi',
           'total_diff_toa', 'relative_total_diff_toa',
           'lake_diff_toa', 'relative_lake_diff_toa',
           'shoreline_diff_toa', 'relative_shoreline_diff_toa']].copy()

# %%


melt_df = plot_df.melt(id_vars=['date', 'month', 'half_month', 'roi', 'main_roi'],
                       value_vars=['total_diff_toa', 'relative_total_diff_toa',
                                   'lake_diff_toa', 'relative_lake_diff_toa',
                                   'shoreline_diff_toa', 'relative_shoreline_diff_toa'],
                       var_name='diff_type', value_name='diff_value')

# %%

plot_df = melt_df[melt_df['diff_type'].isin(
    ['total_diff_toa', 'lake_diff_toa', 'shoreline_diff_toa']
)]

plt.figure(figsize=(12, 8))
sns.boxplot(x='month', y='diff_value', hue='diff_type', data=plot_df)
plt.title('Absolute Differences in Water Fraction (TOA) (LS8% - S2%)')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Month')
plt.ylim(-12, 5)
plt.ylabel('Difference Value %')
plt.ylabel('Difference Value')

# %%

plot_df = melt_df[melt_df['diff_type'].isin(
    ['relative_total_diff_toa', 'relative_lake_diff_toa', 'relative_shoreline_diff_toa']
)]

plt.figure(figsize=(12, 8))
sns.boxplot(x='month', y='diff_value', hue='diff_type', data=plot_df)
plt.title('Relative Differences in Water Fraction (TOA) (LS8% - S2%) / LS8%')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Month')
plt.ylim(-1, 0.25)
plt.ylabel('Difference Value (LS8% - S2%) / LS8%')

# %%

plot_df = melt_df[melt_df['main_roi'] == 'YKF']
print(len(plot_df))

plot_df = plot_df[plot_df['diff_type'].isin(
    ['total_diff_toa', 'lake_diff_toa', 'shoreline_diff_toa']
)]
print(len(plot_df))
plt.figure(figsize=(12, 8))
sns.boxplot(x='month', y='diff_value', hue='diff_type', data=plot_df)
plt.title('YKF Absolute Differences in Water Fraction (TOA) (LS8% - S2%)')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Difference Value %')
plt.ylabel('Difference Value')


# %%

plot_df = melt_df[melt_df['main_roi'] == 'YKF']
print(len(plot_df))

plot_df = plot_df[plot_df['diff_type'].isin(
    ['relative_total_diff_toa', 'relative_lake_diff_toa', 'relative_shoreline_diff_toa']
)]
print(len(plot_df))
plt.figure(figsize=(12, 8))
sns.boxplot(x='half_month', y='diff_value', hue='diff_type', data=plot_df)
plt.title('YKF Relative Differences in Water Fraction (TOA) (LS8% - S2% / LS8%')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Difference Value %')
plt.ylim(-0.8, 0.2)

# %%
