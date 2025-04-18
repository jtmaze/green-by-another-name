# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reflectance_comp_dir = './data/regression_summaries'
area_dir = './data/lake_area_results'
resample_method = 'bilinear30'
# %%

lake_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_0m_lake_{resample_method}.csv')
lake_data['zone'] = 'lake'
print(len(lake_data))
lake_plus_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_60m_lake_{resample_method}.csv')
lake_plus_data['zone'] = 'lake_plus'
print(len(lake_plus_data))
land_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_60m_land_{resample_method}.csv')
land_data['zone'] = 'land'
print(len(land_data))
shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_shoreline_neg60-60_{resample_method}.csv')
shoreline_data['zone'] = 'shoreline'
print(len(shoreline_data))

combined = pd.concat([lake_data, lake_plus_data, land_data, shoreline_data])

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'above_frac']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values='above_frac'
).reset_index()

# %% Plot the above and below frac by level
plot_data = combined[combined['level'] == 'toa']

plt.figure(figsize=(12, 5))
sns.boxplot(
    data=plot_data,
    x='zone',
    y='above_frac',
    hue='band_name',
    palette='Set2'
)
plt.axhline(y=50, color='red', linestyle='--')
plt.xlabel('Landscape Zone')
plt.ylabel('(%) Pixels S2 > LS8 Reflectance')
plt.title('')
plt.show()

# %% Read the area data

toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch2.csv')
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch2.csv')

cols_to_keep =['date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive',
               'buff_lake_s2_water_frac_adaptive']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_water_frac',
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_water_frac',
    }
).copy()

combined_area = pd.concat([toa_data, sr_data])

combined_area['abs_sat_diff'] = combined_area['ls_water_frac'] - combined_area['s2_water_frac']
combined_area['rel_sat_diff'] = combined_area['abs_sat_diff'] / combined_area['ls_water_frac'] * 100

toa_area = combined_area[combined_area['level'] == 'toa']
sr_area = combined_area[combined_area['level'] == 'sr']
# %% 

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'above_frac']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values='above_frac'
).reset_index()

combined_wide['land_lake_diff'] = combined_wide['land'] - combined_wide['lake_plus']
combined_wide['land_lake_ratio'] = combined_wide['land'] / combined_wide['lake_plus']

area_reflectance = pd.merge(sr_area, combined_wide, on=['level', 'roi', 'date'], how='left')

# %% 
plot_data = area_reflectance.copy()
plot_data = plot_data[plot_data['band_name'] == 'NDWI']
#plot_data = plot_data[plot_data['land_lake_diff'] >= 10]
#plot_data = plot_data[plot_data['rel_sat_diff'] >= -100]
plot_data['roi_main'] = plot_data['roi'].apply(lambda x: x.split('_')[0])
print(plot_data['level'].unique())
# %%

plt.figure(figsize=(10, 6))
sns.lmplot(
    data=plot_data,
    x='lake_plus',  
    y='rel_sat_diff', 
    hue='roi_main',
    scatter_kws={'alpha': 0.5},
    ci=None,
    height=6,
    aspect=1.67
)
plt.title('')
plt.xlabel('NDWI fraction (land:lake)')
plt.ylabel('LS8-S2 water frac diff')
plt.tight_layout()
plt.show()