# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'r_squared']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values='r_squared'
).reset_index()

# %% Plot the above and below frac by level
plot_data = combined[combined['level'] == 'sr']

plt.figure(figsize=(12, 7))
sns.boxplot(
    data=plot_data,
    x='zone',
    y='r_squared',
    hue='band_name',
    palette='Set2'
)
#plt.axhline(y=50, color='red', linestyle='--')
plt.xlabel('Landscape Zone')
plt.ylim(0.5, 1)
plt.ylabel('r-squared value')
plt.title('')
plt.show()

# %% Read the area data

toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch2.csv')
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch2.csv')

cols_to_keep =['date', 'roi', 'level', 'shoreline_ls_water_frac_adaptive',
               'shoreline_s2_water_frac_adaptive']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        'shoreline_ls_water_frac_adaptive': 'ls_water_frac',
        'shoreline_s2_water_frac_adaptive': 's2_water_frac',
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        'shoreline_ls_water_frac_adaptive': 'ls_water_frac',
        'shoreline_s2_water_frac_adaptive': 's2_water_frac',
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

area_reflectance = pd.merge(toa_area, combined_wide, on=['level', 'roi', 'date'], how='left')

# %% 
plot_data2 = area_reflectance.copy()
plot_data2 = plot_data2[plot_data2['band_name'] == 'NDWI']
plot_data2 = plot_data2[plot_data2['shoreline'] >= 40]
plot_data2 = plot_data2[plot_data2['rel_sat_diff'] >= -100]
#plot_data['roi_main'] = plot_data['roi'].apply(lambda x: x.split('_')[0])
print(plot_data2['level'].unique())
# %%

slope, intercept, r_value, p_value, std_err = linregress(
    plot_data2['shoreline'], 
    plot_data2['rel_sat_diff']
)

# Print regression statistics
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4g}")
print(f"Standard Error: {std_err:.4f}")

plt.figure(figsize=(10, 6))
sns.lmplot(
    data=plot_data2,
    x='shoreline',  
    y='rel_sat_diff', 
    scatter_kws={'alpha': 0.5},
    ci=None,
    height=6,
    aspect=1.67
)
plt.title('')
plt.xlabel('Shoreline NDWI Reflectance (%) of pixels where S2 > LS8')
plt.ylabel('Relative % LS8-S2 water fraction difference alont shorelines')
plt.tight_layout()
plt.show()
# %%
