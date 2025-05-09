# %% 1.0 Libraries and file paths
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

reflectance_comp_dir = './data/regression_summaries'
area_dir = './data/lake_area_results'
resample_method = 'cubic30'
# %%

lake_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_0m_lake_{resample_method}_batch3.csv')
lake_data['zone'] = 'lake'
print(len(lake_data))
lake_plus_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_60m_lake_{resample_method}_batch3.csv')
lake_plus_data['zone'] = 'lake_plus'
print(len(lake_plus_data))
land_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_60m_land_{resample_method}_batch3.csv')
land_data['zone'] = 'land'
print(len(land_data))
shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv')
shoreline_data['zone'] = 'shoreline'
print(len(shoreline_data))
shoreline_tight_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_shoreline_neg30-30_{resample_method}_batch3.csv')
shoreline_tight_data['zone'] = 'shoreline_tight'


combined = pd.concat([lake_data, lake_plus_data, land_data, shoreline_data, shoreline_tight_data])

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'r_squared', 'slope']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values='r_squared'
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
plt.xlabel(None)
plt.ylim(0, 100)
plt.ylabel('(%) of Image Pixels')
plt.title('SR (%) of pixels with higher Sentinel-2 Reflectance')
plt.show()

# %% Read the area data

toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch3.csv')
print(toa_data.columns)
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch3.csv')

lake_zone = 'small_buff_lake'

cols_to_keep =['date', 'roi', 'level', f'{lake_zone}_ls_water_frac_adaptive',
               f'{lake_zone}_s2_water_frac_adaptive', 'pld_plus_valid_frac']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        f'{lake_zone}_ls_water_frac_adaptive': 'ls_water_frac',
        f'{lake_zone}_s2_water_frac_adaptive': 's2_water_frac',
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        f'{lake_zone}_ls_water_frac_adaptive': 'ls_water_frac',
        f'{lake_zone}_s2_water_frac_adaptive': 's2_water_frac',
    }
).copy()

combined_area = pd.concat([toa_data, sr_data])
#combined_area = combined_area[combined_area['pld_plus_valid_frac'] >= 50]
combined_area = combined_area.drop(columns=['pld_plus_valid_frac'])

combined_area['abs_sat_diff'] = combined_area['ls_water_frac'] - combined_area['s2_water_frac']
combined_area['rel_sat_diff'] = (
    combined_area['abs_sat_diff'] / ((combined_area['ls_water_frac'] + combined_area['s2_water_frac']) * 0.5) * 100
)

toa_area = combined_area[combined_area['level'] == 'toa']
sr_area = combined_area[combined_area['level'] == 'sr']

# %% 

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'above_frac', 'slope']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values=['above_frac', 'slope']
).reset_index()


flat_columns = []
for col in combined_wide.columns:
    if col[1] == '':
        flat_columns.append(col[0])
    else:
        flat_columns.append(f"{col[1]}_{col[0]}")

# Directly assign the new column names
combined_wide.columns = flat_columns


# combined_wide['land_lake_diff'] = combined_wide['land'] - combined_wide['lake_plus']
# combined_wide['land_lake_ratio'] = combined_wide['land'] / combined_wide['lake_plus']

area_reflectance = pd.merge(toa_area, combined_wide, on=['level', 'roi', 'date'], how='left')

# %% 
plot_data2 = area_reflectance.copy()
plot_data2 = plot_data2[plot_data2['band_name'] == 'NDWI']

plot_data2['roi_main'] = plot_data2['roi'].apply(lambda x: x.split('_')[0])
print(plot_data2.columns)
# %%
# Define independent and dependent variables
x_var = 'shoreline_above_frac'
y_var = 'rel_sat_diff'


plot_data2 = plot_data2[plot_data2[x_var] > 50]
plot_data2 = plot_data2[
    (plot_data2[y_var] < 5) &
    (plot_data2[y_var] > -20)
]

# Perform linear regression using the defined variables
slope, intercept, r_value, p_value, std_err = linregress(
    plot_data2[x_var],
    plot_data2[y_var]
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
    x=x_var,  
    y=y_var, 
    scatter_kws={'alpha': 0.5},
    ci=None,
    height=6,
    aspect=1.67
)
plt.title('')
plt.xlabel('NIR Reflectance (%) of PLD+60m pixels where S2 > LS8')
plt.ylabel('S2 Relative water fraction bias')
plt.tight_layout()
plt.show()
# %%
