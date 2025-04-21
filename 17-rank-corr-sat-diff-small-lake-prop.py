# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress

lake_size_summaries = pd.read_csv('./data/lake_size_summaries.csv')
resample_method = 'bilinear30'

sat_diff_path = f'./data/lake_area_results/toa_resampled_{resample_method}_area_summaries_batch2.csv'

# %% 2.0 Format the satellite difference data

sat_diff = pd.read_csv(sat_diff_path)
sat_diff = sat_diff[
    ['roi', 'date', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']
]

sat_diff['rel_ls_s2_diff'] = (
    (sat_diff['buff_lake_ls_water_frac_adaptive'] - sat_diff['buff_lake_s2_water_frac_adaptive']
     ) / sat_diff['buff_lake_ls_water_frac_adaptive'] * 100
)

sat_mean_diff = sat_diff.groupby(['roi'])['rel_ls_s2_diff'].mean().reset_index()
sat_mean_diff = sat_mean_diff.rename(columns={'roi': 'roi_name'})

lake_size_summaries = lake_size_summaries[
    (lake_size_summaries['lake_size'] == 'smallest') |
    (lake_size_summaries['lake_size'] == 'small')
].copy()

below_small_frac = lake_size_summaries.groupby(['roi_name'])['area_proportion'].sum().reset_index()
below_small_frac['category'] = 'small_and_smallest'
merged_below_small = pd.merge(
    sat_mean_diff,
    below_small_frac,
    how='inner',
    on='roi_name'
)

below_smallest_frac = lake_size_summaries[lake_size_summaries['lake_size'] == 'smallest'].copy()
below_smallest_frac = below_smallest_frac.groupby(['roi_name'])['area_proportion'].sum().reset_index()
below_smallest_frac['category'] = 'smallest'
merged_below_smallest = pd.merge(
    sat_mean_diff,
    below_smallest_frac,
    how='inner',
    on='roi_name'
)

roi_lake_fractions = pd.concat([merged_below_small, merged_below_smallest])


# %% 4.0 Generate stats

temp = roi_lake_fractions[roi_lake_fractions['category'] == 'smallest']

rho, pval = pearsonr(
    temp['rel_ls_s2_diff'],
    temp['area_proportion']
)
print(f'Pearson rank correlation for smallest lakes: {rho:.3f}, p-value: {pval:.3f}')

slope, intercept, r_value, p_value, std_err = linregress(
    temp['area_proportion'],
    temp['rel_ls_s2_diff']
)

print("Regression stats for smallest lakes")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Std. Error: {std_err:.4f}")

temp = roi_lake_fractions[roi_lake_fractions['category'] == 'small_and_smallest']

rho, pval = pearsonr(
    temp['rel_ls_s2_diff'],
    temp['area_proportion']
)

print(f'Pearson rank correlation for small and smallest lakes: {rho:.3f}, p-value: {pval:.3f}')

slope, intercept, r_value, p_value, std_err = linregress(
    temp['area_proportion'],
    temp['rel_ls_s2_diff']
)

print("Regression stats for small and smallest lakes")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Std. Error: {std_err:.4f}")

# %% 5.0 Plot the data with linear regression

roi_lake_fractions['category'] = roi_lake_fractions['category'].map(
    {'smallest': 'Smallest Lake %',
     'small_and_smallest': 'Small and Smallest Lake %'}
)

plt.figure(figsize=(10, 6))  # This will be ignored by lmplot
g = sns.lmplot(
    x='area_proportion', 
    y='rel_ls_s2_diff',
    data=roi_lake_fractions,
    hue='category',  # This colors points and regression lines by category
    scatter_kws={'alpha':0.7, 's':60, 'edgecolor':'k'},
    line_kws={'lw':2},
    ci=None,
    height=6,
    legend=False,
    palette={'Smallest Lake %': '#FF1493', 
             'Small and Smallest Lake %': '#008B8B'},
    aspect=1.67  # This gives approximately a 10x6 figure
)

# Add labels and title
plt.xlabel('Lake size bins area proportion (%)', fontsize=12)
plt.ylabel('Relative LS8 vs S2 Difference (%)', fontsize=12)
plt.title(f'Lake size impact on satellite discrepancies -- {resample_method}', fontsize=14)

plt.legend(loc='upper right')
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()