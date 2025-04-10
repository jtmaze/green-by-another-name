# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress

lake_size_summaries = pd.read_csv('./data/lake_size_summaries.csv')
resample_method = 'bilinear30'

toa_path = f'./data/lake_area_results/toa_resampled_{resample_method}_area_summaries_batch2.csv'
sr_path = f'./data/lake_area_results/sr_resampled_{resample_method}_area_summaries_batch2.csv'
toa_data = pd.read_csv(toa_path)
sr_data = pd.read_csv(sr_path)

cols_to_keep = [
    'date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive',
    'smallest_buff_lake_ls_water_frac_adaptive', 'smallest_buff_lake_s2_water_frac_adaptive',
    'small_buff_lake_ls_water_frac_adaptive', 'smallest_buff_lake_s2_water_frac_adaptive'
    'medium_buff_lake_ls_water_frac_adaptive', 'medium_buff_lake_s2_water_frac_adaptive',
    'large_buff_lake_ls_water_frac_adaptive', 'large_buff_lake_s2_water_frac_adaptive',
]

toa_data = toa_data[cols_to_keep]
sr_data = sr_data[cols_to_keep]

# %%

toa_data['']

# %%

combined = pd.concat([toa_data, sr_data], ignore_index=True)


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

# %% 3.0 Format the lake size summaries

lake_size_summaries = lake_size_summaries[
    (lake_size_summaries['lake_size'] == 'smallest')
].copy()

below_small_frac = lake_size_summaries.groupby(['roi_name'])['area_proportion'].sum().reset_index()

# %% 4.0 Merge the dataframes

merged_df = pd.merge(
    sat_mean_diff,
    below_small_frac,
    how='inner',
    on=['roi_name']
)

rho, pval = pearsonr(
    merged_df['rel_ls_s2_diff'],
    merged_df['area_proportion']
)
print(f'Pearson rank correlation: {rho:.3f}, p-value: {pval:.3f}')

# %% 5.0 Plot the data with linear regression

slope, intercept, r_value, p_value, std_err = linregress(
    merged_df['area_proportion'],
    merged_df['rel_ls_s2_diff']
)

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Std. Error: {std_err:.4f}")

plt.figure(figsize=(10, 6))
ax = sns.regplot(
    x='area_proportion', 
    y='rel_ls_s2_diff',
    data=merged_df,
    scatter_kws={'alpha':0.7, 's':60, 'edgecolor':'k'},
    line_kws={'color':'red', 'lw':2},
    ci=None
)

# Add labels and title
plt.xlabel('Proportion of Smallest Lakes (< 0.05 km²) as (%) total lake area', fontsize=12)
plt.ylabel('Relative (LS8-S2) / LS8 Difference (%)', fontsize=12)
plt.title(f'Smallest Lake Proportions Relationship with Satellite Discrepancy -- {resample_method}', fontsize=14)

# Add text annotations for each point (ROI names)
for i, row in merged_df.iterrows():
    plt.annotate(
        row['roi_name'], 
        xy=(row['area_proportion'], row['rel_ls_s2_diff']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9
    )

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()