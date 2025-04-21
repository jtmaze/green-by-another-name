# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
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

# %% 2.0 Read and reformat the area data

toa_data = toa_data.rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_toa_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_toa_frac'
    }
)
sr_data = sr_data.rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'ls_sr_frac',
        'buff_lake_s2_water_frac_adaptive': 's2_sr_frac'
    }
)

# Then keep only the columns you need
toa_cols = ['date', 'roi', 'ls_toa_frac', 's2_toa_frac']
sr_cols = ['date', 'roi', 'ls_sr_frac', 's2_sr_frac']

toa_data = toa_data[toa_cols]
sr_data = sr_data[sr_cols]

print(len(toa_data), len(sr_data))


combined_areas = pd.merge(left=toa_data, right=sr_data, how='inner', on=['date', 'roi'])
print(len(combined_areas))

combined_areas['abs_ls_ac_diff'] = combined_areas['ls_toa_frac'] - combined_areas['ls_sr_frac']
combined_areas['rel_ls_ac_diff'] = combined_areas['abs_ls_ac_diff'] / combined_areas['ls_toa_frac'] * 100
combined_areas['abs_s2_ac_diff'] = combined_areas['s2_toa_frac'] - combined_areas['s2_sr_frac']
combined_areas['rel_s2_ac_diff'] = combined_areas['abs_s2_ac_diff'] / combined_areas['s2_toa_frac'] * 100


# Scatter plot for ac differences by image target region 
slope, intercept, r_value, p_value, std_err = linregress(
    combined_areas['rel_ls_ac_diff'],
    combined_areas['rel_s2_ac_diff']
)

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Std. Error: {std_err:.4f}")

plt.figure(figsize=(10, 8))
# Create scatter plot
combined_areas['roi_main'] = combined_areas['roi'].apply(lambda x: x.split('_')[0])

sns.scatterplot(
    data=combined_areas,
    x='rel_ls_ac_diff',
    y='rel_s2_ac_diff',
    s=80,  # Marker size
    alpha=0.9,
    edgecolor='black',
    #hue='roi_main'
)

# Add trendline
x_vals = np.array(plt.xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--', color='red', linewidth=2, 
         label=f'y = {slope:.3f}x + {intercept:.3f}, R² = {r_value**2:.3f}')
plt.legend()

# Add text labels for each point
# for i, row in ac_area_diffs.iterrows():
#     plt.text(
#         row['rel_ls_ac_diff'] + 0.5,  # Slight offset for readability
#         row['rel_s2_ac_diff'],
#         row['roi'],
#         fontsize=9
#     )

# Add titles and labels
plt.title('Atmospheric Correction Effects on Landsat vs Sentinel-2', fontsize=14)
plt.xlabel('Relative Landsat TOA-SR Difference (%)', fontsize=12)
plt.ylabel('Relative Sentinel-2 TOA-SR Difference (%)', fontsize=12)
# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-axis limits after equal aspect ratio
# plt.xlim(-10, 60)
plt.tight_layout()
plt.show()

# %% 3.0 Read and reformat the roi lake size data and merge with area discrepancy data

ac_area_diffs = combined_areas.groupby(['roi']).agg({
    'rel_ls_ac_diff': 'mean',
    'rel_s2_ac_diff': 'mean'
}).reset_index()

lake_size_summaries.rename(
    columns={'roi_name': 'roi'},
    inplace=True
)

below_small_frac = lake_size_summaries[
    (lake_size_summaries['lake_size'] == 'smallest') |
    (lake_size_summaries['lake_size'] == 'small')
]

below_small_frac = below_small_frac.groupby(['roi'])['area_proportion'].sum().reset_index()
below_small_frac['category'] = 'small_and_smallest'
merged_below_small = pd.merge(
    ac_area_diffs,
    below_small_frac,
    how='inner',
    on='roi'
)

below_smallest_frac = lake_size_summaries[lake_size_summaries['lake_size'] == 'smallest'].copy()
below_smallest_frac = below_smallest_frac.groupby(['roi'])['area_proportion'].sum().reset_index()
below_smallest_frac['category'] = 'smallest'
merged_below_smallest = pd.merge(
    ac_area_diffs,
    below_smallest_frac,
    how='inner',
    on='roi'
)

roi_lake_fractions = pd.concat([merged_below_small, merged_below_smallest])

# %% 4.0 Generate stats 

# ----- SMALLEST LAKES CATEGORY -----
temp = roi_lake_fractions[roi_lake_fractions['category'] == 'smallest'].copy()

# Statistics for Landsat (rel_ls_ac_diff)
rho_ls, pval_ls = pearsonr(temp['area_proportion'], temp['rel_ls_ac_diff'])
slope_ls, intercept_ls, r_value_ls, p_value_ls, std_err_ls = linregress(
    temp['area_proportion'], temp['rel_ls_ac_diff']
)

# Statistics for Sentinel-2 (rel_s2_ac_diff)
rho_s2, pval_s2 = pearsonr(temp['area_proportion'], temp['rel_s2_ac_diff'])
slope_s2, intercept_s2, r_value_s2, p_value_s2, std_err_s2 = linregress(
    temp['area_proportion'], temp['rel_s2_ac_diff']
)

print("\n----- STATISTICS FOR SMALLEST LAKES CATEGORY -----")
print("LANDSAT:")
print(f"  Pearson correlation: {rho_ls:.3f}, p-value: {pval_ls:.3f}")
print(f"  Linear regression - Slope: {slope_ls:.4f}, Intercept: {intercept_ls:.4f}")
print(f"  R-squared: {r_value_ls**2:.4f}, P-value: {p_value_ls:.4f}, Std. Error: {std_err_ls:.4f}")

print("\nSENTINEL-2:")
print(f"  Pearson correlation: {rho_s2:.3f}, p-value: {pval_s2:.3f}")
print(f"  Linear regression - Slope: {slope_s2:.4f}, Intercept: {intercept_s2:.4f}")
print(f"  R-squared: {r_value_s2**2:.4f}, P-value: {p_value_s2:.4f}, Std. Error: {std_err_s2:.4f}")

# ----- SMALL AND SMALLEST LAKES CATEGORY -----
temp = roi_lake_fractions[roi_lake_fractions['category'] == 'small_and_smallest'].copy()

# Statistics for Landsat (rel_ls_ac_diff)
rho_ls, pval_ls = pearsonr(temp['area_proportion'], temp['rel_ls_ac_diff'])
slope_ls, intercept_ls, r_value_ls, p_value_ls, std_err_ls = linregress(
    temp['area_proportion'], temp['rel_ls_ac_diff']
)

# Statistics for Sentinel-2 (rel_s2_ac_diff)
rho_s2, pval_s2 = pearsonr(temp['area_proportion'], temp['rel_s2_ac_diff'])
slope_s2, intercept_s2, r_value_s2, p_value_s2, std_err_s2 = linregress(
    temp['area_proportion'], temp['rel_s2_ac_diff']
)

print("\n----- STATISTICS FOR SMALL AND SMALLEST LAKES CATEGORY -----")
print("LANDSAT:")
print(f"  Pearson correlation: {rho_ls:.3f}, p-value: {pval_ls:.3f}")
print(f"  Linear regression - Slope: {slope_ls:.4f}, Intercept: {intercept_ls:.4f}")
print(f"  R-squared: {r_value_ls**2:.4f}, P-value: {p_value_ls:.4f}, Std. Error: {std_err_ls:.4f}")

print("\nSENTINEL-2:")
print(f"  Pearson correlation: {rho_s2:.3f}, p-value: {pval_s2:.3f}")
print(f"  Linear regression - Slope: {slope_s2:.4f}, Intercept: {intercept_s2:.4f}")
print(f"  R-squared: {r_value_s2**2:.4f}, P-value: {p_value_s2:.4f}, Std. Error: {std_err_s2:.4f}")

# %%

plot_df = roi_lake_fractions.copy()

plot_df['category'] = plot_df['category'].map(
    {'smallest': 'Smallest Lake %',
     'small_and_smallest': 'Small and Smallest Lake %'}
)

g = sns.lmplot(
    x='area_proportion', 
    y='rel_ls_ac_diff',
    data=plot_df,
    hue='category',  # This colors points and regression lines by category
    scatter_kws={'alpha':0.7, 's':60, 'edgecolor':'k'},
    line_kws={'lw':2},
    ci=None,
    height=6,
    legend=False,
    palette={'Smallest Lake %': '#FF1493', 
             'Small and Smallest Lake %': '#008B8B'},
    aspect=1  # This gives approximately a 10x6 figure
)

# Add labels and title
plt.xlabel('Lake size bins area proportion (%)', fontsize=12)
plt.ylabel('Relative TOA vs SR difference (%)', fontsize=12)
plt.title(f'Lake size impact on LS8 AC discrepancies -- {resample_method}', fontsize=14)

plt.legend(loc='upper right')
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()