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

valids = pd.read_csv(f'{area_dir}/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = valids[['roi', 'date']].agg('_'.join, axis=1).unique()
# %%

lake_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_0m_lake_{resample_method}_batch3.csv')
lake_data['zone'] = 'lake'
lake_data = lake_data[lake_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
print(len(lake_data))
lake_plus_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_60m_lake_{resample_method}_batch3.csv')
lake_plus_data['zone'] = 'lake_plus'
lake_plus_data = lake_plus_data[lake_plus_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
print(len(lake_plus_data))
land_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_60m_land_{resample_method}_batch3.csv')
land_data['zone'] = 'land'
land_data = land_data[land_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
print(len(land_data))
shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv')
shoreline_data['zone'] = 'shoreline'
shoreline_data = shoreline_data[shoreline_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
print(len(shoreline_data))
shoreline_tight_data = pd.read_csv(f'{reflectance_comp_dir}/sat_regression_summaries_shoreline_neg30-30_{resample_method}_batch3.csv')
# shoreline_tight_data['zone'] = 'shoreline_tight'
# shoreline_tight_data = shoreline_tight_data[shoreline_tight_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
# print(len(shoreline_tight_data))

combined = pd.concat([lake_data, lake_plus_data, land_data, shoreline_data])

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'r_squared', 'slope', 'intercept']

combined_clean = combined[cols_to_keep]

combined_wide = combined_clean.pivot(
    index=['level', 'roi', 'date', 'band_name'],
    columns='zone', 
    values='r_squared'
).reset_index()

# %% Plot the above and below frac by level
lvl = 'toa'
plot_data = combined[combined['level'] == lvl]

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
plt.title(f'{lvl} (%) of pixels with higher Sentinel-2 Reflectance')
plt.show()

# %% Make a summary table
