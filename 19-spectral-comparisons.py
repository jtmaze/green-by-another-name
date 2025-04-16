# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reflectance_comp_dir = './data/regression_summaries'
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

# %% Plot the above and below frac by level
plot_data = combined[combined['level'] == 'sr']

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