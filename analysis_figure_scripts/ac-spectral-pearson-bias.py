# %% Libraries and file paths

import os 
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

reflectance_comp_dir = './data/regression_summaries'
area_dir = './data/lake_area_results'
resample_method = 'bilinear30'

valids = pd.read_csv(f'{area_dir}/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = valids[['roi', 'date']].agg('_'.join, axis=1).unique()


# %% Read the data

lake_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_0m_lake_{resample_method}_batch3.csv')
lake_data['zone'] = 'lake'
lake_data = lake_data[lake_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

lake_plus_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_60m_lake_{resample_method}_batch3.csv')
lake_plus_data['zone'] = 'lake_plus'
lake_plus_data = lake_plus_data[lake_plus_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

land_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_60m_land_{resample_method}_batch3.csv')
land_data['zone'] = 'land'
land_data = land_data[land_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv')
shoreline_data['zone'] = 'shoreline'
shoreline_data = shoreline_data[shoreline_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

combined = pd.concat([lake_data, lake_plus_data, land_data, shoreline_data], ignore_index=True)

# %% Plot the pearson coefficients

cols_to_keep = ['satellite', 'roi', 'date', 'zone', 'band_name', 'r_squared', 'slope', 'intercept', 'above_frac']
combined_clean = combined[cols_to_keep]

sat = 'Sentinel2' # Landsat8 or Sentinel2
box_plot_data = combined[combined['satellite'] == sat]

zone_label_map = {
    'lake': 'Lakes',
    'lake_plus': 'Buffered Lakes',
    'land': 'Land',
    'shoreline': 'Shoreline'
}

box_plot_data['zone_label'] = box_plot_data['zone'].map(zone_label_map)

plt.figure(figsize=(12, 5))
ax = sns.boxplot(
    data=box_plot_data,
    x='zone_label',
    y='above_frac',
    hue='band_name',
    palette='Set2',
    legend=False
)
ax.set_xlabel(None)
ax.set_ylabel('% of TOA pixels higher', fontsize=14)
plt.ylim(0, 100)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
plt.axhline(y=50, color='red', linestyle='--')
#plt.title(f'{sat} % of pixels with higher SR reflectance')
plt.show()

# %% Summary tables

summary1 = combined.groupby(['satellite', 'band_name']).agg(
    mean_r_squared=('r_squared', 'mean'),
    mean_above_frac=('above_frac', 'mean'),
).reset_index()

summary2 = combined.groupby(['satellite', 'band_name', 'zone']).agg(
    mean_r_squared=('r_squared', 'mean'),
    mean_above_frac=('above_frac', 'mean'),
    mean_slope=('slope', 'mean'),
    sd_slope=('slope', 'std'),
    mean_intercept=('intercept', 'mean'),
    sd_intercept=('intercept', 'std'),
    above_frac_50 =('above_frac', lambda x: (x > 50).sum()),
).reset_index()


# %% Plot the RMA lines
sat = 'Sentinel2' # Landsat8 or Sentinel2
b = 'NDWI' # Green, NIR, or NDWI


plot_data = combined[
    (combined['band_name'] == b) &
    (combined['satellite'] == sat) &
    (combined['zone'] == 'lake_plus')
].copy()

def parse_domain(d):
    """
    Return (xmin, xmax) as floats from a variety of formats:
        ('0.01', '0.13')
        (np.float32(0.01), np.float32(0.13))
        "(np.float32(0.01), np.float32(0.13))"
    """
    # already a sequence of numbers → just cast
    if isinstance(d, (tuple, list)):
        return float(d[0]), float(d[1])

    # string: pull the first two numbers you see
    if isinstance(d, str):
        nums = re.findall(r'[-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?', d)
        nums = [n for n in nums if n not in ("", "+", "-")]  # drop empty matches
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])

    raise ValueError(f"Un‑parsable model_domain: {d!r}")

# Add two numeric columns up‑front for convenience
plot_data[['x_min','x_max']] = (
    plot_data['model_domain']
    .apply(parse_domain)
    .apply(pd.Series)
)


# %% Make the plot

fig, ax = plt.subplots(figsize=(6, 6))

for _, row in plot_data.iterrows():

    x = np.linspace(row['x_min'], row['x_max'], 100)
    y = row['intercept'] + row['slope'] * x

    if b == 'NDWI':
        line_color = '#8da0cb'  # blue from Set2
    elif b == 'Green':
        line_color = '#66c2a5'  # teal from Set
    elif b == 'NIR':
        line_color = '#fc8d62'  # orange from Set2

    ax.plot(x, y, lw=1.5, color=line_color, alpha=0.2)

# Create x and y values for the 1:1 line
if b == 'NDWI':
    plot_min = -1
    plot_max = 1
else: 
    plot_min = 0
    plot_max = 0.25

x_eq = np.linspace(plot_min, plot_max, 100)
y_eq = x_eq  # 1:1 line means slope = 1, intercept = 0

# Plot the 1:1 line on top of everything
ax.plot(x_eq, y_eq, color='black', linestyle='--', linewidth=4, label='1:1 line')
ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)
ax.set_xlabel(f'TOA Reflectance', size=12)
ax.set_ylabel(f'SR Reflectance', size=12)
#ax.set_title(f'{sat} {b} AC RMA over Buffered Lakes')

# %%
