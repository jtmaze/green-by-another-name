# %% Libraries and file paths

import os 
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

reflectance_comp_dir = 'D:/thesis_data/regression_summaries'
area_dir = 'D:/thesis_data/lake_area_results'
resample_method = 'bilinear30'

valids = pd.read_csv(f'{area_dir}/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = valids[['roi', 'date']].agg('_'.join, axis=1).unique()

# %% Read the data

lake_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_60m_lake_{resample_method}_batch3.csv')
lake_data['zone'] = 'lake'
lake_data = lake_data[lake_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

land_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_60m_land_{resample_method}_batch3.csv')
land_data['zone'] = 'land'
land_data = land_data[land_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/AC_regression_summaries_shoreline_neg60-60_{resample_method}_batch3.csv')
shoreline_data['zone'] = 'shoreline'
shoreline_data = shoreline_data[shoreline_data[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

combined = pd.concat([lake_data, land_data, shoreline_data], ignore_index=True)

# %% Plot the pearson coefficients

cols_to_keep = ['satellite', 'roi', 'date', 'zone', 'band_name', 'r_squared', 'slope', 'intercept', 'above_frac']
combined_clean = combined[cols_to_keep]

sat = 'Sentinel2' # Landsat8 or Sentinel2
box_plot_data = combined.copy()
box_plot_data = box_plot_data[box_plot_data['band_name'] == 'NDWI']

zone_label_map = {
    'lake': 'Lakes',
    'land': 'Land',
    'shoreline': 'Shoreline'
}

satellite_label_map = {
    'Landsat8': 'Landsat 8',
    'Sentinel2': 'Sentinel-2'
}

box_plot_data['satellite_label'] = box_plot_data['satellite'].map(satellite_label_map)
box_plot_data['zone_label'] = box_plot_data['zone'].map(zone_label_map)

plt.figure(figsize=(10, 10))
ax = sns.boxplot(
    data=box_plot_data,
    x='zone_label',
    y='above_frac',
    hue='satellite_label',
    palette={'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'},
    width=0.7
)

ax.set_xlabel(None)
ax.set_ylabel('% SR pixels higher', fontsize=20)
plt.ylim(0, 100)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=22)
yticks = ax.get_yticks()
ax.set_yticklabels([f"{y:.0f}" for y in yticks], fontsize=18)
plt.axhline(y=50, color='red', linestyle='--', linewidth=4)
#plt.title(f'{sat} % of pixels with higher SR reflectance')

# Make legend larger
legend = ax.legend(fontsize=22, loc='best', frameon=True)
legend.set_title('')  # Remove title if you don't need it

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
).reset_index()

summary2 = summary2.to_csv(
    f'D:/thesis_data/summary_data_for_SC/AC_pixel_comparsion_summary_stats.csv', 
    index=False
)
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
