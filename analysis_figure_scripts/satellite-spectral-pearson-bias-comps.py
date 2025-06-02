# %% 1.0 Libraries and file paths
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

reflectance_comp_dir = './data/regression_summaries'
area_dir = './data/lake_area_results'
resample_method = 'bilinear30'

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

cols_to_keep = ['level', 'roi', 'date', 'zone', 'band_name', 'r_squared', 'slope', 'intercept', 'above_frac']

combined_clean = combined[cols_to_keep]

# %% TOA plot the r-squared values
lvl = 'toa'
plot_data = combined_clean[combined_clean['level'] == lvl]

zone_label_map = {
    'lake': 'Lakes',
    'lake_plus': 'Buffered Lakes',
    'land': 'Land',
    'shoreline': 'Shoreline'
}

plot_data['zone_label'] = plot_data['zone'].map(zone_label_map)

plt.figure(figsize=(12, 10))
ax = sns.boxplot(
    data=plot_data,
    x='zone_label',
    y='r_squared',
    hue='band_name',
    palette='Set2',
    legend=False
)

plt.xlabel(None)
plt.ylim(0, 1)
ax.set_ylabel('r-squared', fontsize=20)
ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

#plt.title(f'{lvl} (%) of pixels with higher Sentinel-2 Reflectance')
plt.show()

# %% Count bad correlation images for buffered lakes 

lvl = 'toa'
bad_corr = combined_clean[
    (combined_clean['level'] == lvl) &
    (combined_clean['r_squared'] < 0.75) &
    (combined_clean['zone'] == 'shoreline')
]

bad_corr_counts = bad_corr.groupby('band_name').size().reset_index(name='count')
print(bad_corr_counts)


# %% SR plot the r-squared values

lvl = 'toa'
plot_data = combined_clean[combined_clean['level'] == lvl]

zone_label_map = {
    'lake': 'Lakes',
    'lake_plus': 'Buffered Lakes',
    'land': 'Land',
    'shoreline': 'Shoreline'
}

plot_data['zone_label'] = plot_data['zone'].map(zone_label_map)

plt.figure(figsize=(12, 10))
ax = sns.boxplot(
    data=plot_data,
    x='zone_label',
    y='above_frac',
    hue='band_name',
    palette='Set2',
    legend=False
    # Remove the legend=False to allow legend creation
)

plt.axhline(y=50, color='red', linestyle='--', linewidth=4)
plt.xlabel(None)
plt.ylim(0, 100)
ax.set_ylabel('Sentinel-2 % higher', fontsize=20)
ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

# Move the legend below the plot and increase font size
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(
#     handles=handles, 
#     labels=labels,
#     loc='upper center',          # Position at upper center
#     bbox_to_anchor=(0.5, -0.15), # Move below plot (x=0.5 centers it, y=-0.15 places it below)
#     ncol=3,                      # Spread legend items horizontally in 3 columns
#     fontsize=12,                 # Increase font size
#     frameon=True,                # Add a frame around the legend
#     title=None                   # No legend title
# )

# # Add padding at the bottom to make room for the legend
# plt.subplots_adjust(bottom=0.2)

plt.show()

# %% Make a summary table

summary = combined.groupby(['level', 'band_name']).agg(
    mean_r_squared=('r_squared', 'mean'),
    mean_above_frac=('above_frac', 'mean'),
).reset_index()


summary_zones = combined.groupby(['level', 'band_name', 'zone']).agg(
    mean_r_squared=('r_squared', 'mean'),
    mean_above_frac=('above_frac', 'mean'),
    mean_slope=('slope', 'mean'),
    sd_slope=('slope', 'std'),
    mean_intercept=('intercept', 'mean'),
    sd_intercept=('intercept', 'std'),
    above_frac_50=('above_frac', lambda x: (x > 50).sum()),
    mean_ls_marginal_percent=('ls_marginal_percent', 'mean'),
    mean_s2_marginal_percent=('s2_marginal_percent', 'mean'),
).reset_index()


# %% Plot all of the RMA lines

lvl = 'toa'
b = 'NDWI'

plot_data = combined[
    (combined['band_name'] == b) &
    (combined['level'] == lvl) &
    (combined['zone'] == 'lake_plus')
]

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

plot_data[['x_min','x_max']] = (
    plot_data['model_domain']
    .apply(parse_domain)
    .apply(pd.Series)
)

# %% Make the plot

fig, ax = plt.subplots(figsize=(6, 6))

for _, row in plot_data.iterrows():
    
    x = np.linspace(row['x_min'], row['x_max'], 100)
    y= row['intercept'] + row['slope'] * x

    if b == 'NDWI':
        line_color = '#8da0cb'  # blue from Set2
        plot_min = -.25
        plot_max = .25
    elif b == 'NIR':
        line_color = '#fc8d62'
        plot_min = 0
        plot_max = 0.42
    elif b == 'Green':
        line_color = '#66c2a5'
        plot_min = 0
        plot_max = 0.42

    ax.plot(x, y, lw=1.5, color=line_color, alpha=0.2)

x_eq = np.linspace(plot_min, plot_max, 100)
y_eq = x_eq
ax.plot(x_eq, y_eq, color='black', linewidth=4, linestyle='--')
ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)
ax.set_xlabel('Landsat 8 Reflectance', size=12)
ax.set_ylabel('Sentinel-2 Reflectance', size=12)
#ax.set_title(f'{lvl} {b} Reflectance over Buffered Lakes')

    




# %%
