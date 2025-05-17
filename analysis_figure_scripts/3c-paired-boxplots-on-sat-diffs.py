# %% 1.0 Libraries and file paths
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

resample_methods = ['bilinear30', 'noresample']
levels = ['toa', 'sr']
temp = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = temp[['roi', 'date']].agg('_'.join, axis=1).unique()

rel_y_lim = (-150, 125)
abs_y_lim = (-27, 32)

# %% 2.0 Explore differences in water fraction by satellite.

combined_list = []

cols_to_keep = [
    'date', 'roi', 'level', 'resample_method', 'total_ls_water_frac_adaptive',
    'total_s2_water_frac_adaptive', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive',
    'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive',
    'shoreline_s2_water_frac_adaptive'
]

for r in resample_methods:
    for l in levels:

        fp = f'./data/lake_area_results/{l}_resampled_{r}_area_summaries_batch3.csv'
        temp = pd.read_csv(fp)
        temp = temp[temp[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
        temp = temp[cols_to_keep]
        combined_list.append(temp)

# %% 3.0 Combine dataframes and calculate relative and absolute differences

combined = pd.concat(combined_list)

zone_prefixes = ['total_', 'lake_', 'buff_lake_', 'shoreline_']

for z in zone_prefixes:

    # Calculate the Absolute satellite difference
    combined[f'{z}abs_diff'] = (
        combined[f'{z}ls_water_frac_adaptive'] - combined[f'{z}s2_water_frac_adaptive']
    )

    combined[f'{z}rel_diff'] = (
        combined[f'{z}abs_diff'] / (
            (combined[f'{z}ls_water_frac_adaptive'] + combined[f'{z}s2_water_frac_adaptive']) * 0.5
        ) * 100
    )

# %% Plot the TOA absolute lake fraction differences

plot_df = combined[combined['level'] == 'toa']

plot_long = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method'],
    value_vars=['total_abs_diff', 'lake_abs_diff', 'buff_lake_abs_diff', 'shoreline_abs_diff'],
    var_name='zone',
    value_name='abs_diff'
)
plot_long['zone'] = plot_long['zone'].str.replace('_abs_diff', '')

zone_colors = {
    'total': 'blue',
    'lake': 'orange',
    'buff_lake': 'green',
    'shoreline': 'red'
}

zone_names = {
    'total': 'Scene Total',
    'lake': 'Lakes',
    'buff_lake': 'Buffered Lakes',
    'shoreline': 'Shorelines'
}

fig, ax = plt.subplots(figsize=(8, 8))

legend_elements = []

for i, method in enumerate(resample_methods):
    method_data = plot_long[plot_long['resample_method'] == method]

    positions = [j + 0.5*i for j in range(len(zone_colors))]

    for j, (zone, color) in enumerate(zone_colors.items()):
        pos = positions[j]

        zone_data = method_data[method_data['zone'] == zone]['abs_diff']

        box = ax.boxplot(
            zone_data,
            positions=[pos], 
            patch_artist=True,
            widths=0.4,
            showfliers=False
        )

        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            if method == 'noresample':
                patch.set_hatch('//')
            patch.set_edgecolor('black')

        for median in box['medians']:
            median.set(color='black', linewidth=2)  # Set color and linewidth

        # Add to legend elements (only once per combination)
        if i == 0:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=zone))

    # Add method to legend (only once)
    if i == 0:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', label='bilinear30'))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', hatch='//', label='noresample'))

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel('Absolute (%)', fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.axhline(y=0, color='red', linestyle='--')
zone_legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                  alpha=0.5, label=zone_names[zone])
    for zone, color in zone_colors.items()
]

# Create handles for the resample method legend
method_legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='bilinear 30 meters'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', hatch='//', edgecolor='black', label='unresampled')
]

# Add the first legend for zones
legend1 = ax.legend(handles=zone_legend_elements,
                    title="Lake Position",
                    loc='upper left',
                    bbox_to_anchor=(-0.12, -0.1),
                    ncol=2,
                    edgecolor='black',
                    fontsize=14,
                    title_fontsize=14,
                    frameon=True)
ax.add_artist(legend1)  # Keep the first legend

# Add the second legend for resample methods
legend2 = ax.legend(handles=method_legend_elements,
                    title="Resample Method",
                    loc='upper right',
                    bbox_to_anchor=(0.95, -0.1),
                    ncol=1,
                    edgecolor='black',
                    fontsize=14,
                    title_fontsize=14,
                    frameon=True)
plt.tight_layout()
fig.subplots_adjust(bottom=0.3) 

plt.ylim(abs_y_lim)
plt.show()


# %%

plot_df = combined[combined['level'] == 'toa']

plot_long = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method'],
    value_vars=['total_rel_diff', 'lake_rel_diff', 'buff_lake_rel_diff', 'shoreline_rel_diff'],
    var_name='zone',
    value_name='rel_diff'
)
plot_long['zone'] = plot_long['zone'].str.replace('_rel_diff', '')

zone_colors = {
    'total': 'blue',
    'lake': 'orange',
    'buff_lake': 'green',
    'shoreline': 'red'
}

zone_names = {
    'total': 'Scene Total',
    'lake': 'Lakes',
    'buff_lake': 'Buffered Lakes',
    'shoreline': 'Shorelines'
}

fig, ax = plt.subplots(figsize=(8, 8))

legend_elements = []

for i, method in enumerate(resample_methods):
    method_data = plot_long[plot_long['resample_method'] == method]

    positions = [j + 0.5*i for j in range(len(zone_colors))]

    for j, (zone, color) in enumerate(zone_colors.items()):
        pos = positions[j]

        zone_data = method_data[method_data['zone'] == zone]['rel_diff']

        box = ax.boxplot(
            zone_data,
            positions=[pos], 
            patch_artist=True,
            widths=0.4,
            showfliers=False
        )

        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            if method == 'noresample':
                patch.set_hatch('//')
            patch.set_edgecolor('black')

        for median in box['medians']:
            median.set(color='black', linewidth=2)  # Set color and linewidth

        # Add to legend elements (only once per combination)
        if i == 0:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=zone))

    # Add method to legend (only once)
    if i == 0:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', label='bilinear30'))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', hatch='//', label='noresample'))

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel('Relative (%)', fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.axhline(y=0, color='red', linestyle='--')

plt.ylim(rel_y_lim)
plt.show()

# %%
