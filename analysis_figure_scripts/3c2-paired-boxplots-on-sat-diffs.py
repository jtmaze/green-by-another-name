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
        n = 6 if r == 'noresample' else 3 # TODO: remove this later
        fp = f'./data/lake_area_results/{l}_resampled_{r}_area_summaries_batch{n}.csv'
        temp = pd.read_csv(fp)
        temp = temp[temp[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
        temp = temp[cols_to_keep]
        combined_list.append(temp)

# %% 3.0 Combine dataframes and calculate relative and absolute differences

combined = pd.concat(combined_list)

zone_prefixes = ['total_', 'buff_lake_', 'shoreline_']

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
plot_df = combined.copy()

plot_long = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method', 'level'],
    value_vars=['total_abs_diff', 'buff_lake_abs_diff', 'shoreline_abs_diff'],
    var_name='zone',
    value_name='abs_diff'
)
plot_long['zone'] = plot_long['zone'].str.replace('_abs_diff', '')

zone_names = {
    'total': 'Scene Total',
    'buff_lake': 'Lakes',
    'shoreline': 'Shoreline'
}

level_colors = {
    'toa': '#6a9ecf',
    'sr': '#88c999'
}

fig, ax = plt.subplots(figsize=(12, 8))

zones = ['total', 'buff_lake', 'shoreline']
resample_methods_ordered = ['bilinear30', 'noresample']
levels_ordered = ['toa', 'sr']

x_positions = []
x_labels = []
position = 0

for zone in zones:
    zone_data = plot_long[plot_long['zone'] == zone]
    
    for i, method in enumerate(resample_methods_ordered):
        for j, level in enumerate(levels_ordered):
            data = zone_data[(zone_data['resample_method'] == method) & 
                           (zone_data['level'] == level)]['abs_diff']
            
            box = ax.boxplot(
                data,
                positions=[position], 
                patch_artist=True,
                widths=0.6,
                showfliers=False
            )
            
            for patch in box['boxes']:
                patch.set_facecolor(level_colors[level])
                patch.set_alpha(0.7)
                if method == 'noresample':
                    patch.set_hatch('//')
                patch.set_edgecolor('black')
            
            for median in box['medians']:
                median.set(color='black', linewidth=2)
            
            position += 1
    
    # Add zone label position (center of the 4 boxes for this zone)
    x_positions.append(position - 2.5)
    x_labels.append(zone_names[zone])
    
    # Add spacing between zones
    position += 0.5

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylabel('Absolute % Difference (Landsat 8 - Sentinel-2)', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5)

plt.ylim(abs_y_lim)
plt.tight_layout()
plt.show()


# %%
plot_df = combined.copy()

plot_long = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method', 'level'],
    value_vars=['total_rel_diff', 'buff_lake_rel_diff', 'shoreline_rel_diff'],
    var_name='zone',
    value_name='rel_diff'
)
plot_long['zone'] = plot_long['zone'].str.replace('_rel_diff', '')

zone_names = {
    'total': 'Scene Total',
    'buff_lake': 'Lakes',
    'shoreline': 'Shoreline'
}

level_colors = {
    'toa': '#6a9ecf',
    'sr': '#88c999'
}

fig, ax = plt.subplots(figsize=(12, 8))

zones = ['total', 'buff_lake', 'shoreline']
resample_methods_ordered = ['bilinear30', 'noresample']
levels_ordered = ['toa', 'sr']

x_positions = []
x_labels = []
position = 0

for zone in zones:
    zone_data = plot_long[plot_long['zone'] == zone]
    
    for i, method in enumerate(resample_methods_ordered):
        for j, level in enumerate(levels_ordered):
            data = zone_data[
                (zone_data['resample_method'] == method) & (zone_data['level'] == level)
            ]['rel_diff']
            
            box = ax.boxplot(
                data,
                positions=[position], 
                patch_artist=True,
                widths=0.6,
                showfliers=False
            )
            
            for patch in box['boxes']:
                patch.set_facecolor(level_colors[level])
                patch.set_alpha(0.7)
                if method == 'noresample':
                    patch.set_hatch('//')
                patch.set_edgecolor('black')
            
            for median in box['medians']:
                median.set(color='black', linewidth=2)
            
            position += 1
    
    x_positions.append(position - 2.5)
    x_labels.append(zone_names[zone])
    position += 0.5

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylabel('Relative % Difference (Landsat 8 - Sentinel-2)', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5)
plt.ylim(rel_y_lim)

import matplotlib.patches as mpatches
toa_patch = mpatches.Patch(facecolor=level_colors['toa'], label='TOA', alpha=0.7, edgecolor='black')
sr_patch = mpatches.Patch(facecolor=level_colors['sr'], label='SR', alpha=0.7, edgecolor='black')
bilinear_patch = mpatches.Patch(facecolor='lightgrey', label='Bilinear 30 meters', edgecolor='black')
noresample_patch = mpatches.Patch(facecolor='lightgrey', hatch='//', label='No Resampling', edgecolor='black')

# ax.legend(
#      handles=[toa_patch, sr_patch, bilinear_patch, noresample_patch],
#      loc='upper center',
#      bbox_to_anchor=(0.5, -0.08),
#      ncol=2,
#      fontsize=18
# )

plt.tight_layout()
plt.show()

# %%
