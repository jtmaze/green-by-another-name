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

# %%

cols_to_keep = [
    'date', 'roi', 'level', 'resample_method', 'total_ls_water_frac_adaptive',
    'total_s2_water_frac_adaptive', #'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive',
    'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive',
    'shoreline_s2_water_frac_adaptive'
]

calc_df_list = []

for r in resample_methods:
    level_list = []
    for l in levels:
        n = 6 if r == 'noresample' else 3 # TODO: remove this later
        fp = f'./data/lake_area_results/{l}_resampled_{r}_area_summaries_batch{n}.csv'
        temp = pd.read_csv(fp)
        temp = temp[temp[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
        temp = temp[cols_to_keep]
        level_list.append(temp)

    level_df = pd.concat(level_list)

    zone_prefixes = ['total_', 'buff_lake_', 'shoreline_']

    for z in zone_prefixes:

        levels_wide = level_df.pivot_table(
            index=['date', 'roi'],
            columns=['level'],
            values=[f'{z}ls_water_frac_adaptive', f'{z}s2_water_frac_adaptive']
        ).reset_index()
        # flatten the multi-level index
        levels_wide.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in levels_wide.columns]

        levels_wide.rename(
        columns={
            f'{z}ls_water_frac_adaptive_sr': 'ls_sr',
            f'{z}ls_water_frac_adaptive_toa': 'ls_toa',
            f'{z}s2_water_frac_adaptive_sr': 's2_sr',
            f'{z}s2_water_frac_adaptive_toa': 's2_toa'
        }, inplace=True)

        levels_wide['ls_abs_diff'] = (levels_wide['ls_toa'] - levels_wide['ls_sr'])
        levels_wide['ls_rel_diff'] = (
            levels_wide['ls_abs_diff'] / ((levels_wide['ls_toa'] + levels_wide['ls_sr']) * 0.5) * 100
        )

        levels_wide['s2_abs_diff'] = (levels_wide['s2_toa'] - levels_wide['s2_sr'])
        levels_wide['s2_rel_diff'] = (
            levels_wide['s2_abs_diff'] / ((levels_wide['s2_toa'] + levels_wide['s2_sr']) * 0.5) * 100
        )

        levels_wide['zone'] = z.split('_')[0]
        levels_wide['resample_method'] = r

        keep_cols = ['date', 'roi', 'zone', 'resample_method', 'ls_abs_diff', 'ls_rel_diff', 's2_abs_diff', 's2_rel_diff']

        calc_df = levels_wide[keep_cols].copy()
        calc_df_list.append(calc_df)

# %% Concatonate

combined = pd.concat(calc_df_list)

# %% Absolute Landsat 8 plot
plot_df = combined.copy()

# Create long format for plotting
plot_long = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method', 'zone'],
    value_vars=['ls_abs_diff', 's2_abs_diff'],
    var_name='satellite',
    value_name='abs_diff'
)
plot_long['satellite'] = plot_long['satellite'].str.replace('_abs_diff', '')

zone_names = {
    'total': 'Scene Total',
    #'lake': 'Lakes',
    'buff': 'Lakes',
    'shoreline': 'Shoreline'
}

satellite_colors = {
    'ls': '#ff9933',  # Orange for Landsat
    's2': '#9370DB'   # Blue for Sentinel-2
}

fig, ax = plt.subplots(figsize=(12, 8))

zones = ['total', 'buff', 'shoreline']
resample_methods_ordered = ['bilinear30', 'noresample']
satellites_ordered = ['ls', 's2']

x_positions = []
x_labels = []
position = 0

for zone in zones:
    zone_data = plot_long[plot_long['zone'] == zone]
    
    for i, method in enumerate(resample_methods_ordered):
        for j, satellite in enumerate(satellites_ordered):
            data = zone_data[(zone_data['resample_method'] == method) & 
                           (zone_data['satellite'] == satellite)]['abs_diff']
            
            box = ax.boxplot(
                data,
                positions=[position], 
                patch_artist=True,
                widths=0.6,
                showfliers=False
            )
            
            for patch in box['boxes']:
                patch.set_facecolor(satellite_colors[satellite])
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
ax.set_ylabel('Absolute % Difference (TOA - SR)', fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5)

plt.ylim((-10, 55))
plt.tight_layout()
plt.show()

# %%
# %% Relative differences plot (TOA - SR)

plot_long_rel = pd.melt(
    plot_df, 
    id_vars=['date', 'roi', 'resample_method', 'zone'],
    value_vars=['ls_rel_diff', 's2_rel_diff'],
    var_name='satellite',
    value_name='rel_diff'
)
plot_long_rel['satellite'] = plot_long_rel['satellite'].str.replace('_rel_diff', '')

fig, ax = plt.subplots(figsize=(12, 8))

x_positions = []
x_labels = []
position = 0

for zone in zones:
    zone_data = plot_long_rel[plot_long_rel['zone'] == zone]
    
    for i, method in enumerate(resample_methods_ordered):
        for j, satellite in enumerate(satellites_ordered):
            data = zone_data[(zone_data['resample_method'] == method) & 
                           (zone_data['satellite'] == satellite)]['rel_diff']
            
            box = ax.boxplot(
                data,
                positions=[position], 
                patch_artist=True,
                widths=0.6,
                showfliers=False
            )
            
            for patch in box['boxes']:
                patch.set_facecolor(satellite_colors[satellite])
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
ax.set_ylabel('Relative % Difference (TOA - SR)', fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5)
plt.ylim((-45, 210))

import matplotlib.patches as mpatches
ls_patch = mpatches.Patch(facecolor=satellite_colors['ls'], label='Landsat 8', alpha=0.7, edgecolor='black')
s2_patch = mpatches.Patch(facecolor=satellite_colors['s2'], label='Sentinel-2', alpha=0.7, edgecolor='black')
bilinear_patch = mpatches.Patch(facecolor='lightgrey', label='Bilinear 30 meters', edgecolor='black')
noresample_patch = mpatches.Patch(facecolor='lightgrey', hatch='//', label='No Resample', edgecolor='black')

# ax.legend(
#     handles=[ls_patch, s2_patch, bilinear_patch, noresample_patch],
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.08),
#     ncol=2,
#     fontsize=18
# )

plt.tight_layout()
plt.show()


# %%
