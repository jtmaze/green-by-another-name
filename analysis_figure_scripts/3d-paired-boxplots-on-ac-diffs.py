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
    'total_s2_water_frac_adaptive', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive',
    'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive',
    'shoreline_s2_water_frac_adaptive'
]

calc_df_list = []

for r in resample_methods:
    level_list = []
    for l in levels:

        fp = f'./data/lake_area_results/{l}_resampled_{r}_area_summaries_batch3.csv'
        temp = pd.read_csv(fp)
        temp = temp[temp[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
        temp = temp[cols_to_keep]
        level_list.append(temp)

    level_df = pd.concat(level_list)

    zone_prefixes = ['total_', 'lake_', 'buff_lake_', 'shoreline_']

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

plot_var = 'ls_rel_diff'
y_var = 'rel'
rel_y_lim = (-45, 210)
abs_y_lim = (-10, 55)

zone_colors = {
    'total': 'blue',
    'lake': 'orange',
    'buff': 'green',
    'shoreline': 'red'
}

zone_names = {
    'total': 'Scene Total',
    'lake': 'Lakes',
    'buff': 'Buffered Lakes',
    'shoreline': 'Shorelines'
}

fig, ax = plt.subplots(figsize=(8, 8))

legend_elements = []

for i, method in enumerate(resample_methods):

    method_data = combined[combined['resample_method'] == method]

    positions = [j + 0.5*i for j in range(len(zone_colors))]

    for j, (zone, color) in enumerate(zone_colors.items()):
        pos = positions[j]

        zone_data = method_data[method_data['zone'] == zone][f'{plot_var}']

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
            median.set(color='black', linewidth=2)

         # Add to legend elements (only once per combination)
        if i == 0:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=zone))
    
    if i == 0:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', label='bilinear30'))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', hatch='//', label='noresample'))

ax.set_xticks([])
ax.set_xticklabels([])
# NOTE: This shit is pretty hacky
if y_var == 'abs':
    ax.set_ylabel('Absolute (%)', fontsize=16)
    lim = abs_y_lim
elif y_var == 'rel':
    ax.set_ylabel('Relative (%)', fontsize=16)
    lim = rel_y_lim

ax.tick_params(axis='y', labelsize=14)
ax.axhline(y=0, color='red', linestyle='--')
# zone_legend_elements = [
#     plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
#                   alpha=0.5, label=zone_names[zone])
#     for zone, color in zone_colors.items()
# ]

# # Create handles for the resample method legend
# method_legend_elements = [
#     plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='bilinear 30 meters'),
#     plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', hatch='//', edgecolor='black', label='unresampled')
# ]

# # Add the first legend for zones
# legend1 = ax.legend(handles=zone_legend_elements,
#                     title="Lake Position",
#                     loc='upper left',
#                     bbox_to_anchor=(-0.12, -0.1),
#                     ncol=2,
#                     edgecolor='black',
#                     fontsize=14,
#                     title_fontsize=14,
#                     frameon=True)
# ax.add_artist(legend1)  # Keep the first legend

# # Add the second legend for resample methods
# legend2 = ax.legend(handles=method_legend_elements,
#                     title="Resample Method",
#                     loc='upper right',
#                     bbox_to_anchor=(0.95, -0.1),
#                     ncol=1,
#                     edgecolor='black',
#                     fontsize=14,
#                     title_fontsize=14,
#                     frameon=True)
# plt.tight_layout()
# fig.subplots_adjust(bottom=0.3) 

plt.ylim(lim)
plt.show()          



# %%
