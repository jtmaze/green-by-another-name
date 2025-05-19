# %% 1.0 Libraries and directories
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name')

toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch3.csv')

valids = toa_bilinear30[['roi', 'date']].agg('_'.join, axis=1).unique()
sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch3.csv')
sr_bilinear30 = sr_bilinear30[sr_bilinear30[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
combined = pd.concat([sr_bilinear30, toa_bilinear30], ignore_index=True)

combined['main_roi'] = combined['roi'].str.split('_').str[0]

sat_color_pal = {'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'}
ac_color_pal = {'SR': '#88c999', 'TOA': '#6a9ecf'}

# %% 3.0 Plot AC's impact on Lake (+ 60m) Water Fractions by ROI

lake = combined[['roi', 'main_roi', 'date', 'level', 'total_s2_water_frac_adaptive', 'total_ls_water_frac_adaptive']]

df_wide_lake = lake.pivot(
    index=['main_roi', 'date', 'roi'], 
    columns='level', 
    values=['total_s2_water_frac_adaptive', 'total_ls_water_frac_adaptive']
).reset_index()

flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_lake.columns]
df_wide_lake.columns = flat_cols

df_wide_lake['ls_toa_sr_diff'] = df_wide_lake['total_ls_water_frac_adaptive_toa'] - df_wide_lake['total_ls_water_frac_adaptive_sr']
df_wide_lake['rel_ls_toa_sr_diff'] = (
    df_wide_lake['ls_toa_sr_diff'] / ((df_wide_lake['total_ls_water_frac_adaptive_toa'] + df_wide_lake['total_ls_water_frac_adaptive_sr']) * 0.5) * 100
)

df_wide_lake['s2_toa_sr_diff'] = df_wide_lake['total_s2_water_frac_adaptive_toa'] - df_wide_lake['total_s2_water_frac_adaptive_sr']
df_wide_lake['rel_s2_toa_sr_diff'] = (
    df_wide_lake['s2_toa_sr_diff'] / ((df_wide_lake['total_s2_water_frac_adaptive_toa'] + df_wide_lake['total_s2_water_frac_adaptive_sr']) * 0.5) * 100
)

melted_df = pd.melt(
    df_wide_lake,
    id_vars=['main_roi', 'date', 'roi'],
    value_vars=['rel_ls_toa_sr_diff', 'rel_s2_toa_sr_diff'],
    var_name='satellite_type',
    value_name='relative_difference'
)

# Map the satellite types to more readable labels
melted_df['satellite_type'] = melted_df['satellite_type'].map({
    'rel_ls_toa_sr_diff': 'Landsat 8',
    'rel_s2_toa_sr_diff': 'Sentinel-2'
})

main_roi_map = {
    'AKCP': 'Alaska Coastal Plain',
    'AND': 'Anderson Plain',
    'MRD': 'Mackenzie River Delta',
    'TUK': 'Tuktoyaktuk Peninsula',
    'YKD': 'Yukon Delta',
    'YKF': 'Yukon Flats'
}

melted_df['main_roi'] = melted_df['main_roi'].map(main_roi_map)

# Create the boxplot
plt.figure(figsize=(12, 5))
ax = plt.subplot(111)

# Group data for boxplot
grouped_data = []
positions = []
colors = []
labels = []
pos = 0

# For each region
for i, region in enumerate(melted_df['main_roi'].unique()):
    region_data = melted_df[melted_df['main_roi'] == region]
    
    # For each satellite type within the region
    for j, sat_type in enumerate(['Landsat 8', 'Sentinel-2']):
        data = region_data[region_data['satellite_type'] == sat_type]['relative_difference']
        if len(data) > 0:
            grouped_data.append(data)
            positions.append(pos)
            colors.append(sat_color_pal[sat_type])
            if i == 0:  # Only add to labels for the first region
                labels.append(sat_type)
            pos += 1
    
    pos += 0.5  # Add spacing between regions

# Create boxplot
bp = ax.boxplot(grouped_data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)

# Customize box appearance
for i, box in enumerate(bp['boxes']):
    box.set(facecolor=colors[i % 2], alpha=0.6)
    box.set(edgecolor='black')

# Set medians to black
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# Add a horizontal line at y=0
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Set x-axis ticks at the center of each group
region_positions = []
region_names = []
pos = 0
for i, region in enumerate(melted_df['main_roi'].unique()):
    center_pos = pos + 0.5
    region_positions.append(center_pos)
    region_names.append(region)
    pos += 2.5  # 2 boxes + 0.5 spacing

ax.set_xticks(region_positions)
ax.set_xticklabels(region_names)

# Add custom legend
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor=sat_color_pal['Landsat 8'], 
                 alpha=0.6, edgecolor='black', label='Landsat 8'),
    plt.Rectangle((0,0), 1, 1, facecolor=sat_color_pal['Sentinel-2'], 
                 alpha=0.6, edgecolor='black', label='Sentinel-2')
]
ax.legend(
    handles=legend_elements, 
    loc='upper center',  # Position the legend at the top center
    bbox_to_anchor=(0.5, 1.12),  # Move it below the plot (x=0.5 centers it, y=-0.15 places it below)
    ncol=2,  # Arrange legend items in 2 columns
    fontsize=12,  # Adjust font size
    frameon=True  # Add a frame around the legend
)

ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=15)

# Add padding at the bottom to make room for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Adjust bottom padding

# Label axes
plt.ylabel("Relative Difference %", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# %% Summary DataFrame

summary = melted_df.groupby(['main_roi', 'satellite_type'])['relative_difference'].agg(
    mean='mean',
    var='var',
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
).reset_index()

for col in summary.columns[2:]:
    summary[col] = summary[col].round(2)

print(summary)

# %% 3.1 Satellite's impact on Lake (PLD + 60m) Water Fractions by ROI

temp = df_wide_lake.copy()

temp['toa_ls_s2_diff'] = temp['total_ls_water_frac_adaptive_toa'] - temp['total_s2_water_frac_adaptive_toa']
temp['rel_toa_ls_s2_diff'] = (
    temp['toa_ls_s2_diff'] / ((temp['total_ls_water_frac_adaptive_toa'] + temp['total_s2_water_frac_adaptive_toa']) * 0.5) * 100
)

temp['sr_ls_s2_diff'] = temp['total_ls_water_frac_adaptive_sr'] - temp['total_s2_water_frac_adaptive_sr']
temp['rel_sr_ls_s2_diff'] = (
    temp['sr_ls_s2_diff'] / ((temp['total_ls_water_frac_adaptive_sr'] + temp['total_s2_water_frac_adaptive_sr']) * 0.5) * 100
)

# Melt the DataFrame to create a long-format dataset
melted_df = pd.melt(
    temp,
    id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
    value_vars=['rel_toa_ls_s2_diff', 'rel_sr_ls_s2_diff'],  # Columns to melt into rows
    var_name='ac_level',  # Name for the new categorical column
    value_name='relative_difference'  # Name for the values column
)

# Map the satellite types to more readable labels
melted_df['ac_level'] = melted_df['ac_level'].map({
    'rel_sr_ls_s2_diff': 'SR',
    'rel_toa_ls_s2_diff': 'TOA'
})

main_roi_map = {
    'AKCP': 'Alaska Coastal Plain',
    'AND': 'Anderson Plain',
    'MRD': 'Mackenzie River Delta',
    'TUK': 'Tuktoyaktuk Peninsula',
    'YKD': 'Yukon Delta',
    'YKF': 'Yukon Flats'
}

melted_df['main_roi'] = melted_df['main_roi'].map(main_roi_map)

# Create the boxplot
plt.figure(figsize=(12, 5))
ax = plt.subplot(111)

# Group data for boxplot
grouped_data = []
positions = []
colors = []
labels = []
pos = 0

# For each region
for i, region in enumerate(melted_df['main_roi'].unique()):
    region_data = melted_df[melted_df['main_roi'] == region]
    
    # For each AC level within the region
    for j, ac_level in enumerate(['SR', 'TOA']):
        data = region_data[region_data['ac_level'] == ac_level]['relative_difference']
        if len(data) > 0:
            grouped_data.append(data)
            positions.append(pos)
            colors.append(ac_color_pal[ac_level])
            if i == 0:  # Only add to labels for the first region
                labels.append(ac_level)
            pos += 1
    
    pos += 0.5  # Add spacing between regions

# Create boxplot
bp = ax.boxplot(grouped_data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)

# Customize box appearance
for i, box in enumerate(bp['boxes']):
    box.set(facecolor=colors[i % 2], alpha=0.6)
    box.set(edgecolor='black')

# Set medians to black
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# Add a horizontal line at y=0
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Set x-axis ticks at the center of each group
region_positions = []
region_names = []
pos = 0
for i, region in enumerate(melted_df['main_roi'].unique()):
    center_pos = pos + 0.4
    region_positions.append(center_pos)
    region_names.append(region)
    pos += 2.4  # 2 boxes + 0.5 spacing

ax.set_xticks(region_positions)
ax.set_xticklabels(region_names)

# Add custom legend
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor=ac_color_pal['SR'], 
                 alpha=0.6, edgecolor='black', label='SR'),
    plt.Rectangle((0,0), 1, 1, facecolor=ac_color_pal['TOA'], 
                 alpha=0.6, edgecolor='black', label='TOA')
]
ax.legend(
    handles=legend_elements, 
    loc='upper center',  # Position the legend at the top center
    bbox_to_anchor=(0.5, 1.12),  # Move it below the plot (x=0.5 centers it, y=-0.15 places it below)
    ncol=2,  # Arrange legend items in 2 columns
    fontsize=12,  # Adjust font size
    frameon=True  # Add a frame around the legend
)
ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=15)
# Label axes
plt.ylabel("Relative Difference %", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# %% Make a summary DataFrame

summary = melted_df.groupby(['main_roi', 'ac_level'])['relative_difference'].agg(
    mean='mean',
    var='var',
    q25= lambda x: x.quantile(0.25),
    q75= lambda x: x.quantile(0.75),
    IQR= lambda x: x.quantile(0.75) - x.quantile(0.25),
).reset_index()

for col in summary.columns[2:]:
    summary[col] = summary[col].round(2)

print(summary)

