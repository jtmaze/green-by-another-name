# %% 1.0 Libraries and directories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch2.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch2.csv')
combined = pd.concat([sr_bilinear30, toa_bilinear30], ignore_index=True)
valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data'].copy()
valid['main_roi'] = valid['roi'].apply(lambda x: x.split('_')[0])

sat_color_pal = {'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'}
ac_color_pal = {'SR': '#88c999', 'TOA': '#6a9ecf'}
# %% 2.0 Impacts of 1) AC and 2) Satellite along Shoreline

shoreline = valid[['roi', 'main_roi', 'date', 'level', 'shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']]
df_wide_shoreline = shoreline.pivot(
    index=['main_roi', 'date', 'roi'],
    columns='level',
    values=['shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']
).reset_index()
flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_shoreline.columns]
df_wide_shoreline.columns = flat_cols

# %% 2.1 AC's impact on Shoreline Water Fractions by ROI
temp = df_wide_shoreline.copy()

temp['ls_toa_sr_diff'] = temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) - temp['shoreline_ls_water_frac_adaptive_sr'].astype(float)
temp['adj_ls_toa_sr_diff'] = temp['ls_toa_sr_diff'] / temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) * 100
temp['s2_toa_sr_diff'] = temp['shoreline_s2_water_frac_adaptive_toa'].astype(float) - temp['shoreline_s2_water_frac_adaptive_sr'].astype(float)
temp['adj_s2_toa_sr_diff'] = temp['s2_toa_sr_diff'] / temp['shoreline_s2_water_frac_adaptive_toa'].astype(float) * 100

# Melt the DataFrame to create a long-format dataset
melted_df = pd.melt(
    temp,
    id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
    value_vars=['adj_ls_toa_sr_diff', 'adj_s2_toa_sr_diff'],  # Columns to melt into rows
    var_name='satellite_type',  # Name for the new categorical column
    value_name='adjusted_difference'  # Name for the values column
)

# Map the satellite types to more readable labels
melted_df['satellite_type'] = melted_df['satellite_type'].map({
    'adj_ls_toa_sr_diff': 'Landsat 8',
    'adj_s2_toa_sr_diff': 'Sentinel-2'
})

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='adjusted_difference',
    hue='satellite_type',
    palette=sat_color_pal,  
    width=0.7  # Adjust box width
)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Adjusted TOA-SR Difference by Region and Satellite (Shoreline PLD -60m, +60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(bottom=-0.3)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)", fontsize=12)
plt.legend(title='Satellite')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# %% 2.2 Satellite's impact on Shoreline Water Fractions by ROI

temp = df_wide_shoreline.copy()

temp['toa_ls_s2_diff'] = temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) - temp['shoreline_s2_water_frac_adaptive_toa'].astype(float)
temp['adj_toa_ls_s2_diff'] = temp['toa_ls_s2_diff'] / temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) * 100
temp['sr_ls_s2_diff'] = temp['shoreline_ls_water_frac_adaptive_sr'].astype(float) - temp['shoreline_s2_water_frac_adaptive_sr'].astype(float)
temp['adj_sr_ls_s2_diff'] = temp['sr_ls_s2_diff'] / temp['shoreline_ls_water_frac_adaptive_sr'].astype(float) * 100

# Melt the DataFrame to create a long-format dataset
melted_df = pd.melt(
    temp,
    id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
    value_vars=['adj_toa_ls_s2_diff', 'adj_sr_ls_s2_diff'],  # Columns to melt into rows
    var_name='ac_level',  # Name for the new categorical column
    value_name='adjusted_difference'  # Name for the values column
)

# Map the satellite types to more readable labels
melted_df['ac_level'] = melted_df['ac_level'].map({
    'adj_sr_ls_s2_diff': 'SR',
    'adj_toa_ls_s2_diff': 'TOA'
})

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='adjusted_difference',
    hue='ac_level',
    palette=ac_color_pal,  
    width=0.7  
)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Adjusted LS8 - S2 Difference by Region and AC Processing (Shoreline PLD -60m, +60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(bottom=-170)
plt.ylabel("Adjusted TOA-SR Difference (LS8% - S2% / LS8%) * 100", fontsize=12)
plt.legend(title='AC Processing level')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# %% 3.0 Plot AC's impact on Lake (+ 60m) Water Fractions by ROI

lake = valid[['roi', 'main_roi', 'date', 'level', 'buff_lake_s2_water_frac_adaptive', 'buff_lake_ls_water_frac_adaptive']]

df_wide_lake = lake.pivot(
    index=['main_roi', 'date', 'roi'], 
    columns='level', 
    values=['buff_lake_s2_water_frac_adaptive', 'buff_lake_ls_water_frac_adaptive']
).reset_index()

flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_lake.columns]
df_wide_lake.columns = flat_cols

df_wide_lake['ls_toa_sr_diff'] = df_wide_lake['buff_lake_ls_water_frac_adaptive_toa'].astype(float) - df_wide_lake['buff_lake_ls_water_frac_adaptive_sr'].astype(float)
df_wide_lake['adj_ls_toa_sr_diff'] = df_wide_lake['ls_toa_sr_diff'] / df_wide_lake['buff_lake_ls_water_frac_adaptive_toa'].astype(float) * 100
df_wide_lake['s2_toa_sr_diff'] = df_wide_lake['buff_lake_s2_water_frac_adaptive_toa'].astype(float) - df_wide_lake['buff_lake_s2_water_frac_adaptive_sr'].astype(float)
df_wide_lake['adj_s2_toa_sr_diff'] = df_wide_lake['s2_toa_sr_diff'] / df_wide_lake['buff_lake_s2_water_frac_adaptive_toa'].astype(float) * 100

melted_df = pd.melt(
    df_wide_lake,
    id_vars=['main_roi', 'date', 'roi'],
    value_vars=['adj_ls_toa_sr_diff', 'adj_s2_toa_sr_diff'],
    var_name='satellite_type',
    value_name='adjusted_difference'
)

# Map the satellite types to more readable labels
melted_df['satellite_type'] = melted_df['satellite_type'].map({
    'adj_ls_toa_sr_diff': 'Landsat 8',
    'adj_s2_toa_sr_diff': 'Sentinel-2'
})

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='adjusted_difference',
    hue='satellite_type',
    palette=sat_color_pal,  
    width=0.7  # Adjust box width
)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Adjusted TOA-SR Difference by Region and Satellite (PLD Lake +60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(bottom=-30)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%) * 100", fontsize=12)
plt.legend(title='Satellite')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# %% 3.1 Satellite's impact on Lake (+ 60m) Water Fractions by ROI

temp = df_wide_lake.copy()

temp['toa_ls_s2_diff'] = temp['buff_lake_ls_water_frac_adaptive_toa'].astype(float) - temp['buff_lake_s2_water_frac_adaptive_toa'].astype(float)
temp['adj_toa_ls_s2_diff'] = temp['toa_ls_s2_diff'] / temp['buff_lake_ls_water_frac_adaptive_toa'].astype(float) * 100
temp['sr_ls_s2_diff'] = temp['buff_lake_ls_water_frac_adaptive_sr'].astype(float) - temp['buff_lake_s2_water_frac_adaptive_sr'].astype(float)
temp['adj_sr_ls_s2_diff'] = temp['sr_ls_s2_diff'] / temp['buff_lake_ls_water_frac_adaptive_sr'].astype(float) * 100

# Melt the DataFrame to create a long-format dataset
melted_df = pd.melt(
    temp,
    id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
    value_vars=['adj_toa_ls_s2_diff', 'adj_sr_ls_s2_diff'],  # Columns to melt into rows
    var_name='ac_level',  # Name for the new categorical column
    value_name='adjusted_difference'  # Name for the values column
)

# Map the satellite types to more readable labels
melted_df['ac_level'] = melted_df['ac_level'].map({
    'adj_sr_ls_s2_diff': 'SR',
    'adj_toa_ls_s2_diff': 'TOA'
})

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='adjusted_difference',
    hue='ac_level',
    palette=ac_color_pal,  
    width=0.7  
)


plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Adjusted LS8 - S2 Difference by Region and AC Processing (PLD Lake +60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(bottom=-55)
plt.ylabel("Adjusted LS8-S2 Difference (LS8% - S2%) / LS8% * 100", fontsize=12)
plt.legend(title='AC Processing level')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()