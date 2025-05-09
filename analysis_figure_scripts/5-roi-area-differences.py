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

lake = combined[['roi', 'main_roi', 'date', 'level', 'buff_lake_s2_water_frac_adaptive', 'buff_lake_ls_water_frac_adaptive']]

df_wide_lake = lake.pivot(
    index=['main_roi', 'date', 'roi'], 
    columns='level', 
    values=['buff_lake_s2_water_frac_adaptive', 'buff_lake_ls_water_frac_adaptive']
).reset_index()

flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_lake.columns]
df_wide_lake.columns = flat_cols

df_wide_lake['ls_toa_sr_diff'] = df_wide_lake['buff_lake_ls_water_frac_adaptive_toa'] - df_wide_lake['buff_lake_ls_water_frac_adaptive_sr']
df_wide_lake['rel_ls_toa_sr_diff'] = (
    df_wide_lake['ls_toa_sr_diff'] / ((df_wide_lake['buff_lake_ls_water_frac_adaptive_toa'] + df_wide_lake['buff_lake_ls_water_frac_adaptive_sr']) * 0.5) * 100
)

df_wide_lake['s2_toa_sr_diff'] = df_wide_lake['buff_lake_s2_water_frac_adaptive_toa'] - df_wide_lake['buff_lake_s2_water_frac_adaptive_sr']
df_wide_lake['rel_s2_toa_sr_diff'] = (
    df_wide_lake['s2_toa_sr_diff'] / ((df_wide_lake['buff_lake_s2_water_frac_adaptive_toa'] + df_wide_lake['buff_lake_s2_water_frac_adaptive_sr']) * 0.5) * 100
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

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='relative_difference',
    hue='satellite_type',
    palette=sat_color_pal,  
    width=0.7  # Adjust box width
)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Relative AC difference by Region and Satellite (PLD Lake +60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(-15, 110)
plt.ylabel("Relative AC Difference (%)", fontsize=12)
plt.legend(title='Satellite')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

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

temp['toa_ls_s2_diff'] = temp['buff_lake_ls_water_frac_adaptive_toa'] - temp['buff_lake_s2_water_frac_adaptive_toa']
temp['rel_toa_ls_s2_diff'] = (
    temp['toa_ls_s2_diff'] / ((temp['buff_lake_ls_water_frac_adaptive_toa'] + temp['buff_lake_s2_water_frac_adaptive_toa']) * 0.5) * 100
)

temp['sr_ls_s2_diff'] = temp['buff_lake_ls_water_frac_adaptive_sr'] - temp['buff_lake_s2_water_frac_adaptive_sr']
temp['rel_sr_ls_s2_diff'] = (
    temp['sr_ls_s2_diff'] / ((temp['buff_lake_ls_water_frac_adaptive_sr'] + temp['buff_lake_s2_water_frac_adaptive_sr']) * 0.5) * 100
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

# Create the boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='relative_difference',
    hue='ac_level',
    palette=ac_color_pal,  
    width=0.7  
)


plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Relative Satellite Difference by Region and AC (PLD + 60m)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylim(-40, 65)
plt.ylabel("Relative Satellite Differences (%)", fontsize=12)
plt.legend(title='AC Processing Level')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

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

# %% Same plot but for shorelines... not keeping it

# shoreline = combined[['roi', 'main_roi', 'date', 'level', 'shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']]
# df_wide_shoreline = shoreline.pivot(
#     index=['main_roi', 'date', 'roi'],
#     columns='level',
#     values=['shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']
# ).reset_index()
# flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_shoreline.columns]
# df_wide_shoreline.columns = flat_cols

# # %% 2.1 AC's impact on Shoreline Water Fractions by ROI
# temp = df_wide_shoreline.copy()

# temp['ls_toa_sr_diff'] = temp['shoreline_ls_water_frac_adaptive_toa'] - temp['shoreline_ls_water_frac_adaptive_sr']
# temp['adj_ls_toa_sr_diff'] = temp['ls_toa_sr_diff'] / temp['shoreline_ls_water_frac_adaptive_toa'] * 100
# temp['s2_toa_sr_diff'] = temp['shoreline_s2_water_frac_adaptive_toa'] - temp['shoreline_s2_water_frac_adaptive_sr']
# temp['adj_s2_toa_sr_diff'] = temp['s2_toa_sr_diff'] / temp['shoreline_s2_water_frac_adaptive_toa'] * 100

# # %%
# # Melt the DataFrame to create a long-format dataset
# melted_df = pd.melt(
#     temp,
#     id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
#     value_vars=['adj_ls_toa_sr_diff', 'adj_s2_toa_sr_diff'],  # Columns to melt into rows
#     var_name='satellite_type',  # Name for the new categorical column
#     value_name='adjusted_difference'  # Name for the values column
# )

# # Map the satellite types to more readable labels
# melted_df['satellite_type'] = melted_df['satellite_type'].map({
#     'adj_ls_toa_sr_diff': 'Landsat 8',
#     'adj_s2_toa_sr_diff': 'Sentinel-2'
# })

# # Create the boxplot
# plt.figure(figsize=(12, 5))
# sns.boxplot(
#     data=melted_df,
#     x='main_roi',
#     y='adjusted_difference',
#     hue='satellite_type',
#     palette=sat_color_pal,  
#     width=0.7  # Adjust box width
# )

# # Add a horizontal line at y=0 for reference
# plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# # Customize the plot
# plt.title('Adjusted TOA-SR Difference by Region and Satellite (Shoreline PLD -60m, +60m)', fontsize=14)
# plt.xlabel('Region', fontsize=12)
# plt.ylim(bottom=-0.3)
# plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)", fontsize=12)
# plt.legend(title='Satellite')
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()

# # %% 2.2 Satellite's impact on Shoreline Water Fractions by ROI

# temp = df_wide_shoreline.copy()

# temp['toa_ls_s2_diff'] = temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) - temp['shoreline_s2_water_frac_adaptive_toa'].astype(float)
# temp['adj_toa_ls_s2_diff'] = temp['toa_ls_s2_diff'] / temp['shoreline_ls_water_frac_adaptive_toa'].astype(float) * 100
# temp['sr_ls_s2_diff'] = temp['shoreline_ls_water_frac_adaptive_sr'].astype(float) - temp['shoreline_s2_water_frac_adaptive_sr'].astype(float)
# temp['adj_sr_ls_s2_diff'] = temp['sr_ls_s2_diff'] / temp['shoreline_ls_water_frac_adaptive_sr'].astype(float) * 100

# # Melt the DataFrame to create a long-format dataset
# melted_df = pd.melt(
#     temp,
#     id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
#     value_vars=['adj_toa_ls_s2_diff', 'adj_sr_ls_s2_diff'],  # Columns to melt into rows
#     var_name='ac_level',  # Name for the new categorical column
#     value_name='adjusted_difference'  # Name for the values column
# )

# # Map the satellite types to more readable labels
# melted_df['ac_level'] = melted_df['ac_level'].map({
#     'adj_sr_ls_s2_diff': 'SR',
#     'adj_toa_ls_s2_diff': 'TOA'
# })

# # Create the boxplot
# plt.figure(figsize=(12, 5))
# sns.boxplot(
#     data=melted_df,
#     x='main_roi',
#     y='adjusted_difference',
#     hue='ac_level',
#     palette=ac_color_pal,  
#     width=0.7  
# )

# # Add a horizontal line at y=0 for reference
# plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# # Customize the plot
# plt.title('Adjusted LS8 - S2 Difference by Region and AC Processing (Shoreline PLD -60m, +60m)', fontsize=14)
# plt.xlabel('Region', fontsize=12)
# plt.ylim(bottom=-170)
# plt.ylabel("Adjusted TOA-SR Difference (LS8% - S2% / LS8%) * 100", fontsize=12)
# plt.legend(title='AC Processing level')
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
