# %% Libraries and directories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch1.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch1.csv')
combined = pd.concat([sr_bilinear30, toa_bilinear30], ignore_index=True)
valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data'].copy()
valid['main_roi'] = valid['roi'].apply(lambda x: x.split('_')[0])

# %% Pivot the DataFrame

shoreline = valid[['roi', 'main_roi', 'date', 'level', 'shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']]
df_wide_shoreline = shoreline.pivot(
    index=['main_roi', 'date', 'roi'],
    columns='level',
    values=['shoreline_s2_water_frac_adaptive', 'shoreline_ls_water_frac_adaptive']
).reset_index()
flat_cols = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_wide_shoreline.columns]
df_wide_shoreline.columns = flat_cols

df_wide_shoreline['ls_toa_sr_diff'] = df_wide_shoreline['shoreline_ls_water_frac_adaptive_toa'].astype(float) - df_wide_shoreline['shoreline_ls_water_frac_adaptive_sr'].astype(float)
df_wide_shoreline['adj_ls_toa_sr_diff'] = df_wide_shoreline['ls_toa_sr_diff'] / df_wide_shoreline['shoreline_ls_water_frac_adaptive_toa'].astype(float)
df_wide_shoreline['s2_toa_sr_diff'] = df_wide_shoreline['shoreline_s2_water_frac_adaptive_toa'].astype(float) - df_wide_shoreline['shoreline_s2_water_frac_adaptive_sr'].astype(float)
df_wide_shoreline['adj_s2_toa_sr_diff'] = df_wide_shoreline['s2_toa_sr_diff'] / df_wide_shoreline['shoreline_s2_water_frac_adaptive_toa'].astype(float)

# %% Plot 

# Melt the DataFrame to create a long-format dataset
melted_df = pd.melt(
    df_wide_shoreline,
    id_vars=['main_roi', 'date', 'roi'],  # Keep these as identifiers
    value_vars=['adj_ls_toa_sr_diff', 'adj_s2_toa_sr_diff'],  # Columns to melt into rows
    var_name='satellite_type',  # Name for the new categorical column
    value_name='adjusted_difference'  # Name for the values column
)

#melted_df = melted_df[melted_df['adjusted_difference'] > -0.5]

# Map the satellite types to more readable labels
melted_df['satellite_type'] = melted_df['satellite_type'].map({
    'adj_ls_toa_sr_diff': 'Landsat 8',
    'adj_s2_toa_sr_diff': 'Sentinel-2'
})

# Create the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=melted_df,
    x='main_roi',
    y='adjusted_difference',
    hue='satellite_type',
    palette={'Landsat 8': '#ff9933', 'Sentinel-2': '#000080'},  # Red for Landsat, Blue for Sentinel
    width=0.7  # Adjust box width
)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Adjusted TOA-SR Difference by Region and Satellite (Shoreline)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)", fontsize=12)
plt.legend(title='Satellite')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



# %%
