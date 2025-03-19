# %% 1.0 Libraries and directories
import glob
import re
import pandas as pd

from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Improve unique observations for linear models, so p-values are not inflated. 

mask_solar_stats_dir = './data/img_mask_solar_stats/'
mask_solar_stats_files = glob.glob(f'{mask_solar_stats_dir}/*.csv')

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch1.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch1.csv')
combined = pd.concat([sr_bilinear30, toa_bilinear30], ignore_index=True)
valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']

# Read the Sentinel-2 image stats
s2_meta_files = []
s2_meta_dfs = []
for i in mask_solar_stats_files:
    match = re.search(pattern="Sentinel2_attrs", string=i)
    if match:
        s2_meta_files.append(i)
        s2_meta_dfs.append(pd.read_csv(i))
s2_meta = pd.concat(s2_meta_dfs)
s2_solar = s2_meta[
    ['s2_export_name', 'roi_name', 'date', 
     'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 
     'AOT_RETRIVAL_ACCURACY', 'RADIATIVE_TRANSFER_ACCURACY', 'WATER_VAPOUR_RETRIEVAL_ACCURACY']
]

# Read the Landsat image stats
ls_meta_files = []
ls_meta_dfs = []
for i in mask_solar_stats_files:
    match = re.search(pattern="Landsat8_attrs", string=i)
    if match:
        print(i)
        ls_meta_files.append(i)
        ls_meta_dfs.append(pd.read_csv(i))

ls_meta = pd.concat(ls_meta_dfs)
print(ls_meta.columns)

# %% Calculate lake differences

lake = valid[['date', 'roi', 'level', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']]
df_wide_lake = lake.pivot(
    index=['date', 'roi'],
    columns='level',
    values=['lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
).reset_index()

flat_cols = [f'{col[0]}_{col[1]}' for col in df_wide_lake.columns]
df_wide_lake.columns = flat_cols
df_wide_lake = df_wide_lake.rename({'date_': 'date', 'roi_': 'roi'}, axis=1)

df_wide_lake['ls_toa_sr_diff'] = df_wide_lake['lake_ls_water_frac_adaptive_toa'].astype(float) - df_wide_lake['lake_ls_water_frac_adaptive_sr'].astype(float)
df_wide_lake['s2_toa_sr_diff'] = df_wide_lake['lake_s2_water_frac_adaptive_toa'].astype(float) - df_wide_lake['lake_s2_water_frac_adaptive_sr'].astype(float)

# %% Calculate shoreline differences

shoreline = valid[['date', 'roi', 'level', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']]
df_wide_shoreline = shoreline.pivot(
    index=['date', 'roi'],
    columns='level',
    values=['shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
).reset_index()

flat_cols = [f'{col[0]}_{col[1]}' for col in df_wide_shoreline.columns]
df_wide_shoreline.columns = flat_cols
df_wide_shoreline = df_wide_shoreline.rename({'date_': 'date', 'roi_': 'roi'}, axis=1)

df_wide_shoreline['ls_toa_sr_diff'] = df_wide_shoreline['shoreline_ls_water_frac_adaptive_toa'].astype(float) - df_wide_shoreline['shoreline_ls_water_frac_adaptive_sr'].astype(float)
df_wide_shoreline['s2_toa_sr_diff'] = df_wide_shoreline['shoreline_s2_water_frac_adaptive_toa'].astype(float) - df_wide_shoreline['shoreline_s2_water_frac_adaptive_sr'].astype(float)


# %% 
temp = df_wide_lake.copy()
temp = temp[['s2_toa_sr_diff', 'lake_s2_water_frac_adaptive_toa', 'date', 'roi']]
temp['lake_s2_water_frac_adaptive_toa'] = temp['lake_s2_water_frac_adaptive_toa'].astype(float)
temp['adj_s2_toa_sr_diff'] = temp['s2_toa_sr_diff'] / temp['lake_s2_water_frac_adaptive_toa']
temp = temp.rename({'roi': 'roi_name'}, axis=1)

s2_solar_water_diff = pd.merge(temp, s2_solar, on=['roi_name', 'date'])

sns.lmplot(
    data=s2_solar_water_diff,
    x='MEAN_SOLAR_AZIMUTH_ANGLE', 
    y='adj_s2_toa_sr_diff'
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Lakes PLD +60 meters Sentinel-2")

sns.lmplot(
    data=s2_solar_water_diff,
    x='MEAN_SOLAR_ZENITH_ANGLE', 
    y='adj_s2_toa_sr_diff'
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Lakes PLD +60 meters Sentinel-2")

# %% Explore Sun Azimuth and Sun Elevation relationships for Landsat8 along shorelines
temp = df_wide_shoreline.copy()
temp = temp[['ls_toa_sr_diff', 'shoreline_ls_water_frac_adaptive_toa', 'date', 'roi']]
temp['shoreline_ls_water_frac_adaptive_toa'] = temp['shoreline_ls_water_frac_adaptive_toa'].astype(float)
temp['adj_ls_toa_sr_diff'] = temp['ls_toa_sr_diff'] / temp['shoreline_ls_water_frac_adaptive_toa']
temp = temp.rename({'roi': 'roi_name'}, axis=1)

ls_solar_water_diff = pd.merge(temp, ls_meta, on=['roi_name', 'date'])

# For the SUN_AZIMUTH vs adj_ls_toa_sr_diff relationship
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['SUN_AZIMUTH'] < 100])} rows with SUN_AZIMUTH < 100")
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] < -2.5])} with adj_ls_toa_sr_diff < -2.5")

# Filter data
filtered_data_azimuth = ls_solar_water_diff[(ls_solar_water_diff['SUN_AZIMUTH'] > 100) & 
                                           (ls_solar_water_diff['adj_ls_toa_sr_diff'] > -2.5)]

# Run linear regression
azimuth_model = linregress(filtered_data_azimuth['SUN_AZIMUTH'], filtered_data_azimuth['adj_ls_toa_sr_diff'])
print("\nSUN_AZIMUTH linear model:")
print(f"slope: {azimuth_model.slope:.3f}")
print(f"intercept: {azimuth_model.intercept:.3f}")
print(f"r-squared: {azimuth_model.rvalue**2:.3f}")
print(f"p-value: {azimuth_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_azimuth,
    x='SUN_AZIMUTH', 
    y='adj_ls_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Shoreline PLD +-60 meters Landsat8")
plt.show()

# For the SUN_ELEVATION vs adj_ls_toa_sr_diff relationship
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] < -2.5])} with adj_ls_toa_sr_diff < -2.5")

# Filter data
filtered_data_elevation = ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] > -2.5]

# Run linear regression
elevation_model = linregress(filtered_data_elevation['SUN_ELEVATION'], filtered_data_elevation['adj_ls_toa_sr_diff'])
print("\nSUN_ELEVATION linear model:")
print(f"slope: {elevation_model.slope:.3f}")
print(f"intercept: {elevation_model.intercept:.3f}")
print(f"r-squared: {elevation_model.rvalue**2:.3f}")
print(f"p-value: {elevation_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_elevation,
    x='SUN_ELEVATION', 
    y='adj_ls_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Shoreline PLD +-60 meters Landsat8")
plt.show()

# %% Explore Sun Azimuth and Sun Elevation relationships for Landsat8 along lakes
temp = df_wide_lake.copy()
temp = temp[['ls_toa_sr_diff', 'lake_ls_water_frac_adaptive_toa', 'date', 'roi']]
temp['lake_ls_water_frac_adaptive_toa'] = temp['lake_ls_water_frac_adaptive_toa'].astype(float)
temp['adj_ls_toa_sr_diff'] = temp['ls_toa_sr_diff'] / temp['lake_ls_water_frac_adaptive_toa']
temp = temp.rename({'roi': 'roi_name'}, axis=1)

ls_solar_water_diff = pd.merge(temp, ls_meta, on=['roi_name', 'date'])

# For the SUN_AZIMUTH vs adj_ls_toa_sr_diff relationship
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['SUN_AZIMUTH'] < 100])} rows with SUN_AZIMUTH < 100")
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] < -1.5])} with adj_ls_toa_sr_diff < -2.5")

# # Filter data
filtered_data_azimuth = ls_solar_water_diff[(ls_solar_water_diff['SUN_AZIMUTH'] > 100) & 
                                           (ls_solar_water_diff['adj_ls_toa_sr_diff'] > -1.5)]

# # Run linear regression
azimuth_model = linregress(filtered_data_azimuth['SUN_AZIMUTH'], filtered_data_azimuth['adj_ls_toa_sr_diff'])
print("\nSUN_AZIMUTH linear model:")
print(f"slope: {azimuth_model.slope:.3f}")
print(f"intercept: {azimuth_model.intercept:.3f}")
print(f"r-squared: {azimuth_model.rvalue**2:.3f}")
print(f"p-value: {azimuth_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_azimuth,
    x='SUN_AZIMUTH', 
    y='adj_ls_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Lake PLD +60 meters Landsat8")
plt.show()

# For the SUN_ELEVATION vs adj_ls_toa_sr_diff relationship
print(f"Removing {len(ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] < -1.5])} with adj_ls_toa_sr_diff < -1.5")

# Filter data
filtered_data_elevation = ls_solar_water_diff[ls_solar_water_diff['adj_ls_toa_sr_diff'] > -1.5]

# Run linear regression
elevation_model = linregress(filtered_data_elevation['SUN_ELEVATION'], filtered_data_elevation['adj_ls_toa_sr_diff'])
print("\nSUN_ELEVATION linear model:")
print(f"slope: {elevation_model.slope:.3f}")
print(f"intercept: {elevation_model.intercept:.3f}")
print(f"r-squared: {elevation_model.rvalue**2:.3f}")
print(f"p-value: {elevation_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_elevation,
    x='SUN_ELEVATION', 
    y='adj_ls_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Lake PLD +60 meters Landsat8")
plt.show()

# %% Explore Sun Azimuth and Sun Elevation relationships for Sentinel-2 along shorelines

temp = df_wide_shoreline.copy()
temp = temp[['s2_toa_sr_diff', 'shoreline_s2_water_frac_adaptive_toa', 'date', 'roi']]
temp['shoreline_s2_water_frac_adaptive_toa'] = temp['shoreline_s2_water_frac_adaptive_toa'].astype(float)
temp['adj_s2_toa_sr_diff'] = temp['s2_toa_sr_diff'] / temp['shoreline_s2_water_frac_adaptive_toa']
temp = temp.rename({'roi': 'roi_name'}, axis=1)

s2_solar_water_diff = pd.merge(temp, s2_solar, on=['roi_name', 'date'])

# For the SUN_AZIMUTH vs adj_s2_toa_sr_diff relationship
filtered_data_azimuth = s2_solar_water_diff.dropna(subset=['adj_s2_toa_sr_diff'])

# Run linear regression
azimuth_model = linregress(filtered_data_azimuth['MEAN_SOLAR_AZIMUTH_ANGLE'], filtered_data_azimuth['adj_s2_toa_sr_diff'])
print("\nSUN_AZIMUTH linear model:")
print(f"slope: {azimuth_model.slope:.3f}")
print(f"intercept: {azimuth_model.intercept:.3f}")
print(f"r-squared: {azimuth_model.rvalue**2:.3f}")
print(f"p-value: {azimuth_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_azimuth,
    x='MEAN_SOLAR_AZIMUTH_ANGLE', 
    y='adj_s2_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Shoreline PLD +-60 meters Sentinel-2")
plt.show()

# For the SUN_ELEVATION vs adj_s2_toa_sr_diff relationship
filtered_data_elevation = s2_solar_water_diff.dropna(subset=['adj_s2_toa_sr_diff'])
# Run linear regression
elevation_model = linregress(filtered_data_elevation['MEAN_SOLAR_ZENITH_ANGLE'], filtered_data_elevation['adj_s2_toa_sr_diff'])
print("\nSUN_ELEVATION linear model:")
print(f"slope: {elevation_model.slope:.3f}")
print(f"intercept: {elevation_model.intercept:.3f}")
print(f"r-squared: {elevation_model.rvalue**2:.3f}")
print(f"p-value: {elevation_model.pvalue:.3e}")

# Plot with original settings
sns.lmplot(
    data=filtered_data_elevation,
    x='MEAN_SOLAR_ZENITH_ANGLE', 
    y='adj_s2_toa_sr_diff',
    ci=None
)
plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
plt.title("Shoreline PLD +-60 meters Sentinel-2")
plt.show()

# %% See if there's a relationship between toa_sr_diff and total water fraction
"""
This chunk of analysis is probably a wash, 
because the composition of rois, and thus water fraction changes with tile combos and cloud cover. 
Means a given date's water fraction doesn't mean wetnessed changed. 
"""
# temp = df_wide.copy()
# temp = temp[['s2_toa_sr_diff', 'lake_s2_water_frac_adaptive_toa', 'date', 'roi']]
# temp['roi_main'] = temp['roi'].str.split('_').str[0]
# temp['lake_s2_water_frac_adaptive_toa'] = temp['lake_s2_water_frac_adaptive_toa'].astype(float)
# temp['adj_s2_toa_sr_diff'] = temp['s2_toa_sr_diff'] / temp['lake_s2_water_frac_adaptive_toa']


# sns.lmplot(
#     data=temp,
#     x='lake_s2_water_frac_adaptive_toa',
#     y='adj_s2_toa_sr_diff',
#     hue='roi_main',
#     ci=None,
#     aspect=1.2
# )
# plt.title("Adjusted S2 TOA%-SR% Difference vs. TOA Water Fraction by ROI")
# plt.xlabel("TOA Water Fraction")
# plt.ylabel("Adjusted TOA-SR Difference (TOA% - SR% / TOA%)")
# plt.show()

# results = {}
# for roi in temp['roi_main'].unique():
#     subset = temp[temp['roi_main'] == roi]
#     subset = subset.dropna()
#     x = subset['lake_s2_water_frac_adaptive_toa']
#     y = subset['adj_s2_toa_sr_diff']
#     reg = linregress(x, y)
#     results[roi] = {
#         'slope': reg.slope,
#         'r_squared': reg.rvalue**2,
#         'p_val': reg.pvalue
#     }

# for roi, stats in results.items():
#     print(f"{roi}: slope={stats['slope']:.4f}, R2={stats['r_squared']:.4f}, p={stats['p_val']:.4g}")



