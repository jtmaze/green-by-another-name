# %% 1.0 Libraries and file paths

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from functions.misc.week_dt_converters import year_week_to_datetime, add_year_week

ts_dir = './data/s2_weekly_timeseries'
area_dir = './data/lake_area_results'

ts_files = glob.glob(f'{ts_dir}/*.csv')
# %% 2.0 Read and format the timeseries data

dfs = []
for f in ts_files:
    df = pd.read_csv(f)
    df = df[
        ['invalid_pixels', 'water_pixels', 'land_pixels', 
        'total_roi_pixels', 'total_lake_pixels', 'roi_name', 'mosaic_id']
    ].copy()

    df['year'] = df['mosaic_id'].apply(lambda x: x.split('_')[-2])
    df['week'] = df['mosaic_id'].apply(lambda x: x.split('_')[-1])
    df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
    df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int64')
    df['date_for_plot'] = df.apply(lambda row: year_week_to_datetime(row['year'], row['week']), axis=1)

    df['valid_fraction'] = ((df['total_lake_pixels'] - df['invalid_pixels'])
                            / df['total_lake_pixels'] * 100)
    df['lake_water_fraction'] = (df['water_pixels'] / 
                                 (df['total_lake_pixels'] - df['invalid_pixels']) * 100)
    
    print(f'Total observations for {df['roi_name'].iloc[0]} is {len(df)}')

    # plt.figure(figsize=(12, 6))
    # scatter = plt.scatter(
    #     x=df['valid_fraction'],
    #     y=df['lake_water_fraction'],
    #     c=df['week'],
    #     cmap='RdYlBu'
    # )
    # plt.title(f'{df['roi_name'].iloc[0]} lake water % relative to valid obs %')
    # plt.colorbar(scatter, label='Week of year')
    # plt.xlabel('Valid Fraction of ALPOD pixels')
    # plt.ylabel('Water Fraction of ALPOD pixels')
    # plt.show()

    # NOTE: I'm messing around with this ALOT
    df_clean = df[(df['valid_fraction'] >= 80)]
    # df_clean = df_clean[(df_clean['week'] >= 25) &
    #                     (df_clean['week'] <= 31)
    # ]

    print("------------------------------------------------")
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        x=df_clean['valid_fraction'],
        y=df_clean['lake_water_fraction'],
        c=df_clean['week'],
        cmap='RdYlBu'
    )
    plt.title(f'{df_clean['roi_name'].iloc[0]} lake water % relative to valid obs %')
    plt.colorbar(scatter, label='Week of year')
    plt.xlabel('Valid Fraction of ALPOD pixels')
    plt.ylabel('Water Fraction of ALPOD pixels')
    plt.show()

    df_clean = df_clean[
        ['year_week', 'date_for_plot', 'roi_name', 'valid_fraction', 'lake_water_fraction', 'week']
    ].copy()
    max_weekly_wtr_frac = df_clean['lake_water_fraction'].max()
    print(max_weekly_wtr_frac)
    df_clean['wtr_frac_perc_max'] = df_clean['lake_water_fraction'] / max_weekly_wtr_frac * 100

    plt.figure(figsize=(12, 6))
    plt.plot(df_clean['date_for_plot'], df_clean['wtr_frac_perc_max'], 'o-')
    plt.title(f'Water Fraction Timeseries for {df_clean['roi_name'].iloc[0]}')
    plt.xlabel('Date')
    plt.ylabel('ALPOD Water Fraction (%) of max observed')
    plt.show()

    dfs.append(df_clean)

timeseries = pd.concat(dfs, ignore_index=True)
rois = timeseries['roi_name'].unique()

# %% 2.0 Read and format the lake area data

resample_method = 'bilinear30'
toa_data = pd.read_csv(f'{area_dir}/toa_resampled_{resample_method}_area_summaries_batch2.csv')
sr_data = pd.read_csv(f'{area_dir}/sr_resampled_{resample_method}_area_summaries_batch2.csv')

cols_to_keep =['date', 'roi', 'buff_lake_ls_water_frac_adaptive',
               'buff_lake_s2_water_frac_adaptive']

toa_data = toa_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'toa_ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 'toa_s2_water_frac',
        'roi': 'roi_name'
    }
).copy()

sr_data = sr_data[cols_to_keep].rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'sr_ls_water_frac',
        'buff_lake_s2_water_frac_adaptive': 'sr_s2_water_frac',
        'roi': 'roi_name'
    }
).copy()

combined = pd.merge(left=toa_data, right=sr_data, on=['date', 'roi_name'], how='inner')
combined['abs_ls_ac_diff'] = combined['toa_ls_water_frac'] - combined['sr_ls_water_frac']
combined['rel_ls_ac_diff'] = combined['abs_ls_ac_diff'] / combined['toa_ls_water_frac'] * 100
combined['abs_s2_ac_diff'] = combined['toa_s2_water_frac'] - combined['sr_s2_water_frac']
combined['rel_s2_ac_diff'] = combined['abs_s2_ac_diff'] / combined['toa_s2_water_frac'] * 100

combined = add_year_week(combined, datetime_col='date')

print(len(combined))
# %% 3.0 Merge the weekly timeseries with the area discrepances

combined = pd.merge(combined, timeseries, how='left', on=['roi_name', 'year_week'])
print(len(combined))
print(combined['wtr_frac_perc_max'].isna().sum())
combined['roi_main'] = combined['roi_name'].apply(lambda x: x.split('_')[0])

# %% Plot Sentinel-2's absolute buffered PLD% for both TOA and SR vs. Elizabeth's classifacation.
plot_data = combined.copy()
plot_data = combined[['date', 'roi_name', 'toa_s2_water_frac', 'sr_s2_water_frac', 'lake_water_fraction']]

plot_data_long = pd.melt(
    plot_data,
    id_vars=['roi_name', 'date', 'lake_water_fraction'],
    value_vars=['toa_s2_water_frac', 'sr_s2_water_frac'],
    var_name='s2_ac_level',
    value_name='coincident_wtr_frac'
)
label_map = {
    'toa_s2_water_frac': 'TOA',
    'sr_s2_water_frac': 'SR'
}
plot_data_long['s2_ac_level'] = plot_data_long['s2_ac_level'].map(label_map)

p = plt.figure(figsize=(8, 8))
p = sns.lmplot(
    data=plot_data_long,
    x='lake_water_fraction',
    y='coincident_wtr_frac',
    hue='s2_ac_level',
    ci=None
)
p.legend.set_title('S2 AC Level')
plt.xlabel('Weekly S2 ALPOD Water Fraction (%)')
plt.ylabel('PLD Water Fraction (%) (from overlap area)')
plt.title('S2 Footprint Water Fractions vs. ALPOD Lake Water Fraction')

toa = plot_data_long[plot_data_long['s2_ac_level'] == 'TOA']
toa = toa.dropna().copy()
toa_model = linregress(
    x=toa['lake_water_fraction'],
    y=toa['coincident_wtr_frac']
)

sr = plot_data_long[plot_data_long['s2_ac_level'] == 'SR']
sr = sr.dropna().copy()
sr_model = linregress(
    x=sr['lake_water_fraction'],
    y=sr['coincident_wtr_frac']
)
print(f'TOA r-squared: {toa_model.rvalue ** 2:.2f}, slope: {toa_model.slope:.2f}')
print(f'SR r-squared: {sr_model.rvalue ** 2:.2f}, slope: {sr_model.slope:.2f}')
# %%

plot_data = combined.copy()
plot_data = combined[['date', 'roi_name', 'toa_ls_water_frac', 'sr_ls_water_frac', 'lake_water_fraction']]

plot_data_long = pd.melt(
    plot_data,
    id_vars=['roi_name', 'date', 'lake_water_fraction'],
    value_vars=['toa_ls_water_frac', 'sr_ls_water_frac'],
    var_name='ls_ac_level',
    value_name='coincident_wtr_frac'
)
label_map = {
    'toa_ls_water_frac': 'TOA',
    'sr_ls_water_frac': 'SR'
}
plot_data_long['ls_ac_level'] = plot_data_long['ls_ac_level'].map(label_map)

p = plt.figure(figsize=(8, 8))
p = sns.lmplot(
    data=plot_data_long,
    x='lake_water_fraction',
    y='coincident_wtr_frac',
    hue='ls_ac_level',
    ci=None
)
p.legend.set_title('LS8 AC Level')
plt.xlabel('Weekly S2 ALPOD Water Fraction (%) of Maximum Observed')
plt.ylabel('LS8 PLD Water Fraction (from overlap area)')
plt.title('Relative LS8 AC Difference vs. ALPOD Lake Water Fraction')

toa = plot_data_long[plot_data_long['ls_ac_level'] == 'TOA']
toa = toa.dropna().copy()
toa_model = linregress(
    x=toa['lake_water_fraction'],
    y=toa['coincident_wtr_frac']
)

sr = plot_data_long[plot_data_long['ls_ac_level'] == 'SR']
sr = sr.dropna().copy()
sr_model = linregress(
    x=sr['lake_water_fraction'],
    y=sr['coincident_wtr_frac']
)
print(f'TOA r-squared: {toa_model.rvalue ** 2:.2f}, slope: {toa_model.slope:.2f}')
print(f'SR r-squared: {sr_model.rvalue ** 2:.2f}, slope: {sr_model.slope:.2f}')

# %%

plot_data = combined.copy()
plot_data['roi_main'] = plot_data['roi_name'].apply(lambda x: x.split('_')[0])

plt.figure(figsize=(8, 8))
sns.scatterplot(
    data=plot_data,
    x='wtr_frac_perc_max',
    y='rel_s2_ac_diff',
    hue='roi_main'
)
plt.title('Relative Sentinel-2 AC Difference vs. ALPOD Lake Water Fraction')
plt.xlabel('(%) water fraction of maxed obseved over ALPOD')
plt.ylabel('Sentinel-2 (TOA-SR) / TOA * 100')