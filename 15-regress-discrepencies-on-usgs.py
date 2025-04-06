# %% 1.0 Libraries and file paths

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

path_gauge_h = './data/ykf_gauge_height.txt'
path_gauge_dis = './data/ykf_usgs_discharge.txt'
lake_areas_dir = './data/lake_area_results/'

# %% 1.1 Read the USGS height data

# filter out the header file from USGS data
with open(path_gauge_h, 'r') as file:
    lines = [line for line in file if not line.startswith('#')]

data_str = ''.join(lines)

df = pd.read_csv(StringIO(data_str), delimiter='\t', skiprows=[1])
df['date'] = pd.to_datetime(df['datetime']).dt.date
gauge_h = df.groupby('date')['1717_00065'].mean().reset_index()

# %% 1.2 Read the USGS discharge data 

with open(path_gauge_dis, 'r') as file:
    lines = [line for line in file if not line.startswith('#')]

data_str = ''.join(lines)

df = pd.read_csv(StringIO(data_str), delimiter='\t', skiprows=[1])
df['1716_00060'] = pd.to_numeric(df['1716_00060'], errors='coerce')
df['date'] = pd.to_datetime(df['datetime']).dt.date
gauge_dis = df.groupby('date')['1716_00060'].mean().reset_index()


# %% 2.0 Read the area data

toa_data_resamp = pd.read_csv(f'{lake_areas_dir}/toa_resampled_bilinear30_area_summaries_batch2.csv')
sr_data_resamp = pd.read_csv(f'{lake_areas_dir}/sr_resampled_bilinear30_area_summaries_batch2.csv')
combined = pd.concat([toa_data_resamp, sr_data_resamp])

cols_to_keep = ['date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']
sr = sr_data_resamp[cols_to_keep].copy()

sr = sr.rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'sr_ls', 
        'buff_lake_s2_water_frac_adaptive': 'sr_s2'
    }
)

sr['sr_sat_diff'] = sr['sr_ls'] - sr['sr_s2']
sr['rel_sr_sat_diff'] = sr['sr_sat_diff'] / sr['sr_ls']
sr.drop('level', axis=1, inplace=True)

toa = toa_data_resamp[cols_to_keep].copy()

toa = toa.rename(
    columns={
        'buff_lake_ls_water_frac_adaptive': 'toa_ls', 
        'buff_lake_s2_water_frac_adaptive': 'toa_s2'
    }
)

toa['toa_sat_diff'] = toa['toa_ls'] - toa['toa_s2']
toa['rel_toa_sat_diff'] = toa['toa_sat_diff'] / toa['toa_ls']
toa.drop('level', axis=1, inplace=True)

# %% 3.0 Join the satellite data with the gauge data

combined = pd.merge(toa, sr, how='outer', on=['date', 'roi'])

combined['ls8_ac_diff'] = combined['toa_ls'] - combined['sr_ls']
combined['rel_ls8_ac_diff'] = combined['ls8_ac_diff'] / combined['toa_ls']
combined['s2_ac_diff'] = combined['toa_s2'] - combined['sr_s2']
combined['rel_s2_ac_diff'] = combined['s2_ac_diff'] / combined['toa_s2']

ykf = combined[combined['roi'].str.contains('YKF', na=False)].copy()
ykf['date'] = pd.to_datetime(ykf['date']).dt.date

ykf = pd.merge(ykf, gauge_h, how='left', on='date')
ykf = pd.merge(ykf, gauge_dis, how='left', on='date')
ykf['day_of_year'] = pd.to_datetime(ykf['date']).dt.dayofyear

ykf = ykf.rename(
    columns={
        '1717_00065': 'gauge_h_ft',
        '1716_00060': 'gauge_dis_cfs'
    }
)

# %% Plot the relative AC discrepancies to discharge

plot_data = pd.melt(
    ykf,
    id_vars=['date', 'roi', 'gauge_dis_cfs', 'day_of_year'],
    value_vars=['rel_ls8_ac_diff', 'rel_s2_ac_diff'],
    var_name='satellite_metric',
    value_name='ac_difference'
)

# Add a cleaner satellite name column
plot_data['satellite'] = plot_data['satellite_metric'].map({
    'rel_ls8_ac_diff': 'Landsat 8',
    'rel_s2_ac_diff': 'Sentinel-2'
})

# Create the plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_data,
    x='gauge_dis_cfs',
    y='ac_difference',
    hue='satellite',
    style='satellite',  # Use different marker styles
    palette={'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'},
    edgecolor='black',
    linewidth=0.5,
    s=80  # Make points slightly larger
)

# Add title and labels
plt.title('TOA-SR Relative Difference vs. Yukon River Discharge (USGS @ Stevens Village)', fontsize=14)
plt.xlabel('Gauge Discharge (cfs)', fontsize=12)
plt.ylabel('Relative TOA-SR Difference (TOA% - SR% / TOA%)', fontsize=12)

# Add a reference line at y=0
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Improve the legend
plt.legend(title='Satellite', title_fontsize=12)

plt.tight_layout()
plt.show()

# %% Plot the relative Satellite discrepancies to discharge

plot_data = pd.melt(
    ykf,
    id_vars=['date', 'roi', 'gauge_dis_cfs', 'day_of_year'],
    value_vars=['rel_toa_sat_diff', 'rel_sr_sat_diff'],
    var_name='ac_level',
    value_name='sat_difference'
)

# Add a cleaner satellite name column
plot_data['ac_level'] = plot_data['ac_level'].map({
    'rel_toa_sat_diff': 'TOA',
    'rel_sr_sat_diff': 'SR'
})

# Create the plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_data,
    x='gauge_dis_cfs',
    y='sat_difference',
    hue='ac_level',
    style='ac_level',  # Use different marker styles
    palette={'SR': '#4C72B0', 'TOA': '#55A868'},
    edgecolor='black',
    linewidth=0.5,
    s=80  
)

# Add title and labels
plt.title('TOA-SR Relative Difference vs. Yukon River Discharge (USGS @ Stevens Village)', fontsize=14)
plt.ylim((-0.7, 0.7))
plt.xlabel('Gauge Discharge (cfs)', fontsize=12)
plt.ylabel('Relative LS8-S2 Difference (LS8% - S2% / LS8%)', fontsize=12)

# Add a reference line at y=0
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Improve the legend
plt.legend(title='AC Processing', title_fontsize=12)

plt.tight_layout()
plt.show()
