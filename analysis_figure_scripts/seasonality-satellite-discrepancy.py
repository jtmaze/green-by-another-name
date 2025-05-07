# %% 

import os
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')
resample_method = 'bilinear30'
level = 'toa'

area_path = f'./data/lake_area_results/{level}_resampled_{resample_method}_area_summaries_batch3.csv'
seasonality_path = f'./data/gswo_glad_areas.csv'

area_data = pd.read_csv(area_path)
seasonality_data = pd.read_csv(seasonality_path)
#area_data = area_data[area_data['pld_plus_valid_frac'] > 40]

# %%

seasonality_data['gswo_seasonal_frac'] = (
    (seasonality_data['gswo_june_wtr_count'] - seasonality_data['gswo_aug_wtr_count']) / seasonality_data['gswo_june_wtr_count'] * 100
)

seasonality_data['glad_seasonal_frac'] = (
    (seasonality_data['glad_june_wtr_count'] - seasonality_data['glad_aug_wtr_count']) / seasonality_data['glad_june_wtr_count'] * 100
)


# %% 

area_data = area_data[
    ['roi', 'date', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive', 'pld_plus_valid_frac']
].copy()

area_data.rename(
    columns={'buff_lake_ls_water_frac_adaptive': 'ls_water_frac',
             'buff_lake_s2_water_frac_adaptive': 's2_water_frac'
             }, 
    inplace=True
)

area_data['abs_sat_diff'] = area_data['ls_water_frac'] - area_data['s2_water_frac']
area_data['rel_sat_diff'] = area_data['abs_sat_diff'] / area_data['ls_water_frac'] * 100

grouped_areas = area_data.groupby(['roi'])['rel_sat_diff'].mean()

# %% 

merged_df = pd.merge(grouped_areas, seasonality_data, on='roi', how='inner')
print(len(merged_df))


glad_over_20 = (merged_df['glad_seasonal_frac'] > 20).sum()
print(glad_over_20)

gswo_over_20 = (merged_df['gswo_seasonal_frac'] > 20).sum()
print(gswo_over_20)


# %%
plot_data = merged_df.copy()
plot_data = plot_data[['roi', 'gswo_seasonal_frac', 'glad_seasonal_frac', 'rel_sat_diff']]
plot_data['rel_sat_diff'] = plot_data['rel_sat_diff'] * -1
# Melt the dataframe to create a column that identifies the source (gswo or glad)
melted_data = pd.melt(
    plot_data,
    id_vars=['roi', 'rel_sat_diff'],
    value_vars=['gswo_seasonal_frac', 'glad_seasonal_frac'],
    var_name='source',
    value_name='seasonal_frac'
)

# Clean up the source names
melted_data['source'] = melted_data['source'].str.replace('_seasonal_frac', '')
melted_data['source'] = melted_data['source'].map({'gswo': 'GSWO', 'glad': 'GLAD'})

# Create linear regression plot using lmplot without confidence intervals
g = sns.lmplot(
    data=melted_data, 
    x='rel_sat_diff',
    y='seasonal_frac',
    hue='source',
    height=6,
    aspect=1.5,
    legend=False,
    scatter_kws={'alpha': 0.7, 's': 80},  # Increase point size
    line_kws={'linewidth': 2},
    ci=None  # Disable confidence intervals
)

# Add regression statistics to the plot
for source, group in melted_data.groupby('source'):
    slope, intercept, r_value, p_value, std_err = linregress(group['rel_sat_diff'], group['seasonal_frac'])
    print(f"{source}: r²={r_value**2:.4f}, p={p_value:.4f}, slope={slope:.4f}")

plt.xlabel('Relative Satellite Difference (%)')
plt.ylabel('Seasonal Fraction (%)')
plt.title('Buffered Lakes Satellite Difference vs Seasonal Fraction')

# Move legend outside the plot
plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()

# %%
