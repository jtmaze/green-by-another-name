# %% Libraries and file paths

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

seasonal_areas_path = './data/aggregate_seasonality/area_seasonal_results.csv'

resample_method = 'bilinear30'
level = 'toa'
satellite_discrepancies_path = f'./data/lake_area_results/{level}_resampled_{resample_method}_area_summaries_batch2.csv'

# %% 2.0 Read and clean the aggregate seasonal data

agg_seasonality = pd.read_csv(seasonal_areas_path)

agg_seasonality = agg_seasonality[
    (agg_seasonality['scope'] == "all_pld") &
    (agg_seasonality['timeframe'] == "full") &
    (agg_seasonality['buffer'] == 60) &
    (agg_seasonality['threshold'] == 80)
].copy()

agg_seasonality = agg_seasonality[['roi_name', 'dataset', 'net_lake_sn_frac']].copy()

mean_agg_seasonality = agg_seasonality.groupby(['roi_name'])['net_lake_sn_frac'].mean().reset_index()
mean_agg_seasonality.replace(to_replace='anderson_plain', value='AND', inplace=True)

# %% 3.0 Read and clean the satellite discrepancies data

sat_discrepancies = pd.read_csv(satellite_discrepancies_path)
sat_discrepancies = sat_discrepancies[
    ['roi', 'date', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']
].copy()

sat_discrepancies['rel_ls_s2_diff'] = (
    (sat_discrepancies['buff_lake_ls_water_frac_adaptive'] - sat_discrepancies['buff_lake_s2_water_frac_adaptive'])
    / sat_discrepancies['buff_lake_ls_water_frac_adaptive'] * 100
)

sat_discrepancies['roi_name'] = sat_discrepancies['roi'].apply(lambda x: x.split('_')[0])
sat_mean_discepancies = sat_discrepancies.groupby(['roi_name'])['rel_ls_s2_diff'].mean().reset_index()
sat_mean_discepancies.sort_values(by='rel_ls_s2_diff', ascending=True, inplace=True)

# %% 4.0 Merge and run spearman rank correlation

merged_df = pd.merge(mean_agg_seasonality, sat_mean_discepancies, on='roi_name', how='inner')

rho, pval = spearmanr(
    merged_df['net_lake_sn_frac'],
    merged_df['rel_ls_s2_diff']
)
print(f'Spearman correlation: {rho:.3f}, p-value: {pval:.3f}')

# %% Barplot to illustrate spearman rank correlation
merged_sorted = merged_df.sort_values(by='net_lake_sn_frac', ascending=True)
plot_data = pd.melt(
    merged_sorted,
    id_vars=['roi_name'],
    value_vars=['net_lake_sn_frac', 'rel_ls_s2_diff'],
    var_name='metric',
    value_name='value'
)

# Create nicer labels for the metrics
plot_data['metric'] = plot_data['metric'].map({
    'net_lake_sn_frac': 'Net Lake Seasonal Fraction',
    'rel_ls_s2_diff': 'Relative LS8-S2 Difference'
})

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=plot_data,
    x='roi_name',
    y='value',
    hue='metric',
    palette={'Net Lake Seasonal Fraction': '#bcbd22', 'Relative LS8-S2 Difference': '#800000'},
    edgecolor='black',
    linewidth=1
)

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.xlabel('ROI Name', fontsize=12)
plt.ylabel('% Area of PLD Buffered (+60m)', fontsize=12)
plt.title(f'Lake Seasonality vs Satellite Differences by ROI\nSpearman ρ: {rho:.3f} (p: {pval:.3f})', 
         pad=20, fontsize=14)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()
