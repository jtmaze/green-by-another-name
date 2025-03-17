# %% 1.0 Import libraries and data

import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch1.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch1.csv')
combined = pd.concat([sr_bilinear30, toa_bilinear30], ignore_index=True)
valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']

# Define month orders
month_order = ['May','Jun','Jul','Aug','Sep']
half_month_order = [f"{prefix}_{month}" for month in month_order for prefix in ['Early', 'Late']]
month_dtype = CategoricalDtype(categories=month_order, ordered=True)
half_month_dtype = CategoricalDtype(categories=half_month_order, ordered=True)

# Helper functions
def prepare_data(df, ls_col, s2_col):
    # Select and pivot data
    temp = df[['date', 'level', 'roi', ls_col, s2_col]]
    temp_pivoted = temp.pivot(index=['date', 'roi'], columns='level', values=[ls_col, s2_col])
    
    # Flatten columns
    flat_columns = [f"{col[0]}_{col[1]}" for col in temp_pivoted.columns]
    temp_pivoted.columns = flat_columns
    temp_flat = temp_pivoted.reset_index()
    
    # Calculate differences
    s2_toa = temp_flat[f"{s2_col}_toa"].astype(float)
    s2_sr = temp_flat[f"{s2_col}_sr"].astype(float)
    temp_flat['s2_level_diff'] = s2_toa - s2_sr
    
    ls_toa = temp_flat[f"{ls_col}_toa"].astype(float)
    ls_sr = temp_flat[f"{ls_col}_sr"].astype(float)
    temp_flat['ls_level_diff'] = ls_toa - ls_sr
    
    # Add date columns
    temp_flat['date'] = pd.to_datetime(temp_flat['date'])
    temp_flat['month'] = temp_flat['date'].dt.strftime('%b').astype(month_dtype)
    temp_flat['day'] = temp_flat['date'].dt.day
    temp_flat['half_month'] = temp_flat.apply(
        lambda x: f"Early_{x['date'].strftime('%b')}" if x['day'] <= 15 else f"Late_{x['date'].strftime('%b')}", 
        axis=1
    ).astype(half_month_dtype)
    
    return temp_flat

def create_boxplot(data, x_col, title_prefix, y_lim=None):
    melt_df = pd.melt(
        data, 
        id_vars=['date', 'roi', x_col],
        value_vars=['s2_level_diff', 'ls_level_diff'],
        var_name='diff_type', 
        value_name='diff_value'
    )
    
    # Map difference types to labels
    melt_df['diff_type'] = melt_df['diff_type'].map({
        's2_level_diff': 'Sentinel-2',
        'ls_level_diff': 'Landsat8'
    })
    
    # Determine order based on x_col
    x_order = month_order if x_col == 'month' else half_month_order
    
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=melt_df,
        x=x_col,
        y='diff_value',
        hue='diff_type',
        palette={'Sentinel-2': '#3498db', 'Landsat8': '#e74c3c'},
        order=x_order
    )
    plt.axhline(0, color='red', linestyle='--')

    # Option to trim outliers
    if y_lim is not None:
        plt.ylim(y_lim)
    
    period = "Monthly" if x_col == 'month' else "1/2 Monthly"
    plt.title(f"{title_prefix} {period} Difference in Adaptive Water Fraction (SR vs TOA) for All Sites (bilinear 30)")
    plt.xlabel("Month")
    plt.ylabel("Water Frac Difference (%TOA - %SR)")
    plt.legend(title="Satellite")
    plt.tight_layout()
    plt.show()

# %% 2.0 Process PLD +60m data
lake_data = prepare_data(valid, 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive')

# %% 2.1 PLD 60m+ Boxplot over Months for all sites
create_boxplot(lake_data, 'month', "PLD +60m")

# %% 2.2 PLD 60m+ Boxplot over 1/2 Months for all sites
create_boxplot(lake_data, 'half_month', "PLD +60m")

# %% 3.0 Process Shoreline (PLD +-60m) data
shoreline_data = prepare_data(valid, 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive')

# %% 3.1 Shoreline Boxplot over Months for all sites
create_boxplot(shoreline_data, 'month', "PLD +-60m", y_lim=(-10, 20))
# %% 3.2 Shoreline Boxplot over 1/2 Months for all sites
create_boxplot(shoreline_data, 'half_month', "PLD +-60m", y_lim=(-10, 30))

# %% 4.0 Select specific regions of interest (YKF and AND)


valid.loc[:, 'main_roi'] = valid.apply(lambda row: row['roi'].split('_')[0], axis=1)

# %% 4.1 YKF Boxplot over Months for all sites
region = 'YKF'
YKF_data = valid[valid['main_roi'] == region]
shoreline_YKF = prepare_data(YKF_data, 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive')
lake_YKF = prepare_data(YKF_data, 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive')
create_boxplot(shoreline_YKF, 'month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_YKF, 'month', f"{region} PLD +60m", y_lim=None)
create_boxplot(shoreline_YKF, 'half_month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_YKF, 'half_month', f"{region} PLD +60m", y_lim=None)

# %% 4.2 AND Boxplot over Months for all sites
region = 'AND'
AND_data = valid[valid['main_roi'] == region]
shoreline_AND = prepare_data(AND_data, 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive')
lake_AND = prepare_data(AND_data, 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive')
create_boxplot(shoreline_AND, 'month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_AND, 'month', f"{region} PLD +60m", y_lim=None)
create_boxplot(shoreline_AND, 'half_month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_AND, 'half_month', f"{region} PLD +60m", y_lim=None)

# %% 4.3 MRD Boxplot over Months for all sites
region = 'MRD'
MRD_data = valid[valid['main_roi'] == region]
shoreline_MRD = prepare_data(MRD_data, 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive')
create_boxplot(shoreline_MRD, 'month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(shoreline_MRD, 'half_month', f"{region} PLD +-60m", y_lim=None)

# %% 4.4 YKD Boxplot over Months for all sites
region = 'YKD'
YKD_data = valid[valid['main_roi'] == region]
shoreline_YKD = prepare_data(YKD_data, 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive')
lake_YKD = prepare_data(YKD_data, 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive')
create_boxplot(shoreline_YKD, 'month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_YKD, 'month', f"{region} PLD +60m", y_lim=None)
create_boxplot(shoreline_YKD, 'half_month', f"{region} PLD +-60m", y_lim=None)
create_boxplot(lake_YKD, 'half_month', f"{region} PLD +60m", y_lim=None)


# %%
