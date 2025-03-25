# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

resample_methods = ['bilinear30', 'noresample', 'bilinear60', 'noresample']

# %% 2.0 Explore differences in water fraction by satellite.

"""
This section loads the CSV data, calculates satellite water fraction differences by AC level,
creates boxplots to visualize them, and performs t-tests to assess statistical significance.
"""

test_results = []
for i in resample_methods:

    sr = pd.read_csv(f'./data/lake_area_results/sr_resampled_{i}_area_summaries_batch1.csv')
    sr = sr[sr['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']
    toa = pd.read_csv(f'./data/lake_area_results/toa_resampled_{i}_area_summaries_batch1.csv')
    toa = toa[toa['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']

    cols_to_make_float = [
        'total_ls_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_otsu', 'total_s2_water_frac_adaptive',
        'lake_ls_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_otsu', 'lake_s2_water_frac_adaptive',
        'shoreline_ls_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_otsu', 'shoreline_s2_water_frac_adaptive',
    ]

    for col in cols_to_make_float:
        sr[col] = sr[col].astype(float)
        toa[col] = toa[col].astype(float)

    # SR Boxplot and t-tests
    sr['total_diff_sr'] = sr['total_ls_water_frac_adaptive'] - sr['total_s2_water_frac_adaptive'] 
    sr['relative_total_diff_sr'] = sr['total_diff_sr'] / sr['total_ls_water_frac_adaptive']
    sr['lake_diff_sr'] = sr['lake_ls_water_frac_adaptive'] - sr['lake_s2_water_frac_adaptive']
    sr['relative_lake_diff_sr'] = sr['lake_diff_sr'] / sr['lake_ls_water_frac_adaptive']
    sr['shoreline_diff_sr'] = sr['shoreline_ls_water_frac_adaptive'] - sr['shoreline_s2_water_frac_adaptive']
    sr['relative_shoreline_diff_sr'] = sr['shoreline_diff_sr'] / sr['shoreline_ls_water_frac_adaptive']

    abs_plot_data = sr[['total_diff_sr', 'lake_diff_sr', 'shoreline_diff_sr']].copy()
    abs_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline']  # Rename columns
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=abs_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Absolute Difference in Water Fraction (SR, {i})')
    plt.ylabel('LS8% - S2% Water Fraction')
    plt.xlabel('Landscape Zone')

    rel_plot_data = sr[['relative_total_diff_sr', 'relative_lake_diff_sr', 'relative_shoreline_diff_sr']].copy()
    rel_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline']  # Rename columns
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=rel_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    # NOTE: resetting y-axis limits to 0-97.5 percentile
    flat = rel_plot_data.values.flatten()
    flat = flat[~np.isnan(flat)]
    y_min = np.percentile(flat, 2.5)
    y_max = np.percentile(flat, 100)
    padding = (y_max - y_min) * 0.5
    plt.ylim(y_min - padding, y_max + padding)
    plt.title(f'Relative Difference in Water Fraction (SR, {i})')
    plt.ylabel('(LS8% - S2% / LS8%) Water Fraction')
    plt.xlabel('Landscape Zone')

    total_ttest = stats.ttest_1samp(sr['relative_total_diff_sr'].dropna(), 0)
    total_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'total_landscape',
        't_statistic': total_ttest.statistic,
        'p_value': total_ttest.pvalue
    }
    lake_ttest = stats.ttest_1samp(sr['relative_lake_diff_sr'].dropna(), 0)
    lake_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'lake',
        't_statistic': lake_ttest.statistic,
        'p_value': lake_ttest.pvalue
    }
    shoreline_ttest = stats.ttest_1samp(sr['relative_shoreline_diff_sr'].dropna(), 0)
    shoreline_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'shoreline',
        't_statistic': shoreline_ttest.statistic,
        'p_value': shoreline_ttest.pvalue
    }
    test_results.append(total_result)
    test_results.append(lake_result)
    test_results.append(shoreline_result)

    # TOA Boxplot and t-tests
    toa['total_diff_toa'] = toa['total_ls_water_frac_adaptive'] - toa['total_s2_water_frac_adaptive']
    toa['lake_diff_toa'] = toa['lake_ls_water_frac_adaptive'] - toa['lake_s2_water_frac_adaptive']
    toa['shoreline_diff_toa'] = toa['shoreline_ls_water_frac_adaptive'] - toa['shoreline_s2_water_frac_adaptive']
    toa['relative_total_diff_toa'] = toa['total_diff_toa'] / toa['total_ls_water_frac_adaptive']
    toa['relative_lake_diff_toa'] = toa['lake_diff_toa'] / toa['lake_ls_water_frac_adaptive']
    toa['relative_shoreline_diff_toa'] = toa['shoreline_diff_toa'] / toa['shoreline_ls_water_frac_adaptive']

    # TOA Boxplot with renamed columns for absolute differences
    abs_plot_data = toa[['total_diff_toa', 'lake_diff_toa', 'shoreline_diff_toa']].copy()
    abs_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline']  # Rename columns

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=abs_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Absolute Satellite Difference in Water Fraction (TOA, {i})')
    plt.ylabel('LS8% - S2% Water Fraction')
    plt.xlabel('Landscape Zone')

    # TOA Boxplot with renamed columns for relative differences
    rel_plot_data = toa[['relative_total_diff_toa', 'relative_lake_diff_toa', 'relative_shoreline_diff_toa']].copy()
    rel_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline']  # Rename columns

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=rel_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    # NOTE: resetting y-axis limits to 0-97.5 percentile
    flat = rel_plot_data.values.flatten()
    flat = flat[~np.isnan(flat)]
    y_min = np.percentile(flat, 2.5)
    y_max = np.percentile(flat, 100)
    padding = (y_max - y_min) * 0.5
    plt.ylim(y_min - padding, y_max + padding)
    plt.title(f'Relative Difference in Water Fraction (TOA, {i})')
    plt.ylabel('(LS8% - S2% / LS8%) Water Fraction')
    plt.xlabel('Landscape Zone')

    # Relative t-tests for TOA
    total_ttest = stats.ttest_1samp(toa['relative_total_diff_toa'].dropna(), 0)
    total_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'total_landscape',
        't_statistic': total_ttest.statistic,
        'p_value': total_ttest.pvalue
    }
    lake_ttest = stats.ttest_1samp(toa['relative_lake_diff_toa'].dropna(), 0)
    lake_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'lake',
        't_statistic': lake_ttest.statistic,
        'p_value': lake_ttest.pvalue
    }
    shoreline_ttest = stats.ttest_1samp(toa['relative_shoreline_diff_toa'].dropna(), 0)
    shoreline_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'shoreline',
        't_statistic': shoreline_ttest.statistic,
        'p_value': shoreline_ttest.pvalue
    }
    test_results.append(total_result)
    test_results.append(lake_result)
    test_results.append(shoreline_result)


# %% Concatonate the t-test results into a single dataframe

test_results_df = pd.DataFrame(test_results)
test_results_df.head(10)

# %% Make plot of t-statisitics for each zone and resampling method

toa_results = test_results_df[test_results_df['level'] == 'toa']
ax = sns.barplot(
    data=toa_results,
    x='resample_method',
    y='t_statistic',
    hue='zone'
)
ax.set_title('TOA Divergence T-Statistic by Resampling Method and Zone')
ax.set_xlabel('Resampling Method')
ax.set_ylabel('T-Statistic for no difference (i.e., [LS8%-S2%]/LS8% = 0)')
for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1)
plt.show()

sr_reslts = test_results_df[test_results_df['level'] == 'sr']
ax = sns.barplot(
    data=sr_reslts,
    x='resample_method',
    y='t_statistic',
    hue='zone'
)
ax.set_title('SR Divergence T-Statistic by Resampling Method and Zone')
ax.set_xlabel('Resampling Method')
ax.set_ylabel('T-Statistic for no difference (i.e., [LS8%-S2%]/LS8% = 0)')
for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1)
plt.show()

# %%

significant_results = test_results_df[
    (test_results_df['p_value'] < 0.05)
].sort_values(by='p_value')
significant_results.head(15)
print(len(significant_results))

# %%
