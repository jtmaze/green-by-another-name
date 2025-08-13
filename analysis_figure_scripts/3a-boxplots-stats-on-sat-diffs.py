# %% 1.0 Libraries and file paths
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

resample_methods = ['bilinear30', 'noresample']

temp = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch3.csv')
valids = temp[['roi', 'date']].agg('_'.join, axis=1).unique()

rel_y_lim = (-150, 125)
abs_y_lim = (-35, 40)

# %% 2.0 Explore differences in water fraction by satellite.

"""
This section loads the CSV data, calculates satellite water fraction differences by AC level,
creates boxplots to visualize them, and performs t-tests to assess statistical significance.
"""

test_results = []
for i in resample_methods:

    sr = pd.read_csv(f'./data/lake_area_results/sr_resampled_{i}_area_summaries_batch3.csv')
    print(f'{len(sr)} SR observations')
    sr = sr[sr[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
    print(f'{len(sr)} SR observations QA/QC')
    toa = pd.read_csv(f'./data/lake_area_results/toa_resampled_{i}_area_summaries_batch3.csv')
    print(f'{len(toa)} TOA observations')
    toa = toa[toa[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
    print(f'{len(toa)} quality TOA observations')

    cols_to_make_float = [
        'total_ls_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_otsu', 'total_s2_water_frac_adaptive',
        'lake_ls_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_otsu', 'lake_s2_water_frac_adaptive',
        'shoreline_ls_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_otsu', 'shoreline_s2_water_frac_adaptive',
    ]

    for col in cols_to_make_float:
        sr[col] = sr[col].astype(float)
        toa[col] = toa[col].astype(float)

    # SR Boxplot and t-tests
    # Total Landscape
    sr['total_diff_sr'] = sr['total_ls_water_frac_adaptive'] - sr['total_s2_water_frac_adaptive'] 
    sr['relative_total_diff_sr'] = (
        sr['total_diff_sr'] / ((sr['total_ls_water_frac_adaptive'] + sr['total_s2_water_frac_adaptive']) * 0.5) * 100
    )
    # Lake Difference
    sr['lake_diff_sr'] = sr['lake_ls_water_frac_adaptive'] - sr['lake_s2_water_frac_adaptive']
    sr['relative_lake_diff_sr'] = (
        sr['lake_diff_sr'] / ((sr['lake_ls_water_frac_adaptive'] + sr['lake_s2_water_frac_adaptive']) * 0.5) * 100
    )
    # Shoreline Difference
    sr['shoreline_diff_sr'] = sr['shoreline_ls_water_frac_adaptive'] - sr['shoreline_s2_water_frac_adaptive']

    sr['relative_shoreline_diff_sr'] = (
        sr['shoreline_diff_sr'] / ((sr['shoreline_ls_water_frac_adaptive'] + sr['shoreline_s2_water_frac_adaptive']) * 0.5) * 100
    )
    # Lake plus shoreline difference
    sr['lake_shoreline_diff_sr'] = sr['buff_lake_ls_water_frac_adaptive'] - sr['buff_lake_s2_water_frac_adaptive']
    sr['relative_lake_shoreline_diff_sr'] = (
        sr['lake_shoreline_diff_sr'] / ((sr['buff_lake_ls_water_frac_adaptive'] + sr['buff_lake_s2_water_frac_adaptive']) * 0.5) * 100
    )

    abs_plot_data = sr[['total_diff_sr', 'lake_diff_sr', 'shoreline_diff_sr', 'lake_shoreline_diff_sr']].copy()
    abs_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline', 'Lake + Shoreline']  # Rename columns
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=abs_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylim(abs_y_lim)
    plt.title(f'Absolute Difference (%) in Water Fraction (SR, {i})')
    plt.ylabel(None)
    plt.xlabel(None)

    rel_plot_data = sr[['relative_total_diff_sr', 'relative_lake_diff_sr', 'relative_shoreline_diff_sr', 'relative_lake_shoreline_diff_sr']].copy()
    rel_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline', 'Lake + Shoreline']  # Rename columns
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=rel_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylim(rel_y_lim)
    plt.title(f'Relative Difference (%) in Water Fraction (SR, {i})')
    plt.ylabel(None)
    plt.xlabel(None)

    # T-tests for surface reflectance (SR)
    total_ttest = stats.ttest_1samp(sr['relative_total_diff_sr'].dropna(), 0)
    total_ttest_absolute = stats.ttest_1samp(sr['total_diff_sr'].dropna(), 0)
    total_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'total_landscape',
        't_statistic_rel': total_ttest.statistic,
        'p_value_rel': total_ttest.pvalue,
        't_statistic_abs': total_ttest_absolute.statistic, 
        'p_value_abs': total_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(sr['relative_total_diff_sr']),
        'mean_diff_abs': np.mean(sr['total_diff_sr']),
        'var_diff_rel': np.var(sr['relative_total_diff_sr']),
        'var_diff_abs': np.var(sr['total_diff_sr']),
        '75th_percentile_abs': np.percentile(sr['total_diff_sr'], 75),
        '25th_percentile_abs': np.percentile(sr['total_diff_sr'], 25),
        '75th_percentile_rel': np.percentile(sr['relative_total_diff_sr'], 75),
        '25th_percentile_rel': np.percentile(sr['relative_total_diff_sr'], 25),
        'iqr_abs': np.percentile(sr['total_diff_sr'], 75) - np.percentile(sr['total_diff_sr'], 25),
        'iqr_rel': np.percentile(sr['relative_total_diff_sr'], 75) - np.percentile(sr['relative_total_diff_sr'], 25)
    }
    
    lake_ttest = stats.ttest_1samp(sr['relative_lake_diff_sr'].dropna(), 0)
    lake_ttest_absolute = stats.ttest_1samp(sr['lake_diff_sr'].dropna(), 0)
    lake_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'lake',
        't_statistic_rel': lake_ttest.statistic,
        'p_value_rel': lake_ttest.pvalue,
        't_statistic_abs': lake_ttest_absolute.statistic,
        'p_value_abs': lake_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(sr['relative_lake_diff_sr']),
        'mean_diff_abs': np.mean(sr['lake_diff_sr']),
        'var_diff_rel': np.var(sr['relative_lake_diff_sr']),
        'var_diff_abs': np.var(sr['lake_diff_sr']),
        '75th_percentile_abs': np.percentile(sr['lake_diff_sr'], 75),
        '25th_percentile_abs': np.percentile(sr['lake_diff_sr'], 25),
        '75th_percentile_rel': np.percentile(sr['relative_lake_diff_sr'], 75),
        '25th_percentile_rel': np.percentile(sr['relative_lake_diff_sr'], 25),
        'iqr_abs': np.percentile(sr['lake_diff_sr'], 75) - np.percentile(sr['lake_diff_sr'], 25),
        'iqr_rel': np.percentile(sr['relative_lake_diff_sr'], 75) - np.percentile(sr['relative_lake_diff_sr'], 25)
    }
    
    shoreline_ttest = stats.ttest_1samp(sr['relative_shoreline_diff_sr'].dropna(), 0)
    shoreline_ttest_absolute = stats.ttest_1samp(sr['shoreline_diff_sr'].dropna(), 0)
    shoreline_result = {
        'level': 'sr',
        'resample_method': i,
        'zone': 'shoreline',
        't_statistic_rel': shoreline_ttest.statistic,
        'p_value_rel': shoreline_ttest.pvalue,
        't_statistic_abs': shoreline_ttest_absolute.statistic,
        'p_value_abs': shoreline_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(sr['relative_shoreline_diff_sr']),
        'mean_diff_abs': np.mean(sr['shoreline_diff_sr']),
        'var_diff_rel': np.var(sr['relative_shoreline_diff_sr']),
        'var_diff_abs': np.var(sr['shoreline_diff_sr']),
        '75th_percentile_abs': np.percentile(sr['shoreline_diff_sr'], 75),
        '25th_percentile_abs': np.percentile(sr['shoreline_diff_sr'], 25),
        '75th_percentile_rel': np.percentile(sr['relative_shoreline_diff_sr'], 75),
        '25th_percentile_rel': np.percentile(sr['relative_shoreline_diff_sr'], 25),
        'iqr_abs': np.percentile(sr['shoreline_diff_sr'], 75) - np.percentile(sr['shoreline_diff_sr'], 25),
        'iqr_rel': np.percentile(sr['relative_shoreline_diff_sr'], 75) - np.percentile(sr['relative_shoreline_diff_sr'], 25)
    }
    
    lake_shoreline_ttest = stats.ttest_1samp(sr['relative_lake_shoreline_diff_sr'].dropna(), 0)
    lake_shoreline_ttest_absolute = stats.ttest_1samp(sr['lake_shoreline_diff_sr'].dropna(), 0)
    lake_shoreline_result = {
        'level': 'sr', 
        'resample_method': i, 
        'zone': 'shoreline and lake',
        't_statistic_rel': lake_shoreline_ttest.statistic,
        'p_value_rel': lake_shoreline_ttest.pvalue,
        't_statistic_abs': lake_shoreline_ttest_absolute.statistic,
        'p_value_abs': lake_shoreline_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(sr['relative_lake_shoreline_diff_sr']),
        'mean_diff_abs': np.mean(sr['lake_shoreline_diff_sr']),
        'var_diff_rel': np.var(sr['relative_lake_shoreline_diff_sr']),
        'var_diff_abs': np.var(sr['lake_shoreline_diff_sr']),
        '75th_percentile_abs': np.percentile(sr['lake_shoreline_diff_sr'], 75),
        '25th_percentile_abs': np.percentile(sr['lake_shoreline_diff_sr'], 25),
        '75th_percentile_rel': np.percentile(sr['relative_lake_shoreline_diff_sr'], 75),
        '25th_percentile_rel': np.percentile(sr['relative_lake_shoreline_diff_sr'], 25),
        'iqr_abs': np.percentile(sr['lake_shoreline_diff_sr'], 75) - np.percentile(sr['lake_shoreline_diff_sr'], 25),
        'iqr_rel': np.percentile(sr['relative_lake_shoreline_diff_sr'], 75) - np.percentile(sr['relative_lake_shoreline_diff_sr'], 25)
    }

    test_results.append(total_result)
    test_results.append(lake_result)
    test_results.append(shoreline_result)
    test_results.append(lake_shoreline_result)

    # TOA Boxplot and t-tests
    toa['total_diff_toa'] = toa['total_ls_water_frac_adaptive'] - toa['total_s2_water_frac_adaptive']
    toa['lake_diff_toa'] = toa['lake_ls_water_frac_adaptive'] - toa['lake_s2_water_frac_adaptive']
    toa['shoreline_diff_toa'] = toa['shoreline_ls_water_frac_adaptive'] - toa['shoreline_s2_water_frac_adaptive']
    # Lake plus shoreline difference
    toa['lake_shoreline_diff_toa'] = toa['buff_lake_ls_water_frac_adaptive'] - toa['buff_lake_s2_water_frac_adaptive']
    
    toa['relative_total_diff_toa'] = (
        toa['total_diff_toa'] / ((toa['total_ls_water_frac_adaptive'] + toa['total_s2_water_frac_adaptive']) * 0.5) * 100
    )

    toa['relative_lake_diff_toa'] = (
        toa['lake_diff_toa'] / ((toa['lake_ls_water_frac_adaptive'] + toa['lake_s2_water_frac_adaptive']) * 0.5) * 100
    )

    toa['relative_shoreline_diff_toa'] = (
        toa['shoreline_diff_toa'] / ((toa['shoreline_ls_water_frac_adaptive'] + toa['shoreline_s2_water_frac_adaptive']) * 0.5) * 100
    )
    toa['relative_lake_shoreline_diff_toa'] = (
        toa['lake_shoreline_diff_toa'] / ((toa['buff_lake_ls_water_frac_adaptive'] + toa['buff_lake_s2_water_frac_adaptive']) * 0.5) * 100
    )

    # TOA Boxplot with renamed columns for absolute differences
    abs_plot_data = toa[['total_diff_toa', 'lake_diff_toa', 'shoreline_diff_toa', 'lake_shoreline_diff_toa']].copy()
    abs_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline', 'Lake + Shoreline']  # Rename columns

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=abs_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylim(abs_y_lim)
    plt.title(f'Absolute Difference (%) in Water Fraction (TOA, {i})')
    plt.ylabel(None)
    plt.xlabel(None)

    # TOA Boxplot with renamed columns for relative differences
    rel_plot_data = toa[['relative_total_diff_toa', 'relative_lake_diff_toa', 'relative_shoreline_diff_toa', 'relative_lake_shoreline_diff_toa']].copy()
    rel_plot_data.columns = ['Total Landscape', 'Lake', 'Shoreline', 'Lake + Shoreline']  # Rename columns

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=rel_plot_data)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylim(rel_y_lim)
    plt.title(f'Relative Difference (%) in Water Fraction (TOA, {i})')
    plt.ylabel(None)
    plt.xlabel(None)

    # T-tests for TOA
    total_ttest = stats.ttest_1samp(toa['relative_total_diff_toa'].dropna(), 0)
    total_ttest_absolute = stats.ttest_1samp(toa['total_diff_toa'].dropna(), 0)
    total_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'total_landscape',
        't_statistic_rel': total_ttest.statistic,
        'p_value_rel': total_ttest.pvalue,
        't_statistic_abs': total_ttest_absolute.statistic,
        'p_value_abs': total_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(toa['relative_total_diff_toa']),
        'mean_diff_abs': np.mean(toa['total_diff_toa']),
        'var_diff_rel': np.var(toa['relative_total_diff_toa']),
        'var_diff_abs': np.var(toa['total_diff_toa']),
        '75th_percentile_abs': np.percentile(toa['total_diff_toa'], 75),
        '25th_percentile_abs': np.percentile(toa['total_diff_toa'], 25),
        '75th_percentile_rel': np.percentile(toa['relative_total_diff_toa'], 75),
        '25th_percentile_rel': np.percentile(toa['relative_total_diff_toa'], 25),
        'iqr_abs': np.percentile(toa['total_diff_toa'], 75) - np.percentile(toa['total_diff_toa'], 25),
        'iqr_rel': np.percentile(toa['relative_total_diff_toa'], 75) - np.percentile(toa['relative_total_diff_toa'], 25)
    }
    
    lake_ttest = stats.ttest_1samp(toa['relative_lake_diff_toa'].dropna(), 0)
    lake_ttest_absolute = stats.ttest_1samp(toa['lake_diff_toa'].dropna(), 0)
    lake_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'lake',
        't_statistic_rel': lake_ttest.statistic,
        'p_value_rel': lake_ttest.pvalue,
        't_statistic_abs': lake_ttest_absolute.statistic,
        'p_value_abs': lake_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(toa['relative_lake_diff_toa']),
        'mean_diff_abs': np.mean(toa['lake_diff_toa']),
        'var_diff_rel': np.var(toa['relative_lake_diff_toa']),
        'var_diff_abs': np.var(toa['lake_diff_toa']),
        '75th_percentile_abs': np.percentile(toa['lake_diff_toa'], 75),
        '25th_percentile_abs': np.percentile(toa['lake_diff_toa'], 25),
        '75th_percentile_rel': np.percentile(toa['relative_lake_diff_toa'], 75),
        '25th_percentile_rel': np.percentile(toa['relative_lake_diff_toa'], 25),
        'iqr_abs': np.percentile(toa['lake_diff_toa'], 75) - np.percentile(toa['lake_diff_toa'], 25),
        'iqr_rel': np.percentile(toa['relative_lake_diff_toa'], 75) - np.percentile(toa['relative_lake_diff_toa'], 25)
    }
    
    shoreline_ttest = stats.ttest_1samp(toa['relative_shoreline_diff_toa'].dropna(), 0)
    shoreline_ttest_absolute = stats.ttest_1samp(toa['shoreline_diff_toa'].dropna(), 0)
    shoreline_result = {
        'level': 'toa',
        'resample_method': i,
        'zone': 'shoreline',
        't_statistic_rel': shoreline_ttest.statistic,
        'p_value_rel': shoreline_ttest.pvalue,
        't_statistic_abs': shoreline_ttest_absolute.statistic,
        'p_value_abs': shoreline_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(toa['relative_shoreline_diff_toa']),
        'mean_diff_abs': np.mean(toa['shoreline_diff_toa']),
        'var_diff_rel': np.var(toa['relative_shoreline_diff_toa']),
        'var_diff_abs': np.var(toa['shoreline_diff_toa']),
        '75th_percentile_abs': np.percentile(toa['shoreline_diff_toa'], 75),
        '25th_percentile_abs': np.percentile(toa['shoreline_diff_toa'], 25),
        '75th_percentile_rel': np.percentile(toa['relative_shoreline_diff_toa'], 75),
        '25th_percentile_rel': np.percentile(toa['relative_shoreline_diff_toa'], 25),
        'iqr_abs': np.percentile(toa['shoreline_diff_toa'], 75) - np.percentile(toa['shoreline_diff_toa'], 25),
        'iqr_rel': np.percentile(toa['relative_shoreline_diff_toa'], 75) - np.percentile(toa['relative_shoreline_diff_toa'], 25)
    }
    
    lake_shoreline_ttest = stats.ttest_1samp(toa['relative_lake_shoreline_diff_toa'].dropna(), 0)
    lake_shoreline_ttest_absolute = stats.ttest_1samp(toa['lake_shoreline_diff_toa'].dropna(), 0)
    lake_shoreline_result = {
        'level': 'toa', 
        'resample_method': i, 
        'zone': 'shoreline and lake',
        't_statistic_rel': lake_shoreline_ttest.statistic,
        'p_value_rel': lake_shoreline_ttest.pvalue,
        't_statistic_abs': lake_shoreline_ttest_absolute.statistic,
        'p_value_abs': lake_shoreline_ttest_absolute.pvalue,
        'mean_diff_rel': np.mean(toa['relative_lake_shoreline_diff_toa']),
        'mean_diff_abs': np.mean(toa['lake_shoreline_diff_toa']),
        'var_diff_rel': np.var(toa['relative_lake_shoreline_diff_toa']),
        'var_diff_abs': np.var(toa['lake_shoreline_diff_toa']),
        '75th_percentile_abs': np.percentile(toa['lake_shoreline_diff_toa'], 75),
        '25th_percentile_abs': np.percentile(toa['lake_shoreline_diff_toa'], 25),
        '75th_percentile_rel': np.percentile(toa['relative_lake_shoreline_diff_toa'], 75),
        '25th_percentile_rel': np.percentile(toa['relative_lake_shoreline_diff_toa'], 25),
        'iqr_abs': np.percentile(toa['lake_shoreline_diff_toa'], 75) - np.percentile(toa['lake_shoreline_diff_toa'], 25),
        'iqr_rel': np.percentile(toa['relative_lake_shoreline_diff_toa'], 75) - np.percentile(toa['relative_lake_shoreline_diff_toa'], 25)
    }
    
    test_results.append(total_result)
    test_results.append(lake_result)
    test_results.append(shoreline_result)
    test_results.append(lake_shoreline_result)

    """
    Conduct an ANOVA and Tukey HSD to compare TOA shoreline discrepancies to other lake positions.
    """

    landscape_series = toa['relative_total_diff_toa']
    lake_series = toa['relative_lake_diff_toa']
    shoreline_series = toa['relative_shoreline_diff_toa']
    lake_shoreline_series = toa['relative_lake_shoreline_diff_toa']

    f_statistic, p_value = stats.f_oneway(
        landscape_series,
        lake_series,
        shoreline_series,
        lake_shoreline_series
    )
    print(f'ANOVA F-statistic: {f_statistic}, p-value: {p_value}')
    # Double check the series lengths

    tukey_df = pd.DataFrame({
        'landscape': landscape_series,
        'lake': lake_series,
        'shoreline': shoreline_series,
        'lake_shoreline': lake_shoreline_series
    })
    # One-way ANOVA across lake positions
    tukey_df = pd.melt(tukey_df, var_name='zone', value_name='value')
    model = ols('value ~ zone', data=tukey_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=tukey_df['value'], groups=tukey_df['zone'], alpha=0.05)
    print('HSD Results:')
    print(tukey)

# %% Concatonate the t-test results into a single dataframe

test_results_df = pd.DataFrame(test_results)

# %%
toa_total_landscape_mean_bilinear30 = test_results_df[
    (test_results_df['level'] == 'toa') &
    (test_results_df['zone'] == 'total_landscape') &
    (test_results_df['resample_method'] == 'bilinear30')
]
toa_shoreline_mean_bilinear30 = test_results_df[
    (test_results_df['level'] == 'toa') &
    (test_results_df['zone'] == 'shoreline') &
    (test_results_df['resample_method'] == 'bilinear30')
]
sr_total_landscape_mean_bilinear30 = test_results_df[
    (test_results_df['level'] == 'sr') &
    (test_results_df['zone'] == 'total_landscape') &
    (test_results_df['resample_method'] == 'bilinear30')
]
sr_shoreline_mean_bilinear30 = test_results_df[
    (test_results_df['level'] == 'sr') &
    (test_results_df['zone'] == 'shoreline') &
    (test_results_df['resample_method'] == 'bilinear30')
]
print(f'Total landscape toa abs diff mean: {toa_total_landscape_mean_bilinear30["mean_diff_abs"].values[0]}')
print(f'Total landscape toa abs diff t-statistic: {toa_total_landscape_mean_bilinear30["t_statistic_abs"].values[0]}')
print(f'Total landscape toa variance relative: {toa_total_landscape_mean_bilinear30["var_diff_rel"].values[0]}')
print(f'Total landscape sr variance relative: {sr_total_landscape_mean_bilinear30["var_diff_rel"].values[0]}')
print(f'Total landscape toa sd relative: {np.sqrt(toa_total_landscape_mean_bilinear30["var_diff_rel"].values[0])}')
print(f'Total landscape sr sd relative: {np.sqrt(sr_total_landscape_mean_bilinear30["var_diff_rel"].values[0])}')
print(f'Shoreline toa sd relative: {np.sqrt(toa_shoreline_mean_bilinear30["var_diff_rel"].values[0])}')
print(f'Shoreline sr sd relative: {np.sqrt(sr_shoreline_mean_bilinear30["var_diff_rel"].values[0])}')

test_results_df.head(10)

# %%

print_table = test_results_df[
    ['level', 'resample_method', 'zone', 'p_value_rel', 'p_value_abs', 
     'mean_diff_abs', 'mean_diff_rel', 'iqr_abs', 'iqr_rel']
].copy()

# %% Make plot of t-statisitics for each zone and resampling method

toa_results = test_results_df[test_results_df['level'] == 'toa']
ax = sns.barplot(
    data=toa_results,
    x='resample_method',
    y='t_statistic_rel',
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
    y='t_statistic_rel',
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
    (test_results_df['p_value_abs'] < 0.05)
].sort_values(by='p_value_abs')
significant_results.head(15)
print(len(significant_results))

# %%
