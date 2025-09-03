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

# %%

test_results = []

for i in resample_methods:

    sr = pd.read_csv(f'./data/lake_area_results/sr_resampled_{i}_area_summaries_batch3.csv')
    sr = sr[sr[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
    print(len(sr))
    toa = pd.read_csv(f'./data/lake_area_results/toa_resampled_{i}_area_summaries_batch3.csv')
    toa = toa[toa[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]
    print(len(toa))

    """
    Reformat the total data
    """

    total_sr_data = sr[['date', 'roi', 'level', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']]
    total_toa_data = toa[['date', 'roi', 'level', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']]

    total = pd.concat([total_sr_data, total_toa_data])

    total_wide = total.pivot_table(
        index=['date', 'roi'],
        columns='level',
        values=['total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
    ).reset_index()

    total_wide.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in total_wide.columns]
    total_wide.rename(
        columns={
            'total_ls_water_frac_adaptive_sr': 'ls_sr',
            'total_ls_water_frac_adaptive_toa': 'ls_toa',
            'total_s2_water_frac_adaptive_sr': 's2_sr',
            'total_s2_water_frac_adaptive_toa': 's2_toa'
        }, inplace=True)
    
    total_wide['ls_ac_abs_diff'] = (
        total_wide['ls_toa'] - total_wide['ls_sr']
    )
    total_wide['s2_ac_abs_diff'] = (
        total_wide['s2_toa'] - total_wide['s2_sr']
    )
    total_wide['ls_ac_rel_diff'] = (
        total_wide['ls_ac_abs_diff'] / ((total_wide['ls_toa'] + total_wide['ls_sr']) * 0.5) * 100
    )
    total_wide['s2_ac_rel_diff'] = (
        total_wide['s2_ac_abs_diff'] / ((total_wide['s2_toa'] + total_wide['s2_sr']) * 0.5) * 100
    )
    total_wide = total_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    total_wide['zone'] = 'total'

    """
    Reformat the lake data
    """

    """
    lake_sr_data = sr[['date', 'roi', 'level', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']]
    lake_toa_data = toa[['date', 'roi', 'level', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']]
    lake = pd.concat([lake_sr_data, lake_toa_data])

    lake_wide = lake.pivot_table(
        index=['date', 'roi'],
        columns='level',
        values=['lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
    ).reset_index()
    lake_wide.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in lake_wide.columns]
    lake_wide.rename(
        columns={
            'lake_ls_water_frac_adaptive_sr': 'ls_sr',
            'lake_ls_water_frac_adaptive_toa': 'ls_toa',
            'lake_s2_water_frac_adaptive_sr': 's2_sr',
            'lake_s2_water_frac_adaptive_toa': 's2_toa'
        }, inplace=True)
    lake_wide['ls_ac_abs_diff'] = (
        lake_wide['ls_toa'] - lake_wide['ls_sr']
    )
    lake_wide['s2_ac_abs_diff'] = (
        lake_wide['s2_toa'] - lake_wide['s2_sr']
    )
    lake_wide['ls_ac_rel_diff'] = (
        lake_wide['ls_ac_abs_diff'] / ((lake_wide['ls_toa'] + lake_wide['ls_sr']) * 0.5) * 100
    )
    lake_wide['s2_ac_rel_diff'] = (
        lake_wide['s2_ac_abs_diff'] / ((lake_wide['s2_toa'] + lake_wide['s2_sr']) * 0.5) * 100
    )
    lake_wide = lake_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    lake_wide['zone'] = 'lake'
    """

    """
    Reformat the shoreline data
    """
    shoreline_sr_data = sr[['date', 'roi', 'level', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']]
    shoreline_toa_data = toa[['date', 'roi', 'level', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']]
    shoreline = pd.concat([shoreline_sr_data, shoreline_toa_data])
    shoreline_wide = shoreline.pivot_table(
        index=['date', 'roi'],
        columns='level',
        values=['shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
    ).reset_index()
    shoreline_wide.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in shoreline_wide.columns]
    shoreline_wide.rename(
        columns={
            'shoreline_ls_water_frac_adaptive_sr': 'ls_sr',
            'shoreline_ls_water_frac_adaptive_toa': 'ls_toa',
            'shoreline_s2_water_frac_adaptive_sr': 's2_sr',
            'shoreline_s2_water_frac_adaptive_toa': 's2_toa'
        }, inplace=True)
    shoreline_wide['ls_ac_abs_diff'] = (
        shoreline_wide['ls_toa'] - shoreline_wide['ls_sr']
    )
    shoreline_wide['s2_ac_abs_diff'] = (
        shoreline_wide['s2_toa'] - shoreline_wide['s2_sr']
    )
    shoreline_wide['ls_ac_rel_diff'] = (
        shoreline_wide['ls_ac_abs_diff'] / ((shoreline_wide['ls_toa'] + shoreline_wide['ls_sr']) * 0.5) * 100
    )
    shoreline_wide['s2_ac_rel_diff'] = (
        shoreline_wide['s2_ac_abs_diff'] / ((shoreline_wide['s2_toa'] + shoreline_wide['s2_sr']) * 0.5) * 100
    )
    shoreline_wide = shoreline_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    shoreline_wide['zone'] = 'shoreline'

    """
    Reformat the buffer data
    """
    buffer_sr_data = sr[['date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']]
    buffer_toa_data = toa[['date', 'roi', 'level', 'buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']]
    buffer = pd.concat([buffer_sr_data, buffer_toa_data])
    buffer_wide = buffer.pivot_table(
        index=['date', 'roi'],
        columns='level',
        values=['buff_lake_ls_water_frac_adaptive', 'buff_lake_s2_water_frac_adaptive']
    ).reset_index()
    buffer_wide.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in buffer_wide.columns]
    buffer_wide.rename(
        columns={
            'buff_lake_ls_water_frac_adaptive_sr': 'ls_sr',
            'buff_lake_ls_water_frac_adaptive_toa': 'ls_toa',
            'buff_lake_s2_water_frac_adaptive_sr': 's2_sr',
            'buff_lake_s2_water_frac_adaptive_toa': 's2_toa'
        }, inplace=True)
    
    buffer_wide['ls_ac_abs_diff'] = (
        buffer_wide['ls_toa'] - buffer_wide['ls_sr']
    )
    buffer_wide['s2_ac_abs_diff'] = (
        buffer_wide['s2_toa'] - buffer_wide['s2_sr']
    )
    buffer_wide['ls_ac_rel_diff'] = (
        buffer_wide['ls_ac_abs_diff'] / ((buffer_wide['ls_toa'] + buffer_wide['ls_sr']) * 0.5) * 100
    )
    buffer_wide['s2_ac_rel_diff'] = (
        buffer_wide['s2_ac_abs_diff'] / ((buffer_wide['s2_toa'] + buffer_wide['s2_sr']) * 0.5) * 100
    )
    buffer_wide = buffer_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    buffer_wide['zone'] = 'Lake + Shoreline'

    """
    Combine the data
    """

    combined = pd.concat([total_wide, shoreline_wide, buffer_wide], ignore_index=True)

    """
    Generate boxplots and stats
    """
    abs_limits = (-10, 60)
    rel_limits = (-55, 110)

    custom_palette = {
        'total': '#1f77b4',       # blue
       # 'lake': '#ff7f0e',        # orange
        'shoreline': '#2ca02c',   # green
        'Lake + Shoreline': '#d62728'  # red
    }

    # Plots
    # Absolute LS8 AC differences
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=combined,
        x='zone',
        y='ls_ac_abs_diff',
        hue='zone',
        palette=custom_palette,
        legend=False,
    )
    plt.title(f'Absolute Landsat8 AC differences ({i})')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.ylabel('Absolute Difference (%)')
    plt.xlabel("Lake Position")
    plt.ylim(abs_limits)
    plt.show()

    # Relative LS8 AC differences
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=combined,
        x='zone',
        y='ls_ac_rel_diff',
        hue='zone',
        palette=custom_palette,
        legend=False,
    )
    plt.title(f'Relative Landsat8 AC differences ({i})')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.ylabel('Relative Difference (%)')
    plt.xlabel("Lake Position")
    plt.ylim(rel_limits)
    plt.show()

    # Absolute S2 AC differences
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=combined,
        x='zone',
        y='s2_ac_abs_diff',
        hue='zone',
        palette=custom_palette,
        legend=False,
    )
    plt.title(f'Absolute Sentinel2 AC differences ({i})')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.ylabel('Absolute Difference (%)')
    plt.xlabel("Lake Position")
    plt.ylim(abs_limits)
    plt.show()

    # Relative S2 AC differences
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=combined,
        x='zone',
        y='s2_ac_rel_diff',
        hue='zone',
        palette=custom_palette,
        legend=False,
    )
    plt.title(f'Relative Sentinel2 AC differences ({i})')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.ylabel('Relative Difference (%)')
    plt.xlabel("Lake Position")
    plt.ylim(rel_limits)
    plt.show()

    """
    Statistical tests and get stats
    """

    """
    Landsat8 Stats
    """
    # For total landscape
    temp = combined[combined['zone'] == 'total']

    total_ttest_abs = stats.ttest_1samp(temp['ls_ac_abs_diff'].dropna(), 0)
    total_ttest_rel = stats.ttest_1samp(temp['ls_ac_rel_diff'].dropna(), 0)
    total_ls_results = {
        'satellite': 'Landsat8',
        'resample_method': i,
        'zone': 'total',
        't_stat_abs': total_ttest_abs.statistic,
        'p_val_abs': total_ttest_abs.pvalue,
        't_stat_rel': total_ttest_rel.statistic,
        'p_val_rel': total_ttest_rel.pvalue,
        'mean_abs': temp['ls_ac_abs_diff'].mean(),
        'mean_rel': temp['ls_ac_rel_diff'].mean(),
        'var_abs': temp['ls_ac_abs_diff'].var(),
        'var_rel': temp['ls_ac_rel_diff'].var(),
        '75_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['ls_ac_abs_diff'].quantile(0.75) - temp['ls_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['ls_ac_rel_diff'].quantile(0.75) - temp['ls_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(total_ls_results)

    # For lakes
    # temp = combined[combined['zone'] == 'lake']

    # lake_ttest_abs = stats.ttest_1samp(temp['ls_ac_abs_diff'].dropna(), 0)
    # lake_ttest_rel = stats.ttest_1samp(temp['ls_ac_rel_diff'].dropna(), 0)
    # lake_ls_results = {
    #     'satellite': 'Landsat8',
    #     'resample_method': i,
    #     'zone': 'lake',
    #     't_stat_abs': lake_ttest_abs.statistic,
    #     'p_val_abs': lake_ttest_abs.pvalue,
    #     't_stat_rel': lake_ttest_rel.statistic,
    #     'p_val_rel': lake_ttest_rel.pvalue,
    #     'mean_abs': temp['ls_ac_abs_diff'].mean(),
    #     'mean_rel': temp['ls_ac_rel_diff'].mean(),
    #     'var_abs': temp['ls_ac_abs_diff'].var(),
    #     'var_rel': temp['ls_ac_rel_diff'].var(),
    #     '75_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.75),
    #     '25_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.25),
    #     '75_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.75),
    #     '25_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.25),
    #     'iqr_abs': temp['ls_ac_abs_diff'].quantile(0.75) - temp['ls_ac_abs_diff'].quantile(0.25),
    #     'iqr_rel': temp['ls_ac_rel_diff'].quantile(0.75) - temp['ls_ac_rel_diff'].quantile(0.25),
    # }
    # test_results.append(lake_ls_results)

    # For shorelines
    temp = combined[combined['zone'] == 'shoreline']
    shoreline_ttest_abs = stats.ttest_1samp(temp['ls_ac_abs_diff'].dropna(), 0)
    shoreline_ttest_rel = stats.ttest_1samp(temp['ls_ac_rel_diff'].dropna(), 0)
    shoreline_ls_results = {
        'satellite': 'Landsat8',
        'resample_method': i,
        'zone': 'shoreline',
        't_stat_abs': shoreline_ttest_abs.statistic,
        'p_val_abs': shoreline_ttest_abs.pvalue,
        't_stat_rel': shoreline_ttest_rel.statistic,
        'p_val_rel': shoreline_ttest_rel.pvalue,
        'mean_abs': temp['ls_ac_abs_diff'].mean(),
        'mean_rel': temp['ls_ac_rel_diff'].mean(),
        'var_abs': temp['ls_ac_abs_diff'].var(),
        'var_rel': temp['ls_ac_rel_diff'].var(),
        '75_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['ls_ac_abs_diff'].quantile(0.75) - temp['ls_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['ls_ac_rel_diff'].quantile(0.75) - temp['ls_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(shoreline_ls_results)

    # For Lakes + Shoreline
    temp = combined[combined['zone'] == 'Lake + Shoreline']
    buffer_ttest_abs = stats.ttest_1samp(temp['ls_ac_abs_diff'].dropna(), 0)
    buffer_ttest_rel = stats.ttest_1samp(temp['ls_ac_rel_diff'].dropna(), 0)
    buffer_ls_results = {
        'satellite': 'Landsat8',
        'resample_method': i,
        'zone': 'Lake + Shoreline',
        't_stat_abs': buffer_ttest_abs.statistic,
        'p_val_abs': buffer_ttest_abs.pvalue,
        't_stat_rel': buffer_ttest_rel.statistic,
        'p_val_rel': buffer_ttest_rel.pvalue,
        'mean_abs': temp['ls_ac_abs_diff'].mean(),
        'mean_rel': temp['ls_ac_rel_diff'].mean(),
        'var_abs': temp['ls_ac_abs_diff'].var(),
        'var_rel': temp['ls_ac_rel_diff'].var(),
        '75_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['ls_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['ls_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['ls_ac_abs_diff'].quantile(0.75) - temp['ls_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['ls_ac_rel_diff'].quantile(0.75) - temp['ls_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(buffer_ls_results)

    """
    Sentinel2 Stats
    """

    # For total landscape
    temp = combined[combined['zone'] == 'total']
    total_ttest_abs = stats.ttest_1samp(temp['s2_ac_abs_diff'].dropna(), 0)
    total_ttest_rel = stats.ttest_1samp(temp['s2_ac_rel_diff'].dropna(), 0)
    total_s2_results = {
        'satellite': 'Sentinel2',
        'resample_method': i,
        'zone': 'total',
        't_stat_abs': total_ttest_abs.statistic,
        'p_val_abs': total_ttest_abs.pvalue,
        't_stat_rel': total_ttest_rel.statistic,
        'p_val_rel': total_ttest_rel.pvalue,
        'mean_abs': temp['s2_ac_abs_diff'].mean(),
        'mean_rel': temp['s2_ac_rel_diff'].mean(),
        'var_abs': temp['s2_ac_abs_diff'].var(),
        'var_rel': temp['s2_ac_rel_diff'].var(),
        '75_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['s2_ac_abs_diff'].quantile(0.75) - temp['s2_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['s2_ac_rel_diff'].quantile(0.75) - temp['s2_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(total_s2_results)

    # For lakes
    # temp = combined[combined['zone'] == 'lake']
    # lake_ttest_abs = stats.ttest_1samp(temp['s2_ac_abs_diff'].dropna(), 0)
    # lake_ttest_rel = stats.ttest_1samp(temp['s2_ac_rel_diff'].dropna(), 0)  
    # lake_s2_results = {
    #     'satellite': 'Sentinel2',
    #     'resample_method': i,
    #     'zone': 'lake',
    #     't_stat_abs': lake_ttest_abs.statistic,
    #     'p_val_abs': lake_ttest_abs.pvalue,
    #     't_stat_rel': lake_ttest_rel.statistic,
    #     'p_val_rel': lake_ttest_rel.pvalue,
    #     'mean_abs': temp['s2_ac_abs_diff'].mean(),
    #     'mean_rel': temp['s2_ac_rel_diff'].mean(),
    #     'var_abs': temp['s2_ac_abs_diff'].var(),
    #     'var_rel': temp['s2_ac_rel_diff'].var(),
    #     '75_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.75),
    #     '25_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.25),
    #     '75_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.75),
    #     '25_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.25),
    #     'iqr_abs': temp['s2_ac_abs_diff'].quantile(0.75) - temp['s2_ac_abs_diff'].quantile(0.25),
    #     'iqr_rel': temp['s2_ac_rel_diff'].quantile(0.75) - temp['s2_ac_rel_diff'].quantile(0.25),
    # }
    # test_results.append(lake_s2_results)

    # For shorelines
    temp = combined[combined['zone'] == 'shoreline']
    shoreline_ttest_abs = stats.ttest_1samp(temp['s2_ac_abs_diff'].dropna(), 0)
    shoreline_ttest_rel = stats.ttest_1samp(temp['s2_ac_rel_diff'].dropna(), 0)
    shoreline_s2_results = {
        'satellite': 'Sentinel2',
        'resample_method': i,
        'zone': 'shoreline',
        't_stat_abs': shoreline_ttest_abs.statistic,
        'p_val_abs': shoreline_ttest_abs.pvalue,
        't_stat_rel': shoreline_ttest_rel.statistic,
        'p_val_rel': shoreline_ttest_rel.pvalue,
        'mean_abs': temp['s2_ac_abs_diff'].mean(),
        'mean_rel': temp['s2_ac_rel_diff'].mean(),
        'var_abs': temp['s2_ac_abs_diff'].var(),
        'var_rel': temp['s2_ac_rel_diff'].var(),
        '75_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['s2_ac_abs_diff'].quantile(0.75) - temp['s2_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['s2_ac_rel_diff'].quantile(0.75) - temp['s2_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(shoreline_s2_results)

    # For Lakes + Shoreline
    temp = combined[combined['zone'] == 'Lake + Shoreline']
    buffer_ttest_abs = stats.ttest_1samp(temp['s2_ac_abs_diff'].dropna(), 0)
    buffer_ttest_rel = stats.ttest_1samp(temp['s2_ac_rel_diff'].dropna(), 0)
    buffer_s2_results = {
        'satellite': 'Sentinel2',
        'resample_method': i,
        'zone': 'Lake + Shoreline',
        't_stat_abs': buffer_ttest_abs.statistic,
        'p_val_abs': buffer_ttest_abs.pvalue,
        't_stat_rel': buffer_ttest_rel.statistic,
        'p_val_rel': buffer_ttest_rel.pvalue,
        'mean_abs': temp['s2_ac_abs_diff'].mean(),
        'mean_rel': temp['s2_ac_rel_diff'].mean(),
        'var_abs': temp['s2_ac_abs_diff'].var(),
        'var_rel': temp['s2_ac_rel_diff'].var(),
        '75_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.75),
        '25_percentile_abs': temp['s2_ac_abs_diff'].quantile(0.25),
        '75_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.75),
        '25_percentile_rel': temp['s2_ac_rel_diff'].quantile(0.25),
        'iqr_abs': temp['s2_ac_abs_diff'].quantile(0.75) - temp['s2_ac_abs_diff'].quantile(0.25),
        'iqr_rel': temp['s2_ac_rel_diff'].quantile(0.75) - temp['s2_ac_rel_diff'].quantile(0.25),
    }
    test_results.append(buffer_s2_results)

    """
    Asses shoreline's role with Tukey's HSD
    """

    # For Landsat8
    ls_total = combined[combined['zone'] == 'total']['ls_ac_rel_diff'].dropna()
    #ls_lake = combined[combined['zone'] == 'lake']['ls_ac_rel_diff'].dropna()
    ls_shoreline = combined[combined['zone'] == 'shoreline']['ls_ac_rel_diff'].dropna()
    ls_buffer = combined[combined['zone'] == 'Lake + Shoreline']['ls_ac_rel_diff'].dropna()

    f_stat, p_val = stats.f_oneway(ls_total, ls_shoreline, ls_buffer)
    print(f"Landsat8... F-statistic: {f_stat}, p-value: {p_val}")

    ls_tukey_data = []
    # Add each observation with its proper group label
    for values, group_name in [
        (ls_total, 'total'), 
        (ls_shoreline, 'shoreline'), 
        (ls_buffer, 'Lake + Shoreline')
    ]:
        for val in values:
            ls_tukey_data.append({'zone': group_name, 'value': val})

    # Create DataFrame from list of dictionaries
    ls_tukey_df = pd.DataFrame(ls_tukey_data)
    model = ols('value ~ zone', data=ls_tukey_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(endog=ls_tukey_df['value'], groups=ls_tukey_df['zone'], alpha=0.05)
    print("LS8 Tukey's HSD results:")
    print(tukey)

    # Tukey's HSD for Sentinel2 AC
    s2_total = combined[combined['zone'] == 'total']['s2_ac_rel_diff'].dropna()
    #s2_lake = combined[combined['zone'] == 'lake']['s2_ac_rel_diff'].dropna()
    s2_shoreline = combined[combined['zone'] == 'shoreline']['s2_ac_rel_diff'].dropna()
    s2_buffer = combined[combined['zone'] == 'Lake + Shoreline']['s2_ac_rel_diff'].dropna()

    f_stat, p_val = stats.f_oneway(s2_total, s2_shoreline, s2_buffer)
    print(f"Sentinel2... F-statistic: {f_stat}, p-value: {p_val}")

    s2_tukey_data = []
    # Add each observation with its proper group label
    for values, group_name in [
        (s2_total, 'total'), 
       # (s2_lake, 'lake'),
        (s2_shoreline, 'shoreline'), 
        (s2_buffer, 'Lake + Shoreline')
    ]:
        for val in values:
            s2_tukey_data.append({'zone': group_name, 'value': val})

    # Create DataFrame from list of dictionaries
    s2_tukey_df = pd.DataFrame(s2_tukey_data)
    model = ols('value ~ zone', data=s2_tukey_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(endog=s2_tukey_df['value'], groups=s2_tukey_df['zone'], alpha=0.001)
    print("S2 Tukey's HSD results:")
    print(tukey)



# %%

full_results = pd.DataFrame(test_results)
# %%

print(full_results.head(10))

# %%

print_table = full_results[
    ['satellite', 'resample_method', 'zone', 'p_val_abs', 
     'p_val_rel', 'mean_abs', 'mean_rel', 'iqr_abs', 'iqr_rel']
].copy()

print(print_table)
# %%
