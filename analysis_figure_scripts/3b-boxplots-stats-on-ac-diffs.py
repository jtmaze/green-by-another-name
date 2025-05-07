# %% 1.0 Libraries and file paths

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
        total_wide['ls_ac_abs_diff'] / total_wide['ls_toa'] * 100
    )
    total_wide['s2_ac_rel_diff'] = (
        total_wide['s2_ac_abs_diff'] / total_wide['s2_toa'] * 100
    )
    total_wide = total_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    total_wide['zone'] = 'total'

    """
    Reformat the lake data
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
        lake_wide['ls_ac_abs_diff'] / lake_wide['ls_toa'] * 100
    )
    lake_wide['s2_ac_rel_diff'] = (
        lake_wide['s2_ac_abs_diff'] / lake_wide['s2_toa'] * 100
    )
    lake_wide = lake_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    lake_wide['zone'] = 'lake'

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
        shoreline_wide['ls_ac_abs_diff'] / shoreline_wide['ls_toa'] * 100
    )
    shoreline_wide['s2_ac_rel_diff'] = (
        shoreline_wide['s2_ac_abs_diff'] / shoreline_wide['s2_toa'] * 100
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
        buffer_wide['ls_ac_abs_diff'] / buffer_wide['ls_toa'] * 100
    )
    buffer_wide['s2_ac_rel_diff'] = (
        buffer_wide['s2_ac_abs_diff'] / buffer_wide['s2_toa'] * 100
    )
    buffer_wide = buffer_wide[['date', 'roi', 'ls_ac_abs_diff', 's2_ac_abs_diff', 'ls_ac_rel_diff', 's2_ac_rel_diff']]
    buffer_wide['zone'] = 'Lake + Shoreline'

    """
    Combine the data
    """

    combined = pd.concat([total_wide, lake_wide, shoreline_wide, buffer_wide], ignore_index=True)

    """
    Generate boxplots and stats
    """
    abs_limits = (-100, 100)
    rel_limits = (-100, 100)

    custom_palette = {
        'total': '#1f77b4',       # blue
        'lake': '#ff7f0e',        # orange
        'shoreline': '#2ca02c',   # green
        'Lake + Shoreline': '#d62728'  # red
    }

    # Relative Landsat8 AC boxplot and t-tests
    
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
    plt.ylim(rel_limits)

# %%
