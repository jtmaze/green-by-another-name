# %% 1.0 Import libraries and data
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch3.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch3.csv')
sr_noresample = pd.read_csv('./data/lake_area_results/toa_resampled_noresample_area_summaries_batch3.csv')
toa_noresample = pd.read_csv('./data/lake_area_results/sr_resampled_noresample_area_summaries_batch3.csv')

combined = pd.concat(
    [
        sr_bilinear30,
        toa_bilinear30,
        sr_noresample,
        toa_noresample
    ],
    ignore_index=True,
)

combined_valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']
combined_valid = combined_valid[
    combined_valid['pld_plus_valid_frac'] >= 80
].copy()

total_toa_img_pairs = len(combined_valid[
    (combined_valid['level'] == 'toa') &
    (combined_valid['resample_method'] == 'bilinear30')
])
print(total_toa_img_pairs)

total_sr_img_pairs = len(combined_valid[
    (combined_valid['level'] == 'sr') &
    (combined_valid['resample_method'] == 'bilinear30')
])
print(total_sr_img_pairs)

# %% Function for lake area boxplots by resampling method and zone ("Total", "Lake", "Shoreline")

def area_boxplot_maker(
    temp: pd.DataFrame, 
    cols_to_plot: list, 
    zone_label: str, 
):

    keep_cols = ['level', 'date', 'roi', 'resample_method'] + cols_to_plot
    temp = temp[keep_cols].copy()

    # Set up display names for the columns
    if zone_label == 'Total Landscape':
        label_dict = {
            'total_ls_water_frac_adaptive': 'Landsat8',
            'total_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Lake':
        label_dict = {
            'lake_ls_water_frac_adaptive': 'Landsat8',
            'lake_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Shoreline':
        label_dict = {
            'shoreline_ls_water_frac_adaptive': 'Landsat8',
            'shoreline_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Lake + Shoreline':
        label_dict = {
            'buff_lake_ls_water_frac_adaptive': 'Landsat8',
            'buff_lake_s2_water_frac_adaptive': 'Sentinel-2'
        }
    else:
        print("Invalid zone label. Please use 'Total', 'Lake', or 'Shoreline'.")
        return
    
    # Create grouping column combining level and resample method
    temp['group'] = temp['level'] + '_' + temp['resample_method']
    
    # Reshape data for plotting
    melted_df = pd.melt(
        temp,
        id_vars=['level', 'resample_method', 'date', 'roi', 'group'],
        value_vars=cols_to_plot,
        var_name='satellite',
        value_name='water_fraction'
    )
    
    # Map satellite column to more readable names
    melted_df['satellite_name'] = melted_df['satellite'].map(label_dict)
    
    # Convert water fraction to float
    melted_df['water_fraction'] = melted_df['water_fraction'].astype(float)
    
    # Create and display the plot
    order = ['sr_bilinear30', 'toa_bilinear30', 'sr_noresample', 'toa_noresample']
    new_labels = ['SR Bilinear 30m', 'TOA Bilinear 30m', 'SR Native', 'TOA Native']

    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(
        data=melted_df,
        x='group',
        y='water_fraction',
        hue='satellite_name',
        palette={'Landsat8': '#ff9933', 'Sentinel-2': '#9370DB'},
        order=order
    )
    ax.set_xticklabels(new_labels)
    plt.title(f'{zone_label} Water Fraction Comparison')
    plt.xlabel('Processing Level + Resampling Method')
    plt.ylabel('Water Fraction %')
    plt.legend(title='Satellite')
    plt.tight_layout()
    plt.show()

# %% Bilinear 30 Total
"""
Compare bilinear 30m area boxplots
"""
temp = combined_valid[
    (combined_valid['resample_method'] == 'bilinear30') |
    (combined_valid['resample_method'] == 'noresample')
]
cols_to_plot = [
    'total_ls_water_frac_adaptive',
    'total_s2_water_frac_adaptive',
]

area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Total Landscape',
)

cols_to_plot = [
    'lake_ls_water_frac_adaptive',
    'lake_s2_water_frac_adaptive',
]

area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Lake',
)

cols_to_plot = [
    'shoreline_ls_water_frac_adaptive',
    'shoreline_s2_water_frac_adaptive',
]
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Shoreline',
)

cols_to_plot = [
    'buff_lake_ls_water_frac_adaptive',
    'buff_lake_s2_water_frac_adaptive',
]
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Lake + Shoreline',
)

# %% Scratch function, will delete later

"""
This function compares both Otsu and Adaptive, but not useful in final analysis
"""
# def area_boxplot_maker(
#     temp: pd.DataFrame, 
#     cols_to_plot: list, 
#     zone_label: str, 
#     resample_label: str
# ):
#     if zone_label == 'Total':
#         label_dict = {
#             'total_ls_water_frac_otsu': 'Landsat8 - Otsu',
#             'total_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
#             'total_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
#             'total_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
#         }
#     elif zone_label == 'Lake':
#         label_dict = {
#             'lake_ls_water_frac_otsu': 'Landsat8 - Otsu',
#             'lake_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
#             'lake_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
#             'lake_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
#         }
#     elif zone_label == 'Shoreline':
#         label_dict = {
#             'shoreline_ls_water_frac_otsu': 'Landsat8 - Otsu',
#             'shoreline_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
#             'shoreline_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
#             'shoreline_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
#         }
#     elif zone_label == 'Lake + Shoreline':
#         label_dict = {
#             'buff_lake_ls_water_frac_otsu': 'Landsat8 - Otsu',
#             'buff_lake_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
#             'buff_lake_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
#             'buff_lake_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
#         }
#     else:
#         print("Invalid zone label. Please use 'Total', 'Lake', or 'Shoreline'.")
#         return

#     melted_df = pd.melt(
#         temp,
#         id_vars=['level'],
#         value_vars=cols_to_plot,
#         var_name='measurement',
#         value_name='water_fraction'
#     )

#     melted_df['water_fraction'] = melted_df['water_fraction'].astype(float)
#     melted_df['label'] = melted_df['measurement'].map(label_dict)
#     melted_df['display_name'] = melted_df['label'].fillna(melted_df['measurement'])

#     plt.figure(figsize=(14, 7))
#     sns.boxplot(
#         data=melted_df,
#         x='display_name',
#         y='water_fraction',
#         hue='level',
#         palette={'sr': '#2ecc71', 'toa': '#3498db'}
#     )
#     plt.title(f'{zone_label} Water % Comparison (resampled {resample_label})')
#     plt.xlabel('Satellite - Classification')
#     plt.ylabel('Water Fraction %')
#     plt.legend(title='Processing Level', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()


