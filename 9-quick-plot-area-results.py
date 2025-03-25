# %% 1.0 Import libraries and data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear30_area_summaries_batch1.csv')
toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear30_area_summaries_batch1.csv')
sr_bilinear60 = pd.read_csv('./data/lake_area_results/sr_resampled_bilinear60_area_summaries_batch1.csv')
toa_bilinear60 = pd.read_csv('./data/lake_area_results/toa_resampled_bilinear60_area_summaries_batch1.csv')
sr_lanczos30 = pd.read_csv('./data/lake_area_results/sr_resampled_lanczos30_area_summaries_batch1.csv')
toa_lanczos30 = pd.read_csv('./data/lake_area_results/toa_resampled_lanczos30_area_summaries_batch1.csv')
sr_lanczos60 = pd.read_csv('./data/lake_area_results/sr_resampled_lanczos60_area_summaries_batch1.csv')
toa_lanczos60 = pd.read_csv('./data/lake_area_results/toa_resampled_lanczos60_area_summaries_batch1.csv')
sr_noresample = pd.read_csv('./data/lake_area_results/toa_resampled_noresample_area_summaries_batch1.csv')
toa_noresample = pd.read_csv('./data/lake_area_results/sr_resampled_noresample_area_summaries_batch1.csv')

combined = pd.concat(
    [
        sr_bilinear30,
        toa_bilinear30,
        sr_bilinear60,
        toa_bilinear60,
        sr_lanczos30,
        toa_lanczos30,
        sr_lanczos60,
        toa_lanczos60,
        sr_noresample,
        toa_noresample
    ],
    ignore_index=True,
)

combined_valid = combined[combined['total_ls_water_frac_otsu'] != 'Poor Quality Image Data']

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
    resample_label: str
):
    if zone_label == 'Total':
        label_dict = {
            'total_ls_water_frac_otsu': 'Landsat8 - Otsu',
            'total_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
            'total_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
            'total_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
        }
    elif zone_label == 'Lake':
        label_dict = {
            'lake_ls_water_frac_otsu': 'Landsat8 - Otsu',
            'lake_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
            'lake_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
            'lake_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
        }
    elif zone_label == 'Shoreline':
        label_dict = {
            'shoreline_ls_water_frac_otsu': 'Landsat8 - Otsu',
            'shoreline_s2_water_frac_otsu': 'Sentinel-2 - Otsu',
            'shoreline_ls_water_frac_adaptive': 'Landsat8 - Adaptive',
            'shoreline_s2_water_frac_adaptive': 'Sentinel-2 - Adaptive'
        }
    else:
        print("Invalid zone label. Please use 'Total', 'Lake', or 'Shoreline'.")
        return

    melted_df = pd.melt(
        temp,
        id_vars=['level'],
        value_vars=cols_to_plot,
        var_name='measurement',
        value_name='water_fraction'
    )

    melted_df['water_fraction'] = melted_df['water_fraction'].astype(float)
    melted_df['label'] = melted_df['measurement'].map(label_dict)
    melted_df['display_name'] = melted_df['label'].fillna(melted_df['measurement'])

    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=melted_df,
        x='display_name',
        y='water_fraction',
        hue='level',
        palette={'sr': '#2ecc71', 'toa': '#3498db'}
    )
    plt.title(f'{zone_label} Water % Comparison (resampled {resample_label})')
    plt.xlabel('Satellite - Classification')
    plt.ylabel('Water Fraction %')
    plt.legend(title='Processing Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %% Bilinear 30 Total
"""
Compare bilinear 30m area stats
"""
cols_to_plot = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'bilinear30'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Total',
    resample_label='Bilinear 30'
)

cols_to_plot = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'bilinear30'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Lake',
    resample_label='Bilinear 30'
)
cols_to_plot = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'bilinear30'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Shoreline',
    resample_label='Bilinear 30'
)

# %% Unresampled
"""
Compare bilinear unresampled area stats
"""
cols_to_plot = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'noresample'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Total',
    resample_label='Un-resampled (native resolution)'
)

cols_to_plot = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'noresample'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Lake',
    resample_label='Unresampled (native resolution)'
)
cols_to_plot = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
temp = combined_valid[combined_valid['resample_method'] == 'noresample'].copy()
area_boxplot_maker(
    temp=temp,
    cols_to_plot=cols_to_plot,
    zone_label='Shoreline',
    resample_label='Unresampled (native resolution)'
)

# # %% Bilinear 60 
# """
# Compare bilinear 60m area stats
# """
# cols_to_plot = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'bilinear60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Total',
#     resample_label='Bilinear 60'
# )

# cols_to_plot = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'bilinear60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Lake',
#     resample_label='Bilinear 60'
# )
# cols_to_plot = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'bilinear60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Shoreline',
#     resample_label='Bilinear 60'
# )
# # %% Lanczos 30
# """"
# Compare lanczos 30m area stats
# """
# cols_to_plot = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos30'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Total',
#     resample_label='Lanczos 30'
# )

# cols_to_plot = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos30'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Lake',
#     resample_label='Lanczos 30'
# )
# cols_to_plot = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos30'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Shoreline',
#     resample_label='Lanczos 30'
# )

# # %% Lanczos 60
# """"
# Compare lanczos 60m area stats
# """
# cols_to_plot = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Total',
#     resample_label='Lanczos 60'
# )

# cols_to_plot = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Lake',
#     resample_label='Lanczos 60'
# )
# cols_to_plot = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']
# temp = combined_valid[combined_valid['resample_method'] == 'lanczos60'].copy()
# area_boxplot_maker(
#     temp=temp,
#     cols_to_plot=cols_to_plot,
#     zone_label='Shoreline',
#     resample_label='Lanczos 60'
# )



# %% Compare the TOA vs. SR differences by ROI


