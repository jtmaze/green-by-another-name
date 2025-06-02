# %% 1.0 Import libraries and data
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

toa_bilinear30 = pd.read_csv('./data/lake_area_results/toa_resampled_cubic30_area_summaries_batch3.csv')
print(len(toa_bilinear30))
valids = toa_bilinear30[['roi', 'date']].agg('_'.join, axis=1).unique()

sr_bilinear30 = pd.read_csv('./data/lake_area_results/sr_resampled_cubic30_area_summaries_batch3.csv')
sr_bilinear30 = sr_bilinear30[sr_bilinear30[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

sr_noresample = pd.read_csv('./data/lake_area_results/toa_resampled_noresample_area_summaries_batch6.csv')
sr_noresample = sr_noresample[sr_noresample[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

toa_noresample = pd.read_csv('./data/lake_area_results/sr_resampled_noresample_area_summaries_batch6.csv')
toa_noresample = toa_noresample[toa_noresample[['roi', 'date']].agg('_'.join, axis=1).isin(valids)]

combined = pd.concat(
    [
        sr_bilinear30,
        toa_bilinear30,
        sr_noresample,
        toa_noresample
    ],
    ignore_index=True,
)

print(len(combined))

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
            'total_ls_water_frac_adaptive': 'Landsat 8',
            'total_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Lake':
        label_dict = {
            'lake_ls_water_frac_adaptive': 'Landsat 8',
            'lake_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Shoreline':
        label_dict = {
            'shoreline_ls_water_frac_adaptive': 'Landsat 8',
            'shoreline_s2_water_frac_adaptive': 'Sentinel-2'
        }
    elif zone_label == 'Lake + Shoreline':
        label_dict = {
            'buff_lake_ls_water_frac_adaptive': 'Landsat 8',
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
    
        # Setup plot
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Define the groups and colors
    order = ['toa_cubic30', 'toa_noresample', 'sr_cubic30', 'sr_noresample']
    new_labels = ['TOA cubic 30m', 'TOA Unresampled', 'SR cubic 30m', 'SR Unresampled']
    satellite_colors = {'Landsat 8': '#ff9933', 'Sentinel-2': '#9370DB'}
    
    # Calculate positions for the boxes
    positions = []
    group_width = 1.0
    box_width = 0.35  # Width of each box
    gap = 0.05        # Gap between satellite boxes
    legend_elements = []
    
    # For each group (SR/TOA + resampling method)
    for i in range(len(order)):
        # Add two boxes side by side for Landsat and Sentinel
        base_pos = i * group_width
        positions.append([base_pos - box_width/2 - gap/2, base_pos + box_width/2 + gap/2])
    
    # Plot each boxplot manually
    for i, group in enumerate(order):
        for j, (col, sat_name) in enumerate(label_dict.items()):
            # Get data for this boxplot
            data = temp[temp['group'] == group][col].astype(float)
            
            # Create boxplot
            bp = ax.boxplot(
                data,
                positions=[positions[i][j]],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )
            
            # Style the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(satellite_colors[sat_name])
                patch.set_alpha(0.75)
                if 'noresample' in group:
                    patch.set_hatch('///')
                patch.set_edgecolor('black')
            
            # Style the median lines
            for median in bp['medians']:
                median.set(color='black', linewidth=2)

            if i == 0 and j == 0:
                # Create proper Rectangle patches for legend
                for sat_name, color in satellite_colors.items():
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', alpha=0.75, label=sat_name)
                    )
                
                legend_elements.append(
                    plt.Rectangle((0,0), 1, 1, facecolor='lightgray', edgecolor='black', label='bilinear 30 meters')
                )
                legend_elements.append(
                    plt.Rectangle((0,0), 1, 1, facecolor='lightgray', hatch='//', edgecolor='black', label='unresampled')
                )

    # After all boxplot plotting, add a vertical divider between TOA and SR
    ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.9, linewidth=2.5)

    # ax.legend(handles=legend_elements,
    #           loc='upper center',
    #           bbox_to_anchor=(0.5, -0.15),
    #           ncol=2,
    #           frameon=True,
    #           fontsize=12)

    # Set x-axis ticks and labels
    ax.set_xticks([0.45, 2.6])
    ax.set_xticklabels(['TOA', 'SR'], fontsize=16)
    
    # Set y-axis label
    ax.set_ylabel(f"{zone_label} Water Fraction %", fontsize=16)
    
    # Clean up the plot
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print mean values for each series after the boxplot is rendered
    print(f"\n--- Mean Water Fraction Values for {zone_label} ---")
    for group in order:
        for satellite_name in label_dict.values():
            mean_value = melted_df[(melted_df['group'] == group) & 
                                  (melted_df['satellite_name'] == satellite_name)]['water_fraction'].mean()
            print(f"{new_labels[order.index(group)]} - {satellite_name}: {mean_value:.2f}%")
        print("")

# %% Bilinear 30 Total
"""
Compare bilinear 30m area boxplots
"""
temp = combined[
    (combined['resample_method'] == 'cubic30') |
    (combined['resample_method'] == 'noresample')
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


