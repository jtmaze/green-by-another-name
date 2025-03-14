# %%
# 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

area_test = pd.read_csv('./data/test_area_thresholding_summary.csv')

# %%

cols_to_plot1 = ['total_ls_water_frac_otsu', 'total_s2_water_frac_otsu', 'total_ls_water_frac_adaptive', 'total_s2_water_frac_adaptive']
cols_to_plot2 = ['lake_ls_water_frac_otsu', 'lake_s2_water_frac_otsu', 'lake_ls_water_frac_adaptive', 'lake_s2_water_frac_adaptive']
cols_to_plot3 = ['shoreline_ls_water_frac_otsu', 'shoreline_s2_water_frac_otsu', 'shoreline_ls_water_frac_adaptive', 'shoreline_s2_water_frac_adaptive']

# Method 1: Using seaborn's boxplot directly with melted data
plt.figure(figsize=(12, 6))
# Melt the dataframe to get it into the right format for seaborn
melted_df = pd.melt(area_test[cols_to_plot3])
sns.boxplot(x='variable', y='value', data=melted_df)
plt.title('Comparison of Thresholds and Water Fractions')
plt.xticks(rotation=45)  # Rotate labels if needed
plt.tight_layout()
plt.show()

# %% Quick stats

# T-test for otsu thresholding between Landsat and Sentinel-2
otsu_ttest = stats.ttest_rel(
    area_test['total_ls_water_frac_otsu'], 
    area_test['total_s2_water_frac_otsu'],
    nan_policy='omit'  # Handle any NaN values
)

# T-test for adaptive thresholding between Landsat and Sentinel-2
adaptive_ttest = stats.ttest_rel(
    area_test['total_ls_water_frac_adaptive'], 
    area_test['total_s2_water_frac_adaptive'],
    nan_policy='omit'  # Handle any NaN values
)

# Print results with interpretation
alpha = 0.05  # Significance level

print("T-test results for Otsu thresholding (Landsat vs Sentinel-2):")
print(f"  T-statistic: {otsu_ttest.statistic:.4f}")
print(f"  P-value: {otsu_ttest.pvalue:.4f}")
print(f"  Interpretation: {'Significantly different' if otsu_ttest.pvalue < alpha else 'Not significantly different'} at α={alpha}")
print()

print("T-test results for Adaptive thresholding (Landsat vs Sentinel-2):")
print(f"  T-statistic: {adaptive_ttest.statistic:.4f}")
print(f"  P-value: {adaptive_ttest.pvalue:.4f}")
print(f"  Interpretation: {'Significantly different' if adaptive_ttest.pvalue < alpha else 'Not significantly different'} at α={alpha}")
print()

# %%
