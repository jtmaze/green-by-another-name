# %% 1.0 Look at some reflectance comparisons

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reflectance_comp_dir = './data/regression_summaries'
# %%

lake_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_0m_lake_bilinear30.csv')
lake_plus_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_60m_lake_bilinear30.csv')
land_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_60m_land_bilinear30.csv')
shoreline_data = pd.read_csv(f'{reflectance_comp_dir}/regression_summaries_shoreline_neg60-60_bilinear30.csv')

# %%