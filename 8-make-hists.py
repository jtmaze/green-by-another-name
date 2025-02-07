# %%
# 
import pandas as pd
from histogram_functions import overlay_sr_toa_hist_otsu
from histogram_functions import overlay_sr_toa_hist_reflect


regression_data = pd.read_csv('./data/regression_summaries_60m_lake.csv')

area_data = pd.read_csv('./data/area_data_v1.csv')

# %% 
candidate_data = area_data[area_data['level'] == 'sr']
candidate_data = candidate_data.sort_values('ls_s2_percent_diff', ascending=False)
candidate_data = candidate_data[(candidate_data['ls_s2_percent_diff'] > 10)]


# %%

for _, r in candidate_data.iterrows():
    roi = r['roi']
    date = r['date']
    overlay_sr_toa_hist_otsu(summary_data=area_data, roi=roi, date=date)
    overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=roi, date=date, band_name='Green', hist_range=(0.0, 0.1))
    overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=roi, date=date, band_name='NIR', hist_range=(0.0, 0.1))

