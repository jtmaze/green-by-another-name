# %%
# 
import pandas as pd
from histogram_functions import overlay_sr_toa_hist_otsu
from histogram_functions import overlay_sr_toa_hist_reflect


regression_data = pd.read_csv('./data/test_60m_lake.csv')

area_data = pd.read_csv('./data/area_data_no_dict.csv')

# %%

comp_params = {
    'roi': 'MRD_sub1',
    'date': '2019-06-18'
}


#overlay_sr_toa_hist_otsu(summary_data=area_data, roi=comp_params['roi'], date=comp_params['date'])
overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=comp_params['roi'], date=comp_params['date'], band_name='Green', hist_range=(0.0, 0.1))
#overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=comp_params['roi'], date=comp_params['date'], band_name='NIR', hist_range=(0.0, 0.1))

