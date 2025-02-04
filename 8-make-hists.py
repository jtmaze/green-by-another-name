# %%
# 
import pandas as pd
from image_analysis_functions import overlay_sr_toa_hist_otsu
from image_analysis_functions import overlay_sr_toa_hist_reflect


regression_data = pd.read_csv('./data/regression_summaries_60m_lake.csv')

area_data = pd.read_csv('./data/area_data_no_dict.csv')

# %%

comp_params = {
    'roi': 'MRD_sub1',
    'date': '2020-06-11'
}


test_df = area_data[(area_data['roi'] == comp_params['roi']) &
                    (area_data['date'] == comp_params['date']) &
                    (area_data['level'] == 'sr')
                ]                    

overlay_sr_toa_hist_otsu(summary_data=area_data, roi=comp_params['roi'], date=comp_params['date'])
#overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=comp_params['roi'], date=comp_params['date'], band_name='Green', hist_range=(0.0, 0.1))
#overlay_sr_toa_hist_reflect(regression_data=regression_data, roi=comp_params['roi'], date=comp_params['date'], band_name='NIR', hist_range=(0.0, 0.1))

