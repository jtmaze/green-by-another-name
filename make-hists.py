comp_params = {
    'roi': 'MRD_sub1',
    'date': '2024-06-15'
}

overlay_sr_toa_hist_otsu(summary_data=df_area_summary, roi=comp_params['roi'], date=comp_params['date'])

overlay_sr_toa_hist_reflect(regression_data=regression_summary_clean, roi=comp_params['roi'], date=comp_params['date'], band_name='Green', hist_range=(0.0, 0.1))
overlay_sr_toa_hist_reflect(regression_data=regression_summary_clean, roi=comp_params['roi'], date=comp_params['date'], band_name='NIR', hist_range=(0.0, 0.10))

