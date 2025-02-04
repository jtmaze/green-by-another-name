import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_otsu_histograms(
    sr: pd.Series,
    toa: pd.Series,
    date: str,
    roi: str):
    
    """
    Plots the NDWI histograms for Landsat and Sentinel-2 images
    """
    
    # If the data was written to disk, it will be a string, 
    # so convert it back to a dictionary
    # The ast library should do this
    # LS SR
    ls_sr_threshold = sr['ls_threshold'].iloc[0]
    ls_sr_hist = np.array(
        ast.literal_eval(
            sr['ls_hist_counts'].iloc[0]
        )
    )
    ls_sr_water_frac = sr['ls_water_frac'].iloc[0]
    # S2 SR
    s2_sr_threshold = sr['s2_threshold'].iloc[0]
    s2_sr_hist = np.array(
        ast.literal_eval(
            sr['s2_hist_counts'].iloc[0]
        )
    )
    s2_sr_water_frac = sr['s2_water_frac'].iloc[0]

    # LS TOA
    ls_toa_threshold = toa['ls_threshold'].iloc[0]
    ls_toa_hist = np.array(
        ast.literal_eval(
            toa['ls_hist_counts'].iloc[0]
        )
    )
    ls_toa_water_frac = toa['ls_water_frac'].iloc[0]
    # S2 TOA
    s2_toa_threshold = toa['s2_threshold'].iloc[0]
    s2_toa_hist = np.array(
        ast.literal_eval(
            toa['s2_hist_counts'].iloc[0]
        )
    )
    s2_toa_water_frac = toa['s2_water_frac'].iloc[0]

    # Should all have the same bin edges...
    bin_edges = np.array(ast.literal_eval(sr['ls_hist_bins'].iloc[0]))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Plot histogram lines
    ax.plot(bin_centers, ls_sr_hist, linestyle='-', color='red', label='Landsat SR', linewidth=2)
    ax.plot(bin_centers, s2_sr_hist, linestyle='-', color='green', label='Sentinel-2 SR', linewidth=2)
    ax.plot(bin_centers, ls_toa_hist, linestyle=':', color='red', label='Landsat TOA', linewidth=4)
    ax.plot(bin_centers, s2_toa_hist, linestyle=':', color='green', label='Sentinel-2 TOA', linewidth=4)

    # Add threshold lines
    ax.axvline(x=ls_sr_threshold, color='red', linestyle='-', linewidth=3)
    ax.axvline(x=s2_sr_threshold, color='green', linestyle='-', linewidth=3)
    ax.axvline(x=ls_toa_threshold, color='red', linestyle=':', linewidth=3)
    ax.axvline(x=s2_toa_threshold, color='green', linestyle=':', linewidth=3)

    # Text boxes
    ax.text(
        0.95, 0.95,
        f'LS SR water frac: {ls_sr_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='red')
    )
    ax.text(
        0.95, 0.89,
        f'S2 SR water frac: {s2_sr_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='green')
    )
    ax.text(
        0.95, 0.80,
        f'LS TOA water frac: {ls_toa_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='red')
    )
    ax.text(
        0.95, 0.74,
        f'S2 TOA water frac: {s2_toa_water_frac:.1f}',
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=1, edgecolor='green')
    )

    # Customize plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of NDWI Values for {date} in {roi}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reflectance_histograms(
    sr: pd.Series, 
    toa: pd.Series, 
    band: str,
    date: str, 
    roi: str,
    hist_range: tuple[float, float] = (0.0, 0.1)
    ):

    """
    Plots the Green or NIR histograms for Landsat and Sentinel-2 TOA/SR images
    """
    # Access histograms for ls and s2 images
    ls_sr_hist_data = np.array(
        ast.literal_eval(
            sr['ls_hist_counts'].iloc[0]
        )
    )
    ls_toa_hist_data = np.array(
        ast.literal_eval(
            toa['ls_hist_counts'].iloc[0]
        )
    )
    s2_sr_hist_data = np.array(
        ast.literal_eval(
            sr['s2_hist_counts'].iloc[0]
        )
    )
    s2_toa_hist_data = np.array(
        ast.literal_eval(
            toa['s2_hist_counts'].iloc[0]
        )
    )


    # Should all have the same bin edges...
    bin_edges = np.array(ast.literal_eval(sr['ls_hist_bins'].iloc[0]))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ls_sr_mean = np.average(bin_centers, weights=np.array(ls_sr_hist_data))
    ls_toa_mean = np.average(bin_centers, weights=np.array(ls_toa_hist_data))
    s2_sr_mean = np.average(bin_centers, weights=np.array(s2_sr_hist_data))
    s2_toa_mean = np.average(bin_centers, weights=np.array(s2_toa_hist_data))

    mask = (bin_centers >= hist_range[0]) & (bin_centers <= hist_range[1])
    
    # Crop the bin centers and histograms by the mask
    bin_centers_cropped = bin_centers[mask]
    ls_sr_hist_cropped = np.array(ls_sr_hist_data)[mask]
    ls_toa_hist_cropped = np.array(ls_toa_hist_data)[mask]
    s2_sr_hist_cropped = np.array(s2_sr_hist_data)[mask]
    s2_toa_hist_cropped = np.array(s2_toa_hist_data)[mask]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_centers_cropped, ls_sr_hist_cropped, linestyle='-', color='red', label='Landsat SR', linewidth=2)
    ax.plot(bin_centers_cropped, s2_sr_hist_cropped, linestyle='-', color='green', label='Sentinel-2 SR', linewidth=2)
    ax.plot(bin_centers_cropped, ls_toa_hist_cropped, linestyle=':', color='red', label='Landsat TOA', linewidth=4)
    ax.plot(bin_centers_cropped, s2_toa_hist_cropped, linestyle=':', color='green', label='Sentinel-2 TOA', linewidth=4)

    # Add histogram means
    ax.axvline(x=ls_sr_mean, color='red', linestyle='-', linewidth=3)
    ax.axvline(x=s2_sr_mean, color='green', linestyle='-', linewidth=3)
    ax.axvline(x=ls_toa_mean, color='red', linestyle=':', linewidth=3)
    ax.axvline(x=s2_toa_mean, color='green', linestyle=':', linewidth=3)

    # Customize plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {band} values for {date} in {roi}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    #plt.show()

def overlay_sr_toa_hist_otsu(summary_data: pd.DataFrame, roi: str, date: str):
    
    sr = summary_data[
        (summary_data['roi'] == roi) & 
        (summary_data['date'] == date) & 
        (summary_data['level'] == 'sr')
    ]

    toa = summary_data[
        (summary_data['roi'] == roi) & 
        (summary_data['date'] == date) & 
        (summary_data['level'] == 'toa')
    ]

    if sr.empty or toa.empty:
        print('Error: a level is missing')
        return None

    plot_otsu_histograms(sr, toa, date, roi)

def overlay_sr_toa_hist_reflect(regression_data: pd.DataFrame,
                                roi: str,
                                date: str,
                                band_name: str,
                                hist_range: tuple[float, float] = (0.0, 0.1)
                                ):
    
    sr = regression_data[
        (regression_data['roi'] == roi) &
        (regression_data['date'] == date) &
        (regression_data['level'] == 'sr') &
        (regression_data['band_name'] == band_name)
    ]

    toa = regression_data[
        (regression_data['roi'] == roi) &
        (regression_data['date'] == date) &
        (regression_data['level'] == 'toa') &
        (regression_data['band_name'] == band_name)
    ]

    if sr.empty or toa.empty:
        print('Error: a level is missing')
        return None

    plot_reflectance_histograms(sr, toa, band_name, date, roi, hist_range)