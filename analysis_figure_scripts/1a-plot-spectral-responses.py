# %% 1.0 Libraries and file paths
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


os.chdir('/Users/jmaze/Documents/projects/green-by-another-name/')

# %% 2.0 Read the data

s2_srf_a = pd.read_excel(
    './data/sentinel2_spectral_response.xlsx',
    sheet_name='Spectral Responses (S2A)',
)

s2_srf_a = s2_srf_a.rename(
    columns={
        'SR_WL': 'Wavelength',
        'S2A_SR_AV_B8': 'S2_NIR',
        'S2A_SR_AV_B3': 'S2_Green',
        'S2A_SR_AV_B4': 'S2_Red',
        'S2A_SR_AV_B2': 'S2_Blue',
    }
)

s2_nir = s2_srf_a[['Wavelength', 'S2_NIR']].copy()
s2_g = s2_srf_a[['Wavelength', 'S2_Green']].copy()
s2_r = s2_srf_a[['Wavelength', 'S2_Red']].copy()
s2_b = s2_srf_a[['Wavelength', 'S2_Blue']].copy()

# s2_srf_b = pd.read_excel(
#     './data/sentinel2_spectral_response.xlsx',
#     sheet_name='Spectral Responses (S2B)',
# )


ls8_g = pd.read_excel(
    './data/ls8_spectral_response.xlsx',
    sheet_name='Green',
)

ls8_g = ls8_g.rename(
    columns={
         'BA RSR [watts]': 'LS8_Green',
    }
)
ls8_g = ls8_g[['Wavelength', 'LS8_Green']].copy()

ls8_nir = pd.read_excel(
    './data/ls8_spectral_response.xlsx',
    sheet_name='NIR',
)

ls8_nir = ls8_nir.rename(
    columns={
         'BA RSR [watts]': 'LS8_NIR',
    }
)
ls8_nir = ls8_nir[['Wavelength', 'LS8_NIR']].copy()

ls8_r = pd.read_excel(
    './data/ls8_spectral_response.xlsx',
    sheet_name='Red',
)

ls8_r = ls8_r.rename(
    columns={
         'BA RSR [watts]': 'LS8_Red',
    }
)
ls8_r = ls8_r[['Wavelength', 'LS8_Red']].copy()

ls8_b = pd.read_excel(
    './data/ls8_spectral_response.xlsx',
    sheet_name='Blue',
)
ls8_b = ls8_b.rename(
    columns={
         'BA RSR [watts]': 'LS8_Blue',
    }
)
ls8_b = ls8_b[['Wavelength', 'LS8_Blue']].copy()

# %% 3.0 Combine the data

combined = pd.concat(
    [
        s2_nir,
        s2_g,
        s2_r,
        s2_b,
        ls8_nir,
        ls8_g,
        ls8_r,
        ls8_b,
    ],
    axis=0
)

# %% Plot the Relage Spectral Response (RSR) curves

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plot_cols = [
    'S2_NIR', 'S2_Green', 'S2_Red', 'S2_Blue', 
    'LS8_NIR', 'LS8_Green', 'LS8_Red', 'LS8_Blue'
]

fig, ax = plt.subplots(figsize=(12, 7))

# Define colors for each band
band_colors = {
    'NIR': 'maroon',
    'Green': 'green',
    'Red': 'red',
    'Blue': 'blue'
}

# ---- 1.  Plot the four response curves  ----
for col in plot_cols:
    sat, band = col.split('_')
    ax.plot(
        combined['Wavelength'],
        combined[col],
        linestyle='--' if sat == 'S2' else '-',
        color=band_colors[band],
        linewidth=2,
    )

ax.set_xlabel('Wavelength (nm)', fontsize=20)
ax.set_ylabel('Relative Spectral Response', fontsize=18)
ax.set_xlim(425, 925)
ax.set_ylim(0.01, 1.05)

# ---- 2.  Build two mini-legends: one for sensor (line style) & one for band (color)  ----
style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Sentinel-2 (MSI)'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='Landsat 8 (OLI)')
]
band_handles = [
    Line2D([0], [0], color='blue',   lw=2, label='Blue'),
    Line2D([0], [0], color='green',  lw=2, label='Green'),
    Line2D([0], [0], color='red',    lw=2, label='Red'),
    Line2D([0], [0], color='maroon', lw=2, label='NIR')
]

# First legend: sensor styles
leg1 = ax.legend(handles=style_handles,
                 loc='upper center',
                 bbox_to_anchor=(0.15, -0.10),  # Position below x-axis
                 frameon=True,
                 title='Sensor',
                 fontsize=14,
                 title_fontsize=16,
                 edgecolor='black')
# Second legend: spectral bands
leg2 = ax.legend(handles=band_handles,
                 loc='upper center',
                 bbox_to_anchor=(0.75, -0.15),  # Position below x-axis
                 ncol=4, 
                 frameon=True,
                 title='Band',
                 fontsize=14,
                 title_fontsize=16,
                 edgecolor='black')
# Keep both legends
ax.add_artist(leg1)
ax.tick_params(axis='both', which='major', labelsize=14)

# Add extra bottom margin to make room for legends
plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()

# %%
