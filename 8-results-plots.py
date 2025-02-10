# %% 

import pandas as pd
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %%

area_data = pd.read_csv('./data/area_data_v1.csv')
area_data.loc[:, 'month'] = pd.to_datetime(area_data.loc[:, 'date']).dt.month
regression_data = pd.read_csv('./data/regression_summaries_60m_lake.csv')

roi_pair_counts = area_data.groupby(['roi', 'level']).size().reset_index(name='counts')
roi_pair_counts.head(15)


# %% Box plot to compare lake areas with sr and toa data

area_data_clean = area_data.copy()
area_data_clean = area_data_clean[
    (area_data['ls_s2_percent_diff'] > -25) &
    (area_data['ls_s2_percent_diff'] < 25)
]

sr_data = area_data_clean[area_data_clean['level'] == 'sr']
toa_data = area_data_clean[area_data_clean['level'] == 'toa']
sr_data['ls_s2_percent_diff'] = pd.to_numeric(sr_data['ls_s2_percent_diff'], errors='coerce')
toa_data['ls_s2_percent_diff'] = pd.to_numeric(toa_data['ls_s2_percent_diff'], errors='coerce')

print(len(sr_data), len(toa_data))

plt.figure(figsize=(10, 6))
box = plt.boxplot([sr_data['ls_s2_percent_diff'], toa_data['ls_s2_percent_diff']],
                  patch_artist=True,
                  labels=['SR', 'TOA'])

for patch in box['boxes']:
    patch.set_facecolor('skyblue')
    patch.set_alpha(1)

plt.axhline(y=0, color='red', linestyle='--')

plt.title('LS8 vs S2 area difference ((LS8 - S2) / LS8)')
plt.xlabel('Processing Level')
plt.ylabel('Percent Lake Area Difference')
plt.show(True)

# %% TOA T-test to see if LS8 is biased higher
toa_data = area_data[area_data['level'] == 'toa']
plt.hist(toa_data['ls_s2_percent_diff'], bins=50)
plt.xlabel('LS8 vs S2 area difference ((LS8 - S2) / LS8)')
plt.title("TOA area comparison for Sentinel-2 and Landsat8")
plt.axvline(x=0, color="red", linestyle='--', linewidth=2)

t_stat, p_val = stats.ttest_1samp(toa_data['ls_s2_percent_diff'], popmean=0)

print(t_stat)
print(p_val)

# %% TOA T-test to see if LS8 is biased higher
temp = toa_data[(toa_data['month'] == 9) | (toa_data['month'] == 8)]
#temp = toa_data[(toa_data['month'] == 7)]
# temp = temp[(temp['ls_s2_percent_diff'] < 30) &
#             (temp['ls_s2_percent_diff'] > -30)
# ]

print(f'n={len(temp)}')
plt.hist(temp['ls_s2_percent_diff'], bins=20)
plt.xlabel('LS8 vs S2 area difference ((LS8 - S2) / LS8)')
plt.title("Aug & Sep TOA area comparison for Sentinel-2 and Landsat8")
plt.axvline(x=0, color="red", linestyle='--', linewidth=2)

t_stat, p_val = stats.ttest_1samp(temp['ls_s2_percent_diff'], popmean=0)

print(t_stat)
print(p_val)

# %% Two-Sample t-test June vs August/Sep

toa_data_trimmed= toa_data[
    (toa_data['ls_s2_percent_diff'] < 30) &
    (toa_data['ls_s2_percent_diff'] > -30)
]

june = toa_data_trimmed[toa_data_trimmed['month'] == 6]
aug_sep = toa_data_trimmed[(toa_data_trimmed['month'] == 9) | (toa_data_trimmed['month'] == 8)]

t_stat, p_val = stats.ttest_ind(june['ls_s2_percent_diff'], aug_sep['ls_s2_percent_diff'])

print(t_stat)
print(p_val)

# %% Box plot to compare lake areas across rois with sr and toa data

rois = [r for r in area_data_clean['roi'].unique() if r != 'YKD_sub1']

fig, ax = plt.subplots(figsize=(10, 6))

# We'll place each ROI's two boxplots (SR vs TOA) side by side
positions = []
labels = []
current_pos = 0

for roi in rois:
    sr_vals = area_data_clean[
        (area_data_clean['roi'] == roi) &
        (area_data_clean['level'] == 'sr')
    ]['ls_s2_percent_diff'].dropna()

    toa_vals = area_data_clean[
        (area_data_clean['roi'] == roi) &
        (area_data_clean['level'] == 'toa')
    ]['ls_s2_percent_diff'].dropna()

    sr_vals = pd.to_numeric(sr_vals)
    toa_vals = pd.to_numeric(toa_vals)

    data = [sr_vals, toa_vals]

    # Create paired boxplots at current_pos and current_pos + 1
    box = ax.boxplot(data,
                     positions=[current_pos, current_pos + 1],
                     widths=0.55,
                     patch_artist=True)

    # Color the boxes
    box['boxes'][0].set_facecolor('skyblue')
    box['boxes'][1].set_facecolor('lightgreen')

    # Store x-axis position for label
    midpoint = current_pos + 0.5
    positions.append(midpoint)
    labels.append(roi)

    # Advance position for the next pair
    current_pos += 3

# Set x-axis ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels)

# Add a horizontal line at y=0
ax.axhline(y=0, color='red', linestyle='--')

# Legend
skyblue_patch = mpatches.Patch(color='skyblue', label='SR')
lightgreen_patch = mpatches.Patch(color='lightgreen', label='TOA')
ax.legend(handles=[skyblue_patch, lightgreen_patch], title='Data Level')

ax.set_title('LS8 vs S2 area difference by ROI (SR vs. TOA)')
ax.set_ylabel('Percent Lake Area Difference')
plt.tight_layout()
plt.show()


# %% Monthly boxplot of TOA and SR data


months = sorted(area_data_clean['month'].dropna().unique())
months = [5, 6, 7, 8, 9]


fig, ax = plt.subplots(figsize=(10, 6))

positions = []
labels = []
current_pos = 0

for month in months:
    sr_vals = area_data_clean[
        (area_data_clean['month'] == month) & (area_data_clean['level'] == 'sr')
    ]['ls_s2_percent_diff'].dropna()

    toa_vals = area_data_clean[
        (area_data_clean['month'] == month) & (area_data_clean['level'] == 'toa')
    ]['ls_s2_percent_diff'].dropna()

    sr_vals = pd.to_numeric(sr_vals)
    toa_vals = pd.to_numeric(toa_vals)

    data = [sr_vals, toa_vals]

    box = ax.boxplot(data,
                     positions=[current_pos, current_pos + 1],
                     widths=0.55,
                     patch_artist=True)

    # Color the boxes
    box['boxes'][0].set_facecolor('skyblue')
    box['boxes'][1].set_facecolor('lightgreen')

    # Update label positions
    midpoint = current_pos + 0.5
    positions.append(midpoint)
    labels.append(str(month))  # Convert month to string if needed

    # Increase spacing
    current_pos += 3

# Set x-axis
ax.set_xticks(positions)
ax.set_xticklabels(labels)

# Horizontal line at y=0
ax.axhline(y=0, color='red', linestyle='--')

# Legend
skyblue_patch = mpatches.Patch(color='skyblue', label='SR')
lightgreen_patch = mpatches.Patch(color='lightgreen', label='TOA')
ax.legend(handles=[skyblue_patch, lightgreen_patch], title='Data Level')

ax.set_title('LS8 vs S2 area difference by Month (SR vs. TOA)')
ax.set_ylabel('Percent Lake Area Difference')
ax.set_xlabel('Month of the Year')
plt.tight_layout()
plt.show()


# %%

months = sorted(area_data_clean['month'].dropna().unique())
months = [5, 6, 7, 8, 9]

fig, ax = plt.subplots(figsize=(10, 6))

positions = []
labels = []
current_pos = 0

for month in months:
    toa_vals = area_data_clean[
        (area_data_clean['month'] == month) & (area_data_clean['level'] == 'toa')
    ]['ls_s2_percent_diff'].dropna()

    toa_vals = pd.to_numeric(toa_vals)

    # Plot only TOA data for this month:
    data = [toa_vals]
    box = ax.boxplot(data,
                     positions=[current_pos],
                     widths=0.55,
                     patch_artist=True)

    # Set the box color for TOA
    box['boxes'][0].set_facecolor('lightgreen')

    # Update label positions
    positions.append(current_pos)
    labels.append(str(month))
    
    current_pos += 3

# Set x-axis
ax.set_xticks(positions)
ax.set_xticklabels(labels)

# Horizontal line at y=0
ax.axhline(y=0, color='red', linestyle='--')

# Legend
toa_patch = mpatches.Patch(color='lightgreen', label='TOA')
ax.legend(handles=[toa_patch], title='Data Level')

ax.set_title('LS8 vs S2 area difference by Month (TOA only)')
ax.set_ylabel('Percent Lake Area Difference')
ax.set_xlabel('Month of the Year')
plt.tight_layout()
plt.show()