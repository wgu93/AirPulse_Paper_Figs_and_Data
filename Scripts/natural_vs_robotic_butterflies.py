import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.cm import get_cmap

# --- Global style ---
mpl.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})

fontsize = 15

# --- Load data ---
df = pd.read_csv('data/Nan2018_Table1_NaturalFlyerWingParams.csv').dropna(subset=['Wing Loading', 'AR', 'Name'])
robotic_names = ['AirPulse', 'Zhang et al.', 'USTButterfly', 'USTButterfly-II', 'eMotionButterfly']

# --- Name abbreviation function ---
def abbreviate_name(full_name):
    """
    Abbreviate scientific names by shortening genus to first letter.
    Example: 'Battus polydamas' -> 'B. polydamas'
    """
    if pd.isna(full_name):
        return full_name
    
    # Handle robotic names (keep as is)
    if full_name in robotic_names:
        return full_name
    
    # Handle natural butterfly names
    parts = str(full_name).split()
    if len(parts) >= 2:
        # Take first letter of first word, keep the rest
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    else:
        # If only one word, return as is
        return full_name

# Apply name abbreviation
df['Name_Abbr'] = df['Name'].apply(abbreviate_name)

# Split data
natural_df = df[~df['Name'].isin(robotic_names)].copy()
robotic_df = df[df['Name'].isin(robotic_names)]

# Get abbreviated names for natural butterflies
natural_names_abbr = natural_df['Name_Abbr'].unique()

# Assign unique colors to each natural butterfly 
cmap = get_cmap('tab20')
natural_colors = {name: cmap(i % 20) for i, name in enumerate(natural_df['Name'].unique())}

# Create mapping from abbreviated name to original name for color lookup
name_to_abbr = dict(zip(natural_df['Name'], natural_df['Name_Abbr']))
abbr_to_original = {v: k for k, v in name_to_abbr.items()}

# --- Plotting ---
fig, ax = plt.subplots(figsize=(6, 2.5))

# Plot each natural butterfly as individual scatter with abbreviated legend entry
for name in natural_df['Name'].unique():
    row = natural_df[natural_df['Name'] == name]
    abbr_name = name_to_abbr[name]
    ax.scatter(
        row['Wing Loading'], row['AR'],
        s=40, color=natural_colors[name], alpha=0.8,
        edgecolor='white', linewidth=0.5,
        label=abbr_name, marker='o', zorder=2
    )

# Robotic markers
robotic_markers = ['*', 'o', 'D', 's', 'X']
robotic_colors = ['#8172B2', '#55A868', '#C44E52', '#CCB974', '#4C72B0']
robotic_marker_sizes = [120, 75, 58, 68, 75]

for idx, name in enumerate(robotic_names):
    row = robotic_df[robotic_df['Name'] == name]
    ax.scatter(
        row['Wing Loading'], row['AR'],
        s=robotic_marker_sizes[idx], color=robotic_colors[idx], alpha=0.95,
        edgecolor='black', linewidth=0.8,
        marker=robotic_markers[idx], label=name, zorder=3
    )

# --- Axis & Title ---
ax.set_xlabel('Wing Loading (N/m²)', fontsize=fontsize, labelpad=6)
ax.set_ylabel('Aspect Ratio', fontsize=fontsize, labelpad=6)
ax.set_aspect('equal', adjustable='box') 

# --- Legend below the plot with 3 columns ---
legend = ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.15),
    frameon=True,
    fontsize=10,
    title='Butterfly Species',
    title_fontsize=11,
    ncol=3,  # 3 columns
    handlelength=1.2,
    columnspacing=1.0,  # Column spacing for 3 columns
    borderaxespad=0.5
)

# --- Layout ---
plt.tight_layout()
plt.savefig("fig_natural_vs_robotic_butterflies.tiff", dpi=600, bbox_inches='tight')
plt.show()