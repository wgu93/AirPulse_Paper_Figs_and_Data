import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from matplotlib.colors import ListedColormap
sns.set_style("whitegrid")

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,  
    'axes.labelsize': 12, 
    'axes.titlesize': 14,  
    'xtick.labelsize': 11,  
    'ytick.labelsize': 11,  
    'legend.fontsize': 11,  
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'figure.autolayout': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,  
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'axes.edgecolor': '0.3',
    'grid.color': '0.8',  
})

# Load data
df = pd.read_csv('data/20250904_cog_moi_variation_sw_data.csv')

# --- Filter data for flapping angles from -70 to 70 with 10° increments ---
valid_angles = np.arange(-70, 71, 10)
df = df[df['FlapAngle'].isin(valid_angles)]

# --- Find CG values at FlapAngle = 0 to use as origin ---
cg_x_origin = df[df['FlapAngle'] == 0]['CG_X'].values[0]
cg_z_origin = df[df['FlapAngle'] == 0]['CG_Z'].values[0]

# --- Convert inertia from g/mm² to kg/m² ---
inertia_columns = ['Lxx', 'Lyy', 'Lzz', 'Lxy', 'Lxz', 'Lyz']
for col in inertia_columns:
    df[col] = df[col] * 1e-9  # Convert g/mm² to kg/m²

# --- Custom color palette ---
custom_colors = [
    '#26487a', '#26598c', '#387394', '#4d8c8f', '#669e8a',
    '#8cb385', '#bfd18f', '#f2e699', '#f5c785', '#f5a680',
    '#ed8087', '#e06194', '#d14794', '#c2388c', '#b32e80'
]
custom_cmap = ListedColormap(custom_colors)
norm = plt.Normalize(vmin=-70, vmax=70)

# --- Create 1x4 layout ---
fig, axes = plt.subplots(1, 4, figsize=(12, 3))  
fig.subplots_adjust(bottom=0.35, top=0.9, left=0.06, right=0.98, wspace=0.4) 

# Calculate deviations from the origin
cg_x_deviation = df['CG_X'] - cg_x_origin
cg_z_deviation = df['CG_Z'] - cg_z_origin

# Subplot 1: CG_Y vs. CG_X Deviation
sc1 = axes[0].scatter(df['CG_Y'], cg_x_deviation, c=df['FlapAngle'], cmap=custom_cmap, norm=norm,
                     alpha=0.9, edgecolor='white', linewidth=0.5, s=70)  
axes[0].set_xlim(-1, 1)
axes[0].set_xlabel('CG$_Y$ Deviation (mm)', labelpad=6)
axes[0].set_ylabel('CG$_X$ Deviation (mm)', labelpad=6)
# axes[0].set_title('Longitudinal CG Variation', pad=14, fontweight='semibold')
axes[0].grid(True, linestyle=':', alpha=0.4, color='0.7')  
axes[0].axhline(y=0, color='0.5', linestyle='-', alpha=0.8, linewidth=0.8)
axes[0].set_aspect('equal')

# Subplot 2: CG_X Deviation vs. CG_Z Deviation
sc2 = axes[1].scatter(cg_x_deviation, cg_z_deviation, c=df['FlapAngle'], cmap=custom_cmap, norm=norm,
                     alpha=0.9, edgecolor='white', linewidth=0.5, s=70)  
axes[1].set_xlabel('CG$_X$ Deviation (mm)', labelpad=6)
axes[1].set_ylabel('CG$_Z$ Deviation (mm)', labelpad=6)
# axes[1].set_title('Vertical CG Variation', pad=14, fontweight='semibold')
axes[1].grid(True, linestyle=':', alpha=0.4, color='0.7')  
axes[1].axhline(y=0, color='0.5', linestyle='-', alpha=0.8, linewidth=0.8)
axes[1].axvline(x=0, color='0.5', linestyle='-', alpha=0.8, linewidth=0.8)
axes[1].set_aspect('equal')

# Subplot 3: FlapAngle vs. Ixx/Iyy/Izz
markers = ['o', 's', '^']
labels = ['$I_{xx}$', '$I_{yy}$', '$I_{zz}$']

for i, col in enumerate(['Lxx', 'Lyy', 'Lzz']):
    axes[2].scatter(df['FlapAngle'], df[col], c=df['FlapAngle'], cmap=custom_cmap, norm=norm, 
                   marker=markers[i], label=labels[i], alpha=0.9, edgecolor='white', linewidth=0.5, s=70)  

axes[2].set_xlabel('Flapping Angle (deg)', labelpad=6)
axes[2].set_ylabel('Moment of Inertia (kg$\cdot$m$^2$)', labelpad=6)
# axes[2].set_title('Principal Moments of Inertia', pad=14, fontweight='semibold')
axes[2].legend(frameon=True, framealpha=0.95, loc='lower right')
axes[2].grid(True, linestyle=':', alpha=0.4, color='0.7')  # More visible grid
# Format y-axis to use scientific notation with 1e-4
axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

# Subplot 4: FlapAngle vs. Ixy/Ixz/Iyz
markers = ['P', 'D', 'X']
labels = ['$I_{xy}$', '$I_{xz}$', '$I_{yz}$']

for i, col in enumerate(['Lxy', 'Lxz', 'Lyz']):
    axes[3].scatter(df['FlapAngle'], df[col], c=df['FlapAngle'], cmap=custom_cmap, norm=norm,
                   marker=markers[i], label=labels[i], alpha=0.9, edgecolor='white', linewidth=0.5, s=70) 

axes[3].set_xlabel('Flapping Angle (deg)', labelpad=6)
axes[3].set_ylabel('Product of Inertia (kg$\cdot$m$^2$)', labelpad=6)
# axes[3].set_title('Products of Inertia', pad=14, fontweight='semibold')
axes[3].legend(frameon=True, framealpha=0.95, loc='lower right')
axes[3].grid(True, linestyle=':', alpha=0.4, color='0.7')  # More visible grid
# Format y-axis to use scientific notation with 1e-4
axes[3].ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

from matplotlib.colors import BoundaryNorm, ListedColormap 

# Create a custom colormap
custom_cmap = ListedColormap(custom_colors)

# Define the tick labels 
tick_labels = np.arange(-70, 71, 10)  # [-70, -60, -50, ..., 60, 70]

# Create boundaries that match 15 colors
# We need 16 boundaries for 15 colors, evenly spaced from -70 to 70
boundaries = np.linspace(-70, 70, len(custom_colors) + 1)
norm = BoundaryNorm(boundaries, len(custom_colors))

# Calculate tick positions at the center of each color segment
tick_positions = (boundaries[:-1] + boundaries[1:]) / 2

# Add common colorbar at the bottom
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])

# Create the colorbar with explicit boundaries
cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal', 
                   boundaries=boundaries, ticks=tick_positions)

# Set the tick labels (15 labels for 15 color segments)
cbar.set_ticklabels([f'{label:.0f}' for label in tick_labels])
cbar.set_label('Flapping Angle (deg)', fontsize=12, labelpad=10, loc='center')
cbar.outline.set_visible(False)

# Save figure 
plt.savefig('fig_cog_moi_analysis.tiff', format='tiff', bbox_inches='tight', dpi=600)

plt.show()