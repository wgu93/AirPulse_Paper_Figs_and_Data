import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

scirob_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

def set_scirob_style(fontsize=10, dpi=600):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': fontsize,
        'figure.dpi': dpi,
        'axes.prop_cycle': cycler(color=scirob_colors),
        'savefig.format': 'tiff',
        'savefig.dpi': dpi,
        'figure.constrained_layout.use': True,
    })
    sns.set_context("notebook", font_scale=1)
    sns.set_style("whitegrid")
set_scirob_style(fontsize=9, dpi=600)  

def get_scirob_colors(n=None):
    if n is None:
        return scirob_colors
    return scirob_colors[:n]

# Components and corresponding masses (g)
components = [
    "3D-printed fuselage",
    "Micro servos",
    "Flapping wings",
    "Flight control board",
    "ELRS receiver",
    "1S LiPo battery"
]
masses = [1.01, 9.2, 10.03, 3.3, 0.46, 1.7]
total_mass = sum(masses)
percentages = [m / total_mass * 100 for m in masses]

# Sort by mass (descending) for better visualization
sorted_indices = np.argsort(masses)[::-1]
components = [components[i] for i in sorted_indices]
masses = [masses[i] for i in sorted_indices]
percentages = [percentages[i] for i in sorted_indices]

# Create compact labels
labels = [f"{comp} ({mass:.2f}g, {p:.1f}%)" for comp, mass, p in zip(components, masses, percentages)]

# Use consistent color palette
colors = get_scirob_colors(len(components))

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 4))  

# Create pie chart
wedges, texts = ax.pie(
    masses,
    radius=0.8,
    labels=None,
    colors=colors,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=0.5)
)

# Make it donut chart
centre_circle = plt.Circle((0, 0), 0.5, fc='white')
fig.gca().add_artist(centre_circle)

# Add total mass in the center
ax.text(0, 0, f"Total\n{total_mass:.2f}g", 
        ha='center', va='center', fontsize=10, fontweight='bold')

ax.axis('equal')
ax.set_title("Mass Distribution of Flapping-Wing Robot", fontsize=11, fontweight='bold', pad=15)

legend = ax.legend(
    wedges,
    labels,
    title="Components",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=9,
    title_fontsize=10,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    borderpad=0.8 
)
plt.tight_layout()

# Save with high resolution
plt.savefig("fig_mass_distribution.tiff", dpi=600, bbox_inches='tight')
plt.show()