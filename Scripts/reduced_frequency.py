import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# --- Global style ---
mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})

# Create a dictionary with the data from the table
data = {
    'Robot': ['USTButterfly-II', 'Zhang et al.', 'USTButterfly', 'AirPulse'],
    'Year': [2022, 2023, 2024, 2025],
    'Weight_g': [54, 39.6, 107.1, 26],
    'Wingspan_cm': [50, 62, 90, 62],
    'Flapping_Frequency_Hz': [4.4, 2.25, 3.25, 3.25],  
    'Flapping_Amplitude_deg': [80, 105, 50, 90],  
    'Wing_Surface_mm2': [71052.6, 113380, 123184.3, 54916.83],  
    'Flight_Speed_ms': [2.35, 1.5, 0.53, 0.8],
    'Color': ['#CCB974', '#55A868', '#C44E52', '#8172B2'],
    'Marker': ['s', 'o', 'D', '*'],
    'MarkerSize': [9.5, 10.5, 9.12, 13]  
}
# Convert to DataFrame
df = pd.DataFrame(data)

# Constants
kinematic_viscosity = 1.79e-5  # m^2/s

def calculate_velocity_ranges(df):
    results = []
    
    for idx, row in df.iterrows():
        # Convert units
        wingspan = row['Wingspan_cm'] / 100  # m
        wing_surface = row['Wing_Surface_mm2'] / 1e6  # m^2
        frequency = row['Flapping_Frequency_Hz']  # Hz
        amplitude = np.radians(row['Flapping_Amplitude_deg'])  # radians
        current_speed = row['Flight_Speed_ms']  # m/s
        
        # Calculate mean chord length
        chord_length = wing_surface / wingspan  # m
        
        # Current reduced frequency
        current_reduced_freq = (np.pi * frequency * chord_length) / current_speed if current_speed > 0 else np.inf
        
        results.append({
            'Robot': row['Robot'],
            'Color': row['Color'],
            'Marker': row['Marker'],
            'MarkerSize': row['MarkerSize'],
            'Chord_Length_m': chord_length,
            'Flapping_Frequency_Hz': frequency,
            'Current_Speed_ms': current_speed,
            'Current_Reduced_Frequency': current_reduced_freq
        })
    
    return pd.DataFrame(results)

# Calculate velocity ranges
velocity_df = calculate_velocity_ranges(df)

# Reorder the dataframe according to specified order
robot_order = ["AirPulse", "Zhang et al.", "USTButterfly", "USTButterfly-II"]
velocity_df = velocity_df.set_index('Robot').loc[robot_order].reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Create gradient background for aerodynamic regimes
velocities = np.linspace(0.1, 5, 300)  # Velocity range from 0.1 to 5 m/s
reduced_freq = np.logspace(-2, 2, 300)  # Reduced frequency range

V, K = np.meshgrid(velocities, reduced_freq)

# Create custom colormap for regimes
colors = ['#2E8B57', '#FFD700', '#DC1433']  # Green, Yellow, Red
cmap = LinearSegmentedColormap.from_list('regime_cmap', colors, N=256)

# Create regime indicator (log scale for smooth transition)
regime_indicator = np.zeros_like(V)
for i in range(len(reduced_freq)):
    for j in range(len(velocities)):
        k_val = K[i,j]
        if k_val < 0.1:
            regime_indicator[i,j] = 0.0  # Quasi-steady
        elif k_val > 1.0:
            regime_indicator[i,j] = 1.0  # Unsteady
        else:
            # Smooth transition between 0.1 and 1.0
            regime_indicator[i,j] = (np.log10(k_val) - np.log10(0.1)) / (np.log10(1.0) - np.log10(0.1))

# Plot gradient background
im = ax.contourf(V, K, regime_indicator, levels=50, alpha=0.3, cmap=cmap, zorder=1)
ax.set_yscale('log')

# Plot reduced frequency curves for each robot in specified order
line_styles = ['-', '--', '-.', ':']

for i, robot in enumerate(velocity_df['Robot']):
    row = velocity_df.iloc[i]
    chord = row['Chord_Length_m']
    freq = row['Flapping_Frequency_Hz']
    color = row['Color']
    marker = row['Marker']
    
    # Calculate reduced frequency over velocity range
    k_values = (np.pi * freq * chord) / velocities
    
    # Plot the curve
    ax.plot(velocities, k_values, color=color, linewidth=2.5, 
            linestyle=line_styles[i], label=robot, zorder=3)

# Add regime boundaries
ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
ax.axhline(y=0.1, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

# Customize the plot
ax.set_xlabel('Forward Velocity (m/s)', fontsize=16)
ax.set_ylabel('Reduced Frequency, k', fontsize=16)

ax.set_xlim(0, 5)
ax.set_ylim(0.01, 100)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Create robot legend
from matplotlib.lines import Line2D

# Create legend elements for robots only
legend_elements = []
for i, robot in enumerate(velocity_df['Robot']):
    row = velocity_df.iloc[i]
    legend_elements.append(
        Line2D([0], [0], color=row['Color'], 
               linestyle=line_styles[i], linewidth=2.5, 
               label=robot, 
            )
    )

# Add robot legend to the plot
robot_legend = ax.legend(handles=legend_elements, loc='upper right', 
                         framealpha=0.95, fontsize=12, 
                         handlelength=3, handletextpad=1.5)

# Add colorbar for the gradient background
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Quasi-Steady\n(k < 0.1)', 'Transition\n(0.1 ≤ k ≤ 1)', 'Unsteady\n(k > 1)'])
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("fig_reduced_frequency.tiff", dpi=600, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("Detailed Aerodynamic Regime Analysis:")
print("=" * 50)
for idx, row in velocity_df.iterrows():
    current_k = row['Current_Reduced_Frequency']
    if current_k > 1:
        regime = "UNSTEADY"
    elif current_k < 0.1:
        regime = "QUASI-STEADY"
    else:
        regime = "TRANSITION"
    
    print(f"\n{row['Robot']}:")
    print(f"  Current Speed: {row['Current_Speed_ms']:.2f} m/s")
    print(f"  Reduced Frequency: {current_k:.3f}")
    print(f"  Aerodynamic Regime: {regime}")
    print(f"  Chord Length: {row['Chord_Length_m']*1000:.2f} mm")
    print(f"  Flapping Frequency: {row['Flapping_Frequency_Hz']:.2f} Hz")