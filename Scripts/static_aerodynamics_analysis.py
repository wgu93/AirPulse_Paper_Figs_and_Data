import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 15,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})

# Custom color palette
custom_colors = [
    '#26598c', '#387394', '#4d8c8f', '#669e8a',
    '#8cb385', '#bfd18f', '#f2e699', '#f5c785', '#f5a680',
    '#ed8087', '#e06194', '#d14794', '#c2388c',
]

# === Load results ===
results_df = pd.read_csv("data/aerodynamic_coefficients_summary.csv")

if not results_df.empty:
    # Extract and sort data
    aoas = results_df["aoa_deg"].values
    lift = results_df["L"].values
    drag = results_df["D"].values
    l_d  = results_df["L_D"].values
    pitch_moment = -results_df["M"].values

    sorted_idx = np.argsort(aoas)
    aoas_sorted = aoas[sorted_idx]
    lift_sorted = lift[sorted_idx]
    drag_sorted = -drag[sorted_idx]     # reverse sign
    l_d_sorted  = -l_d[sorted_idx]      # reverse sign
    pitch_sorted = pitch_moment[sorted_idx]

    # Assign colors 
    n_points = len(aoas_sorted)
    colors = [custom_colors[i % len(custom_colors)] for i in range(n_points)]

    # === Create single figure with 4 subplots in a column ===
    fig, axes = plt.subplots(4, 1, figsize=(6, 6.5))
    
    # Plot data on each subplot
    plot_data = [
        (lift_sorted, "Lift (N)", "b", "Lift (N)"),
        (drag_sorted, "Drag (N)", "r", "Drag (N)"),
        (l_d_sorted, "Lift-to-Drag Ratio", "g", "Lift/Drag"),
        (pitch_sorted, "Pitch Moment (Nm)", "m", "Pitch Moment (Nm)")
    ]
    
    legend_elements = []
    
    for i, (ax, (y_data, ylabel, linecolor, label)) in enumerate(zip(axes, plot_data)):
        # Plot connecting line
        line = ax.plot(aoas_sorted, y_data, color=linecolor, linewidth=4, zorder=1, alpha=0.2)
        # Scatter with custom gradient colors
        ax.scatter(aoas_sorted, y_data, c=colors, linewidths=5, s=50, zorder=2)
        # ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        proxy_line = plt.Line2D([0], [0], color=linecolor, linewidth=4, alpha=0.2, label=label)
        legend_elements.append(proxy_line)
        
        # Only add x-label to bottom plot
        if i == 3:
            ax.set_xlabel("Angle of Attack (deg)")
    
    # Add shared legend on top of the figure
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, 
               fancybox=False, shadow=False)
    
    # Save
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    fig.savefig("fig_aerodyn_all_plots.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    plt.show()