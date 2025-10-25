import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

# ----------------- User settings -----------------
files = [
    "data/aerodynamic_forces_results_4_veins_complete_new.csv",
    "data/aerodynamic_forces_results_2_veins_complete_new.csv", 
    "data/aerodynamic_forces_results_4_veins_forewing_only_new.csv"
]

legend_names = {
    "data/aerodynamic_forces_results_4_veins_complete_new.csv": "Intact (4 veins)",
    "data/aerodynamic_forces_results_2_veins_complete_new.csv": "Intact (2 veins)",
    "data/aerodynamic_forces_results_4_veins_forewing_only_new.csv": "De-winged (4 veins)"
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
line_styles = ['-', '--', '-.']
line_widths = [2.5, 2.5, 2.5]
fontsize = 18

plt.style.use('default')
mpl.rcParams.update({
    "font.size": 15,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})
# -------------------------------------------------

# Read files 
dfs = {}
for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            required = {'normalized_time', 'Fx', 'Fy', 'Fz'}
            if not required.issubset(df.columns):
                raise ValueError(f"Missing required columns in {f}. Found columns: {df.columns.tolist()}")
            df = df.sort_values('normalized_time').reset_index(drop=True)
            dfs[f] = df
        except Exception as e:
            print(f"Error reading {f}: {e}. Skipping this file.")
    else:
        print(f"Warning: File {f} not found. Skipping...")

if not dfs:
    raise SystemExit("No valid CSV files found. Please check file paths.")

# Compute per-component min/max across data 
components = ['Fx', 'Fy', 'Fz']
comp_minmax = {}
for comp in components:
    all_vals = np.hstack([df[comp].values for df in dfs.values()])
    absmax = max(abs(np.min(all_vals)), abs(np.max(all_vals)))
    vmin, vmax = -absmax, absmax
    comp_minmax[comp] = (vmin, vmax)

# Compute an offset (R0) per component so that R = R0 + r is always > 0
R0_map = {}
for comp, (vmin, vmax) in comp_minmax.items():
    rng = vmax - vmin
    margin = 0.05 * max(rng, 1.0) 
    R0 = -vmin + margin
    R0_map[comp] = R0

# Prepare figure and polar axes for 1x3 layout
fig = plt.figure(figsize=(15, 5))  
axes = [fig.add_subplot(1, 3, 1, projection='polar')]  
axes.append(fig.add_subplot(1, 3, 2, projection='polar'))  
force_titles = ['(a) Time-Integrated Axial Force ($F_x$)', '(b) Time-Integrated Vertical Force ($F_z$)']

# Plot each experiment using shifted radius R = R0 + F
for idx, (fname, df) in enumerate(dfs.items()):
    exp_label = legend_names.get(fname, fname)
    theta = 2 * np.pi * df['normalized_time'].values
    # Plot only Fx and Fz 
    for ax, comp in zip(axes, ['Fx', 'Fz']):
        R0 = R0_map[comp]
        shifted_r = df[comp].values + R0
        ax.plot(theta, shifted_r,
                label=exp_label,
                color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=line_widths[idx % len(line_widths)],
                alpha=0.95)

# Customize polar axes: use shifted limits and show tick labels as signed values
theta_full = np.linspace(0, 2 * np.pi, 800)  # for baseline ring plotting
for ax, comp, title in zip(axes, ['Fx', 'Fz'], force_titles):
    vmin, vmax = comp_minmax[comp]
    R0 = R0_map[comp]

    # Shifted min/max
    shifted_min = vmin + R0
    shifted_max = vmax + R0
    ax.set_ylim(shifted_min, shifted_max)

    ax.set_title(title, fontweight='bold', pad=12, fontsize=16, loc="left", y=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
    ax.set_xticklabels(['(0/4)T', '(1/4)T', '(2/4)T', '(3/4)T'], fontsize=fontsize)
    ax.tick_params(axis='x', pad=12)  # Reduced pad from 12 to 8 to bring labels closer

    # Create radial ticks in shifted coordinates but label them as signed forces (original scale)
    n_ticks = 5
    shifted_ticks = np.linspace(shifted_min, shifted_max, n_ticks)
    ax.set_yticks(shifted_ticks)
    y_labels = [f"{(val - R0):.2f}" for val in shifted_ticks]
    ax.set_yticklabels(y_labels, fontsize=fontsize)

    # Shade upstroke sector (using shifted coords)
    upstroke_start = np.pi
    upstroke_end = 2 * np.pi
    angs = np.linspace(upstroke_start, upstroke_end, 200)
    ax.fill_between(angs, shifted_min, shifted_max, color='lightgray', alpha=0.5, zorder=0)

    # Plot baseline circle for force = 0 (shifted radius = R0)
    baseline_r = np.full_like(theta_full, R0)
    ax.plot(theta_full, baseline_r, color='gray', linewidth=2.0, linestyle='--', zorder=2)

# ---------------- Integrated forces ----------------
results = []
for fname, df in dfs.items():
    time = df['normalized_time'].values
    Fx_signed = np.trapz(df['Fx'].values, time)
    Fy_signed = np.trapz(df['Fy'].values, time)
    Fz_signed = np.trapz(df['Fz'].values, time)
    total_signed = np.sqrt(Fx_signed**2 + Fy_signed**2 + Fz_signed**2)

    Fx_abs = np.trapz(np.abs(df['Fx'].values), time)
    Fy_abs = np.trapz(np.abs(df['Fy'].values), time)
    Fz_abs = np.trapz(np.abs(df['Fz'].values), time)
    total_abs = np.sqrt(Fx_abs**2 + Fy_abs**2 + Fz_abs**2)

    results.append({
        'experiment': legend_names.get(fname, fname),
        'Fx_signed': Fx_signed, 'Fy_signed': Fy_signed, 'Fz_signed': Fz_signed, 'total_signed': total_signed,
        'Fx_abs': Fx_abs, 'Fy_abs': Fy_abs, 'Fz_abs': Fz_abs, 'total_abs': total_abs
    })

results_df = pd.DataFrame(results)

# Third subplot: stacked bar plot of absolute integrals
ax3 = fig.add_subplot(1, 3, 3)
x_pos = np.arange(len(results_df))
width = 0.6
components_abs = ['Fx_abs', 'Fz_abs']  # Only axial and vertical forces
comp_labels = ['Axial', 'Vertical']  

# Use the corresponding colors for each configuration with different transparency
comp_alphas = [0.7, 0.4] 

bottom = np.zeros(len(results_df))
for i, (comp, label, alpha) in enumerate(zip(components_abs, comp_labels, comp_alphas)):
    for j, row in enumerate(results_df.iterrows()):
        config_color = colors[j % len(colors)]
        vals = results_df[comp].values
        ax3.bar(x_pos[j], vals[j], width, bottom=bottom[j], label=label if i == 0 and j == 0 else "", 
                color=config_color, alpha=alpha, edgecolor='black', linewidth=1)
    bottom += results_df[comp].values

ax3.set_ylabel('Average Force (|N|)', fontsize=fontsize)
ax3.set_title('(c) Time-Integrated Magnitudes (absolute)', fontsize=16, fontweight='bold', pad=25)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['experiment'], rotation=30, ha='right', fontsize=fontsize)
ax3.grid(True, axis='y', alpha=0.3, linewidth=0.8)

# Create custom legend for bar chart showing both transparency levels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='Axial'),
    Patch(facecolor='gray', alpha=0.4, label='Vertical')
]
legend = ax3.legend(handles=legend_elements, frameon=True, fontsize=fontsize-1, loc='best')
legend.get_frame().set_alpha(0.5)  # semi-transparent background

# # Add the signed net impulse values under each bar
# for i, row in results_df.iterrows():
#     signed_text = (f"Net signed (Fx,Fy,Fz):\n"
#                    f"{row['Fx_signed']:.3f}, {row['Fy_signed']:.3f}, {row['Fz_signed']:.3f}\n"
#                    f"|Net| = {row['total_signed']:.3f}")
#     y_text = 1 - 0.05 * bottom.max()
#     ax3.text(x_pos[i], y_text, signed_text, ha='center', va='top', fontsize=14, 
#              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.98, wspace=0.4)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.06, -0.08), ncol=len(handles), frameon=True, fontsize=fontsize-1)

out_png = 'fig_wing_ablation.png'
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.show()

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print("\nIntegrated forces (signed and absolute):")
print(results_df.to_string(index=False))
print(f"\nFigure saved to: {out_png}")