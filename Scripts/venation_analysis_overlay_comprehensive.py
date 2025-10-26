import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# ======================================================
# 1. Load and merge data
# ======================================================
def load_and_merge_data(line_ratio_csv, angle_csv):
    line_df = pd.read_csv(line_ratio_csv)
    angle_df = pd.read_csv(angle_csv)
    
    print("Line ratio data columns:", line_df.columns.tolist())
    print("Angle data columns:", angle_df.columns.tolist())
    print(f"Line ratio data shape: {line_df.shape}")
    print(f"Angle data shape: {angle_df.shape}")
    
    merged_df = pd.merge(line_df, angle_df, on='filename', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    print(merged_df.head())
    return merged_df

# ======================================================
# 2. Aesthetic settings
# ======================================================
def set_science_style():
    plt.style.use('default')
    sns.set_theme(style='whitegrid')
    mpl.rcParams.update({
        "font.size": 16,
        "axes.linewidth": 1.2,
        "font.family": "Arial",
        "mathtext.fontset": "cm",  
        "xtick.labelsize": 14,  
        "ytick.labelsize": 14,  
    })

# Unified color palette
PALETTE = {
    'Nymphalidae': '#2E86AB',
    'Pieridae': '#A23B72',
    'Papillionidae': '#F18F01'
}

# ======================================================
# 3. Create three-subplot combined figure
# ======================================================
def create_three_panel_figure(merged_df, output_file='fig_venation_analysis_overlay_comprehensive.png'):
    set_science_style()
    fig = plt.figure(figsize=(15, 4))  
    
    # Define subplot grid: 1 row, 3 columns
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)
    
    ax_violin = fig.add_subplot(gs[0])       # Left: violin plot
    ax_hist = fig.add_subplot(gs[1])         # Middle: stacked histogram
    ax_scatter = fig.add_subplot(gs[2])      # Right: scatter + regression
    
    # ------------------ LEFT PANEL: Violin ------------------
    sns.violinplot(
        data=merged_df,
        x='family', y='obtuse_angle',
        palette=PALETTE,
        inner='box',
        linewidth=1.2,
        ax=ax_violin,
        alpha=0.9
    )
    ax_violin.set_xlabel('Butterfly Species', fontsize=15)
    ax_violin.set_ylabel('Dc–Cu1 Angle (deg)', fontsize=15)
    ax_violin.grid(linestyle="--", linewidth=0.6, alpha=0.5)

    # ------------------ MIDDLE PANEL: Stacked Histogram ------------------
    # Create stacked histogram showing family contributions
    family_data = [merged_df[merged_df['family'] == fam]['obtuse_angle'] for fam in PALETTE.keys()]
    
    ax_hist.hist(
        family_data,
        bins=8,
        stacked=True,
        color=[PALETTE[fam] for fam in PALETTE.keys()],
        edgecolor='black',
        alpha=0.7,
    )

    # Get plot limits for optimal text placement
    y_max = ax_hist.get_ylim()[1]
    x_min, x_max = ax_hist.get_xlim()
    x_center = (x_min + x_max) / 2

    # Family-specific means (using the same color palette)
    for i, family in enumerate(PALETTE.keys()):
        family_mean = merged_df[merged_df['family'] == family]['obtuse_angle'].mean()
        ax_hist.axvline(family_mean, color=PALETTE[family], linestyle='--', linewidth=2)
        
        # Determine optimal position for family mean text
        if family_mean > x_center:
            # If family mean is on the right side, place text to the left
            ha_pos = 'right'
            text_x = family_mean - (x_max - x_min) * 0.02
        else:
            # If family mean is on the left side, place text to the right
            ha_pos = 'left'
            text_x = family_mean + (x_max - x_min) * 0.02
        
        # Stagger the vertical positions to avoid overlap
        y_pos = y_max * (0.95 - i * 0.12)
        
        # ax_hist.text(text_x, y_pos, f'{family_mean:.1f}°', #f'{family}: {family_mean:.1f}°',
        #             ha=ha_pos, va='top', fontsize=14, color=PALETTE[family], fontweight='bold',
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=PALETTE[family]))

    ax_hist.set_xlabel('Dc–Cu1 Angle (deg)', fontsize=15)
    ax_hist.set_ylabel('Count', fontsize=15)
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Force integer y-ticks only
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add statistics box
    # ax_hist.text(1.05, 0.97,
    #             f'Total samples: {len(merged_df)}\n'
    #             f'SD: {merged_df["obtuse_angle"].std():.1f}°\n'
    #             f'Range: {merged_df["obtuse_angle"].min():.1f}°–{merged_df["obtuse_angle"].max():.1f}°',
    #             transform=ax_hist.transAxes, va='top',
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #             fontsize=10)

    # ------------------ RIGHT PANEL: Scatter + Regression ------------------
    regression_results = {}

    for fam, color in PALETTE.items():
        data = merged_df.loc[merged_df['family'] == fam]
        if len(data) < 2:
            continue
        
        x = data['obtuse_angle']
        y = data['ratio_blue_over_red']
        ax_scatter.scatter(x, y, s=80, alpha=0.7, color=color,
                           edgecolors='white', linewidth=1, label=fam)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        regression_results[fam] = dict(
            slope=slope, intercept=intercept, r2=r_value**2,
            p=p_value, stderr=std_err
        )
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color=color, linestyle='--', linewidth=2)

    # All-sample regression
    all_x = merged_df['obtuse_angle']
    all_y = merged_df['ratio_blue_over_red']
    if len(all_x) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
        regression_results['All'] = dict(
            slope=slope, intercept=intercept, r2=r_value**2,
            p=p_value, stderr=std_err
        )
        x_line_all = np.linspace(all_x.min(), all_x.max(), 100)
        y_line_all = slope * x_line_all + intercept
        # ax_scatter.plot(x_line_all, y_line_all, color='black', linewidth=2.5,
        #                 alpha=0.9, label='All Samples')

    ax_scatter.set_xlabel('Dc–Cu1 Angle (deg)', fontsize=15)
    ax_scatter.set_ylabel('1/AR', fontsize=15)
    # ax_scatter.legend(frameon=True, loc='best', fontsize=14)
    ax_scatter.grid(linestyle='--', linewidth=0.6, alpha=0.5)
    
    # ------------------ Layout & save ------------------
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    print(f"\n✅ Three-panel figure saved as: {output_file}")
    
    # Print regression summary
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("="*60)
    for fam, res in regression_results.items():
        print(f"{fam}: slope={res['slope']:.4f}, R²={res['r2']:.4f}, p={res['p']:.4f}")
    print("="*60)
    
    return fig, regression_results

# ======================================================
# 4. Statistical summary
# ======================================================
def create_statistical_summary(merged_df, output_file='statistical_summary.csv'):
    summary_stats = merged_df.groupby('family').agg({
        'obtuse_angle': ['count', 'mean', 'std', 'min', 'max'],
        'ratio_blue_over_red': ['mean', 'std', 'min', 'max']
    }).round(4)
    summary_stats.to_csv(output_file)
    print(f"📊 Statistical summary saved as: {output_file}")
    return summary_stats

# ======================================================
# 5. Main
# ======================================================
def main():
    line_ratio_csv = "data/aspect_ratios.csv"
    angle_csv = "data/butterfly_vein_angles_dataaug.csv"
    
    for f in [line_ratio_csv, angle_csv]:
        if not Path(f).exists():
            print(f"File not found: {f}")
            return

    merged_df = load_and_merge_data(line_ratio_csv, angle_csv)
    if merged_df.empty:
        print("Empty merged dataframe.")
        return
    
    print(f"\nNumber of data points: {len(merged_df)}")
    print("Species distribution:")
    print(merged_df['family'].value_counts())

    fig, regression_results = create_three_panel_figure(merged_df)
    create_statistical_summary(merged_df)
    plt.show()

if __name__ == "__main__":
    main()