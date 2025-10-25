import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def find_forewing_period(forewing_times, forewing_angles):
    # Find peaks (maxima) in the forewing signal
    peaks, _ = find_peaks(forewing_angles, prominence=5, distance=10)
    
    if len(peaks) < 2:
        # If not enough peaks, estimate period from the data range
        period = forewing_times[-1] - forewing_times[0]
        peak_times = [forewing_times[0], forewing_times[-1]]
    else:
        # Calculate period from consecutive peaks
        peak_times = forewing_times[peaks]
        periods = np.diff(peak_times)
        period = np.mean(periods)
    
    return period, peaks, peak_times

def align_data_to_first_maximum(df):
    # Get interpolated forewing data
    interp_data = df[df['data_type'] == 'interpolated']
    forewing_interp = interp_data.dropna(subset=['forewing_angle'])
    
    if forewing_interp.empty:
        print("Warning: No interpolated forewing data found. Using raw data for alignment.")
        # Fall back to raw data
        raw_data = df[df['data_type'] == 'raw']
        forewing_interp = raw_data.dropna(subset=['forewing_angle'])
    
    if forewing_interp.empty:
        print("Error: No forewing data found for alignment.")
        return df, 0, 0
    
    forewing_times = forewing_interp['time'].values
    forewing_angles = forewing_interp['forewing_angle'].values
    
    # Find period and peaks
    period, peaks, peak_times = find_forewing_period(forewing_times, forewing_angles)
    
    if len(peak_times) == 0:
        print("Warning: No peaks found. Using start of data as reference.")
        time_shift = 0
        start_time = forewing_times[0]
    else:
        # Use the first peak as reference
        start_time = peak_times[0]
        time_shift = -start_time
    
    # Apply time shift to all data
    aligned_df = df.copy()
    aligned_df['aligned_time'] = aligned_df['time'] + time_shift
    
    return aligned_df, time_shift, period

def load_and_align_csv_files(csv_files):
    aligned_dfs = []
    periods = []
    labels = []
    
    label_mapping = {
        'wing_angles_custom_time_8.000_9.000': 'Intact (4 veins)',
        'wing_angles_custom_time_11.800_12.500': 'Intact (2 veins)'
    }
    
    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file, comment='#')
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        label = label_mapping.get(filename, filename)
        labels.append(label)
        
        # Align data
        aligned_df, time_shift, period = align_data_to_first_maximum(df)
        aligned_dfs.append(aligned_df)
        periods.append(period)
        
        print(f"  Time shift applied: {time_shift:.3f}s")
        print(f"  Estimated period: {period:.3f}s")
    
    return aligned_dfs, periods, labels

def plot_all_curves_single_plot(csv_files, output_file=None, show_plot=True):
    # Load and align all CSV files
    aligned_dfs, periods, labels = load_and_align_csv_files(csv_files)
    
    if not aligned_dfs:
        print("Error: No valid data to plot.")
        return
    
    # Use the minimum period among all files to ensure we show complete cycles
    min_period = min(periods)
    display_duration = 1.25 * min_period  # Show 1.25 periods
    
    print(f"\nUsing display duration: {display_duration:.3f}s (1.25 periods)")
    
    plt.rcParams.update({
        'font.size': 25,
        'axes.labelsize': 25,
        'axes.titlesize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 19,
        'figure.titlesize': 15,
        'font.family': 'Arial'
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(4.3, 8))
    colors = [(67/255,124/255,179/255), (239/255,139/255,59/255)]  # Blue for 4 veins, Orange for 2 veins
    line_styles = {
        'forewing': '-',
        'hindwing': '--'
    }
    
    line_width = 4
    
    for i, (aligned_df, label, color) in enumerate(zip(aligned_dfs, labels, colors)):
        # Get interpolated data for clean curves
        interp_data = aligned_df[aligned_df['data_type'] == 'interpolated']
        interp_data_display = interp_data[interp_data['aligned_time'] <= display_duration]
        
        # Plot forewing AOA (solid lines)
        forewing_interp = interp_data_display.dropna(subset=['forewing_angle'])
        if not forewing_interp.empty:
            ax.plot(forewing_interp['aligned_time'], forewing_interp['forewing_angle'], 
                    color=color, linewidth=line_width, linestyle=line_styles['forewing'],
                    label=f'{label}')
        
        # Plot hindwing (dashed lines)
        hindwing_interp = interp_data_display.dropna(subset=['hindwing_angle'])
        if not hindwing_interp.empty:
            ax.plot(hindwing_interp['aligned_time'], -hindwing_interp['hindwing_angle'], 
                    color=color, linewidth=line_width, linestyle=line_styles['hindwing'],
                    label=f'{label}')
    
    # Customize the plot
    ax.set_xlabel('Time (s)', fontsize=25)
    ax.set_ylabel('Wing Angle (°)', fontsize=25)
    ax.set_title('Wing Kinematics Comparison: 4 Veins vs 2 Veins', fontsize=15, pad=20)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.2, linewidth=3)
    ax.set_xlim(0, display_duration)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=line_width, linestyle='-', label='Forewing'),
        Line2D([0], [0], color='black', linewidth=line_width, linestyle='--', label='Hindwing')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=19, framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='png')
    
    if show_plot:
        plt.show()
    
    print("\n=== ALIGNMENT SUMMARY ===")
    for label, period in zip(labels, periods):
        print(f"{label}: period = {period:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='Plot all wing angle curves in a single plot')
    parser.add_argument('csv_files', nargs='+', help='Paths to CSV files containing wing angle data')
    parser.add_argument('-o', '--output', help='Output file path for the plot')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    
    args = parser.parse_args()
    
    # Check if files exist
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found")
            return
    
    show_plot = not args.no_show
    plot_all_curves_single_plot(args.csv_files, args.output, show_plot)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        csv_files = [
            "data/wing_angles_custom_time_8.000_9.000.csv",
            "data/wing_angles_custom_time_11.800_12.500.csv"
        ]
        existing_files = [f for f in csv_files if os.path.exists(f)]
        
        if existing_files:
            print("Plotting all curves in single plot for files:")
            for f in existing_files:
                print(f"  - {f}")
            
            plot_all_curves_single_plot(existing_files, output_file="fig_forewing_hindwing_comparison.png")
        else:
            print("Please provide CSV files as arguments or modify the default paths in the script.")
            print("Usage examples:")
            print("  python plot_single_figure.py file1.csv file2.csv")
            print("  python plot_single_figure.py file1.csv file2.csv -o single_plot_figure.png")
    else:
        main()