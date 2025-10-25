import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib as mpl


def print_info(text): 
    print(f"\033[92m{text}\033[0m")

def plot_marker_trajectories(file_path, marker_prefix, figure_prefix, hinge_marker, second_marker, third_marker, tip_marker, cycle_index='middle'):
    data = pd.read_csv(file_path)
    filtered_data = data[data['makerName'].str.startswith(marker_prefix, na=False)]

    if filtered_data.empty:
        print_info(f'No data found for markers starting with {marker_prefix}')
        return

    # Convert Unix time to seconds relative to start time
    start_time = filtered_data['BroadcastTime'].iloc[0]
    filtered_data['TimeSec'] = (filtered_data['BroadcastTime'] - start_time) / 1e6
    unique_times = np.sort(filtered_data['TimeSec'].unique())

    # TODO Make sure the sequence below is from wing hinge to wing tip
    marker_name_map = {
        hinge_marker: r'$0$',
        second_marker: r'$\frac{1}{3} L_{rod}$',
        third_marker: r'$\frac{2}{3} L_{rod}$',
        tip_marker: r'$L_{rod}$',
    }
    marker_shapes = ['^', 'o', 's', 'd']

    fig = plt.figure(figsize=(18, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[20, 1], width_ratios=[1.5, 1, 1])
    
    # Main plots
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax_down = fig.add_subplot(gs[0, 1])  # Downstroke subplot
    ax_up = fig.add_subplot(gs[0, 2])    # Upstroke subplot
    
    # Colorbar axis
    cax = fig.add_subplot(gs[1, 0])

    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(unique_times.min(), unique_times.max())

    # --- Figure: 3D trajectories of Main Rod and Wind Bending Over Time ---
    # Plot 3D trajectories
    for i, marker_name in enumerate(marker_name_map.keys()):
        marker_data = filtered_data[filtered_data['makerName'] == marker_name]
        if marker_data.empty:
            continue
        x = marker_data['makerX'].values
        y = marker_data['makerY'].values
        z = marker_data['makerZ'].values
        time = marker_data['TimeSec'].values

        norm_time = norm(time)
        colors = cmap(norm_time)
        marker_shape = marker_shapes[i]
        label = marker_name_map.get(marker_name, marker_name)

        ax1.plot(x, y, z, color='lightgray', linewidth=0.8, alpha=0.7)
        ax1.scatter(x, y, z, c=colors, marker=marker_shape, label=label, edgecolor='black', linewidth=0.1)

    ax1.set_title(f'3D Trajectories of markers starting with {marker_prefix}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax1.view_init(elev=24, azim=60, roll=-1)

    # Prepare for wing bending plots
    angle_deg = 25
    angle_rad = np.deg2rad(angle_deg)
    rot_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
    rot_axis /= np.linalg.norm(rot_axis)

    z_axis = np.array([0, 0, 1])
    v = np.cross(rot_axis, z_axis)
    if np.linalg.norm(v) < 1e-8:
        v = np.cross(rot_axis, np.array([1, 0, 0]))
    v /= np.linalg.norm(v)
    u = np.cross(v, rot_axis)
    u /= np.linalg.norm(u)

    # Get tip marker data for stroke detection
    tip_data = filtered_data[filtered_data['makerName'] == tip_marker]
    if tip_data.empty:
        print_info(f"Tip marker {tip_marker} not found")
        return

    # Find peaks (highest points) and troughs (lowest points) in Z coordinate
    z_values = tip_data['makerZ'].values
    peaks, _ = find_peaks(z_values, prominence=0.5)  
    troughs, _ = find_peaks(-z_values, prominence=0.5) 

    # Select the desired cycle based on cycle_index parameter
    if len(peaks) < 2 or len(troughs) < 1:
        print_info("Couldn't detect complete wing stroke cycles")
        return
        
    if cycle_index == 'middle':
        cycle_idx = len(peaks) // 2
    elif isinstance(cycle_index, int):
        cycle_idx = min(cycle_index, len(peaks)-2)
    else:
        cycle_idx = 0  # Default to first cycle if invalid input
        
    start_peak = peaks[cycle_idx]
    next_trough = troughs[troughs > start_peak][0] if any(troughs > start_peak) else troughs[-1]
    next_peak = peaks[peaks > next_trough][0] if any(peaks > next_trough) else peaks[-1]
    
    # Get time values for this cycle
    cycle_start = tip_data['TimeSec'].iloc[start_peak]
    mid_cycle = tip_data['TimeSec'].iloc[next_trough]
    cycle_end = tip_data['TimeSec'].iloc[next_peak]

    print_info(f"Visualizing cycle {cycle_idx+1} of {len(peaks)-1} (time: {cycle_start:.2f}s to {cycle_end:.2f}s)")

    # Initialize variables to store max bending angles and corresponding shapes
    max_downstroke_angle = 0
    max_upstroke_angle = 0
    max_downstroke_coords = None
    max_upstroke_coords = None
    max_downstroke_time = None
    max_upstroke_time = None

    # First pass: find maximum bending angles and store the corresponding shapes
    for t in unique_times:
        if t < cycle_start or t > cycle_end:
            continue
            
        time_slice = filtered_data[filtered_data['TimeSec'] == t]
        coords = []
        for marker_name in marker_name_map.keys():
            marker_row = time_slice[time_slice['makerName'] == marker_name]
            if marker_row.empty:
                continue
            x_m = marker_row['makerX'].values[0]
            y_m = marker_row['makerY'].values[0]
            z_m = marker_row['makerZ'].values[0]
            coords.append([x_m, y_m, z_m])
        coords = np.array(coords)
        if coords.shape[0] < 2:
            continue

        hinge_pos = coords[0, :]
        coords_centered = coords - hinge_pos
        coords_2d = np.zeros((coords_centered.shape[0], 2))
        coords_2d[:, 0] = coords_centered @ u
        coords_2d[:, 1] = coords_centered @ v

        # Calculate bending angle (angle between first and last segment)
        vec1 = coords_2d[1] - coords_2d[0]  # Hinge to first marker
        vec2 = coords_2d[-1] - coords_2d[0]   # Hinge to tip marker
        bending_angle = np.degrees(np.arccos(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        ))
        
        if t <= mid_cycle:  # Downstroke phase
            if bending_angle > max_downstroke_angle:
                max_downstroke_angle = bending_angle
                max_downstroke_coords = coords_2d.copy()
                max_downstroke_time = t
        else:  # Upstroke phase
            if bending_angle > max_upstroke_angle:
                max_upstroke_angle = bending_angle
                max_upstroke_coords = coords_2d.copy()
                max_upstroke_time = t

    # Print maximum bending information
    print_info("=== MAXIMUM BENDING ANGLES ===")
    if max_downstroke_time is not None:
        print_info(f"Downstroke:  Max bending = {max_downstroke_angle:.2f}° at time = {max_downstroke_time:.3f}s")
        print_info(f"             (Cycle time: {max_downstroke_time - cycle_start:.3f}s after cycle start)")
    else:
        print_info("Downstroke:  No maximum bending found")
        
    if max_upstroke_time is not None:
        print_info(f"Upstroke:    Max bending = {max_upstroke_angle:.2f}° at time = {max_upstroke_time:.3f}s")
        print_info(f"             (Cycle time: {max_upstroke_time - cycle_start:.3f}s after cycle start)")
    else:
        print_info("Upstroke:    No maximum bending found")
    
    # Calculate timing relative to cycle
    if max_downstroke_time is not None and max_upstroke_time is not None:
        downstroke_duration = mid_cycle - cycle_start
        upstroke_duration = cycle_end - mid_cycle
        total_cycle_duration = cycle_end - cycle_start
        
        if downstroke_duration > 0:
            downstroke_timing = (max_downstroke_time - cycle_start) / downstroke_duration * 100
            print_info(f"Downstroke timing: {downstroke_timing:.1f}% through downstroke phase")
        
        if upstroke_duration > 0:
            upstroke_timing = (max_upstroke_time - mid_cycle) / upstroke_duration * 100
            print_info(f"Upstroke timing:   {upstroke_timing:.1f}% through upstroke phase")
    
    print_info("==============================")

    # Second pass: plot all shapes with blue color and gradually increasing transparency
    downstroke_times = [t for t in unique_times if cycle_start <= t <= mid_cycle]
    
    if downstroke_times:
        # Normalize time for alpha calculation (early = transparent, late = opaque)
        time_min = min(downstroke_times)
        time_max = max(downstroke_times)
        time_range = time_max - time_min
        
        for t in downstroke_times:
            if t < cycle_start or t > mid_cycle:
                continue
                
            time_slice = filtered_data[filtered_data['TimeSec'] == t]
            coords = []
            for marker_name in marker_name_map.keys():
                marker_row = time_slice[time_slice['makerName'] == marker_name]
                if marker_row.empty:
                    continue
                x_m = marker_row['makerX'].values[0]
                y_m = marker_row['makerY'].values[0]
                z_m = marker_row['makerZ'].values[0]
                coords.append([x_m, y_m, z_m])
            coords = np.array(coords)
            if coords.shape[0] < 2:
                continue

            hinge_pos = coords[0, :]
            coords_centered = coords - hinge_pos
            coords_2d = np.zeros((coords_centered.shape[0], 2))
            coords_2d[:, 0] = coords_centered @ u
            coords_2d[:, 1] = coords_centered @ v

            # Calculate alpha based on time progression (early = transparent, late = opaque)
            if time_range > 0:
                alpha = 0.1 + 0.7 * (t - time_min) / time_range  # Alpha from 0.1 to 0.8
            else:
                alpha = 0.5
            
            # Check if this is the maximum bending shape
            is_max_downstroke = (t == max_downstroke_time)
            
            if is_max_downstroke:
                # Highlight maximum bending in red with thicker line
                ax_down.plot(coords_2d[:, 1], coords_2d[:, 0], color='red', alpha=1.0, linewidth=4.0, zorder=10)
                for i, coord in enumerate(coords_2d):
                    ax_down.scatter(coord[1], coord[0], color='red', edgecolor='black', 
                                  marker=marker_shapes[i], s=60, zorder=11)
            else:
                rgb_color = (67/255,124/255,179/255) # blue: (67/255,124/255,179/255) orange: (239/255,139/255,59/255)
                # Use rgb color with calculated transparency
                ax_down.plot(coords_2d[:, 1], coords_2d[:, 0], color=rgb_color, alpha=alpha, linewidth=2.5, zorder=10) 
                for i, coord in enumerate(coords_2d):
                    ax_down.scatter(coord[1], coord[0], color=rgb_color, edgecolor='black', 
                                  marker=marker_shapes[i], s=30, alpha=alpha, zorder=11)

    # Continue with upstroke plotting
    for t in unique_times:
        if t < mid_cycle or t > cycle_end:
            continue
            
        time_slice = filtered_data[filtered_data['TimeSec'] == t]
        coords = []
        for marker_name in marker_name_map.keys():
            marker_row = time_slice[time_slice['makerName'] == marker_name]
            if marker_row.empty:
                continue
            x_m = marker_row['makerX'].values[0]
            y_m = marker_row['makerY'].values[0]
            z_m = marker_row['makerZ'].values[0]
            coords.append([x_m, y_m, z_m])
        coords = np.array(coords)
        if coords.shape[0] < 2:
            continue

        hinge_pos = coords[0, :]
        coords_centered = coords - hinge_pos
        coords_2d = np.zeros((coords_centered.shape[0], 2))
        coords_2d[:, 0] = coords_centered @ u
        coords_2d[:, 1] = coords_centered @ v

        color = cmap(norm(t))
        
        # Check if this is the maximum bending shape
        is_max_upstroke = (t == max_upstroke_time)
        
        # if is_max_upstroke:
        #     # Highlight maximum bending in red with thicker line
        #     ax_up.plot(coords_2d[:, 1], coords_2d[:, 0], color='red', alpha=1.0, linewidth=4.0, zorder=10)
        #     for i, coord in enumerate(coords_2d):
        #         ax_up.scatter(coord[1], coord[0], color='red', edgecolor='black', 
        #                     marker=marker_shapes[i], s=60, zorder=11,
        #                     label=marker_name_map[list(marker_name_map.keys())[i]] if t == mid_cycle else "")
        # else:
        #     # Regular time-colored plotting
        #     ax_up.plot(coords_2d[:, 1], coords_2d[:, 0], color=color, alpha=0.3, linewidth=1.0)
        #     for i, coord in enumerate(coords_2d):
        #         ax_up.scatter(coord[1], coord[0], color=color, edgecolor='black', 
        #                     marker=marker_shapes[i], s=30, alpha=0.5,
        #                     label=marker_name_map[list(marker_name_map.keys())[i]] if t == mid_cycle else "")

    # Add max bending angle annotations with timing information
    if max_downstroke_time is not None:
        ax_down.annotate(f'Max bending: {max_downstroke_angle:.1f}°', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9),
                        fontsize=13, weight='bold')
    
    # if max_upstroke_time is not None:
    #     ax_up.annotate(f'Max bending: {max_upstroke_angle:.1f}°\nTime: {max_upstroke_time:.3f}s', 
    #                   xy=(0.05, 0.95), xycoords='axes fraction',
    #                   ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9),
    #                   fontsize=10, weight='bold')

    # Configure downstroke plot (no legend)
    ax_down.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax_down.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax_down.set_aspect('equal', 'box')
    ax_down.grid(True, linestyle=':')
    ax_down.set_xlabel(r'Main rod direction (mm)', fontsize=17)
    ax_down.set_ylabel(r'Vertical displacement (mm)', fontsize=17)
    ax_down.set_title(f'Downstroke Wing Bending (Cycle {cycle_idx+1})\nRed = Maximum Bending')
    ax_down.tick_params(axis='x', which='both', labelsize=17)
    ax_down.tick_params(axis='y', which='both', labelsize=17)

    # # Configure upstroke plot (with legend)
    # ax_up.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # ax_up.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    # ax_up.set_aspect('equal', 'box')
    # ax_up.grid(True, linestyle=':')
    # ax_up.set_xlabel(r'Perpendicular direction $\mathbf{v}$')
    # ax_up.set_ylabel(r'Perpendicular direction $\mathbf{u}$')
    # ax_up.set_title(f'Upstroke Wing Bending (Cycle {cycle_idx+1})\nRed = Maximum Bending')

    # Add colorbar below the left subplot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', aspect=50, pad=0.)
    ticks = cbar.get_ticks()
    t_min, t_max = unique_times.min(), unique_times.max()
    norm_ticks = (ticks - t_min) / (t_max - t_min)
    cbar.set_ticklabels([f"{v:.2f}" for v in norm_ticks])
    cbar.set_label('Normalized Time (s)')

    plt.tight_layout()
    fig.savefig(f'fig_wing_bending.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot marker trajectories from motion capture data.')
    parser.add_argument('-c', '--cycle', type=str, default='middle', 
                       help='Cycle index to plot ("middle", or integer index starting from 0)')
    args = parser.parse_args()

    try:
        cycle_index = int(args.cycle)
    except ValueError:
        cycle_index = args.cycle  

    file_path = 'data/membrane_wing_4_veins_mocap_trial2.csv'  # TODO Update path
    prefix = file_path.rsplit('_motion_data', 1)[0]
    plot_marker_trajectories(file_path, 'Rigid_Wing_4', prefix, 
                             hinge_marker = 'Rigid_Wing_4_RMarker20601', 
                             second_marker = 'Rigid_Wing_4_RMarker20605', 
                             third_marker = 'Rigid_Wing_4_RMarker20602', 
                             tip_marker = 'Rigid_Wing_4_RMarker20603',
                             cycle_index=cycle_index) # -c 44
    
    # file_path = 'data/membrane_wing_2_veins_mocap_trial3.csv'  # TODO Update path
    # prefix = file_path.rsplit('_motion_data', 1)[0]
    # plot_marker_trajectories(file_path, 'Rigid_Wing_2', prefix, 
    #                          hinge_marker = 'Rigid_Wing_2_RMarker20700', 
    #                          second_marker = 'Rigid_Wing_2_RMarker20701', 
    #                          third_marker = 'Rigid_Wing_2_RMarker20703', 
    #                          tip_marker = 'Rigid_Wing_2_RMarker20702',
    #                          cycle_index=cycle_index)