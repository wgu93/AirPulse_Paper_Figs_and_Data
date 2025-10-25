import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

###
# ==== Control handles for plots ====
# Figure 1: Euler angles vs. PWM signals
# Figure 2: Yaw PID tracking
# Figure 3: PWM vs. Power
# Figure 4: Altitude
# Figure 5: Linear Acceleration Analysis
# Figure 6: Acceleration Magnitude and Statistics
# Figure 7: Acceleration vs Flapping Cycle Phase with Violin Plots
# Figure 8: Pitch Control Performance with Command Signal
###
PLOT_EULER_VS_PWM = True # Figure 1
PLOT_YAW_PID = True # Figure 2
PLOT_PWM_VS_POWER = True # Figure 3
PLOT_ALT = True # Figure 4
PLOT_LINEAR_ACC = True # Figure 5, 6, 7
PLOT_PITCH_CONTROL = True # Figure 8
SAVE_FIG = True

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

###
# ==== Step 1: Load CSV ====
###
# esp_file_path = "0921_Nankou_PitchPID/A_offset_0.6_0.45_0.05_trial4.csv"
# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_strokvel_0.17_0_0.007_trial2_trajectoryCircle.csv"
# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_offset_0.15_0.0_0.0_trial2.csv"
# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_strokvel_0.17_0_0.007_trial1_trajectoryS.csv"
# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_strokvel_0.5_0_0_trial1_unstable.csv"

# PAPER FIGURES
esp_file_path = "0921_Nankou_PitchPID/A_offset_0.6_0.55_0.05_trial2.csv"
# esp_file_path = "0921_Nankou_PitchPID/A_strokvel_0.6_0.7_0.07_trial2.csv"

# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_strokvel_0.17_0_0.007_trial3.csv"
# esp_file_path = "0927_Nankou_YawPID/RenamedData/A_offset_0.15_0.0_0.0_trial5.csv"
esp_df = pd.read_csv(esp_file_path, sep=',')  

###
# ==== Step 2: Convert "_x" columns from ms to seconds ====
###
for col in esp_df.columns:
    if col.endswith("_x"):
        esp_df[col] = esp_df[col] / 1000.0  # ms → sec

###
# ==== Step 3: Remove first row ====
###
esp_df = esp_df.iloc[1:].reset_index(drop=True)

###
# ==== Step 4: Ask user for x-axis range ====
###
x_min_str = input("Enter x-axis min time in seconds (space or Enter to skip): ").strip()
x_max_str = input("Enter x-axis max time in seconds (space or Enter to skip): ").strip()
x_min = float(x_min_str) if x_min_str and x_min_str != " " else None
x_max = float(x_max_str) if x_max_str and x_max_str != " " else None

###
# ==== Step 5: Extract data ====
###

# Step 5-1: Euler angles
# Note 1: roll and pitch measurements are opposite due to the mounting direction of the flight control unit.
# Note 2: recursive least squares algorithm is only applied to roll and pitch measurements, but not to yaw.
# Note 3: the field of target roll in csv file is temporarily reused for desired roll or yaw angle.
# Roll (deg)
time_roll = esp_df["LRS_roll_x"] - esp_df["LRS_roll_x"].iloc[0]
roll_imu = esp_df["pitch_6_y"] # TODO: this is a workaround only; fix this later
roll_rls = -esp_df["LRS_roll_y"] # TODO: this is a workaround only; fix this later
roll_target = esp_df["Target_roll_y"]

# Pitch (deg)
time_pitch = esp_df["LRS_roll_x"] - esp_df["LRS_roll_x"].iloc[0]
pitch_imu = esp_df["roll_6_y"] # TODO: this is a workaround only; fix this later
pitch_rls = esp_df["LRS_pitch_y"]
pitch_target = esp_df["Target_pitch_y"]

# Yaw (deg)
time_yaw = esp_df["yaw_6_x"] - esp_df["yaw_6_x"].iloc[0]
yaw_imu = esp_df["yaw_6_y"]
yaw_target = esp_df["Target_roll_y"] # TODO: this is a workaround only; fix this later

# Step 5-2: control signals
# PWM
time_pwm = esp_df["pwm1_x"] - esp_df["pwm1_x"].iloc[0]
pwm1_y = esp_df["pwm1_y"]
pwm2_y = esp_df["pwm2_y"]

# Currents (mA)
time_servo_current = esp_df["adc_x"] - esp_df["adc_x"].iloc[0] 
left_servo_current = esp_df["adc_y"]     # Servo 1 current  # TODO: confusing naming; fix this later
right_servo_current = esp_df["vol_y"]     # Servo 2 current  # TODO: confusing naming; fix this later

# Angle offset modulation command
time_cmd_offset_pitch = esp_df["Pitch_offset_x"] - esp_df["Pitch_offset_x"].iloc[0]
cmd_offset_pitch = esp_df["Pitch_offset_y"] 
time_cmd_offset_yaw = esp_df["Roll_offset_x"] - esp_df["Roll_offset_x"].iloc[0] # TODO: this is a workaround only; fix this later
cmd_offset_yaw = esp_df["Roll_offset_y"] # TODO: this is a workaround only; fix this later

# Stroke velocity modulation command
time_cmd_strokvel_pitch = esp_df["Pitch_p_x"] - esp_df["Pitch_p_x"].iloc[0]
cmd_strokvel_pitch = 1/esp_df["Pitch_p_y"] # this field now stores 1/p # TODO: this is a workaround only; fix this later
time_cmd_strokvel_yaw = esp_df["Pitch_p_x"] - esp_df["Pitch_p_x"].iloc[0] # TODO: this is a workaround only; fix this later
cmd_strokvel_yaw = 1/esp_df["Pitch_p_y"] # this field now stores 1/p # TODO: this is a workaround only; fix this later

# Step 5-3: other states
# Linear acceleration (in unit g?)
mask = esp_df[['ax_y','ay_y','az_y']].isna().all(axis=1)  # adjust cols # TODO: this is a workaround only; fix this later
esp_df = esp_df[~mask].copy() # TODO: this is a workaround only; fix this later
time_linear_acc = esp_df["ax_x"] - esp_df["ax_x"].iloc[0]
ax_imu = -esp_df["ay_y"] # TODO: this is a workaround only; fix this later
ay_imu = esp_df["ax_y"] # TODO: this is a workaround only; fix this later
az_imu = esp_df["az_y"]

# Altitude (in m)
time_alt = esp_df["alt_x"] - esp_df["alt_x"].iloc[0]
alt_imu = esp_df["alt_y"]

# print(esp_df.tail(10).to_string())

###
# ==== Step 6: Apply time filtering if range provided ====
###
def crop_data(t, *signals):
    """Return cropped time & corresponding signals based on x_min and x_max."""
    mask = pd.Series(True, index=range(len(t)))
    if x_min is not None:
        mask &= (t >= x_min)
    if x_max is not None:
        mask &= (t <= x_max)
    return t[mask], [sig[mask] for sig in signals]

time_roll, (roll_imu, roll_rls, roll_target) = crop_data(time_roll, roll_imu, roll_rls, roll_target)
time_pitch, (pitch_imu, pitch_rls, pitch_target) = crop_data(time_pitch, pitch_imu, pitch_rls, pitch_target)
time_yaw, (yaw_imu, yaw_target) = crop_data(time_yaw, yaw_imu, yaw_target)
time_pwm, (pwm1_y, pwm2_y) = crop_data(time_pwm, pwm1_y, pwm2_y)
time_servo_current, (left_servo_current, right_servo_current) = crop_data(time_servo_current, left_servo_current, right_servo_current)
time_cmd_offset_pitch, (cmd_offset_pitch,) = crop_data(time_cmd_offset_pitch, cmd_offset_pitch)
time_cmd_strokvel_pitch, (cmd_strokvel_pitch,) = crop_data(time_cmd_strokvel_pitch, cmd_strokvel_pitch)
time_cmd_offset_yaw, (cmd_offset_yaw,) = crop_data(time_cmd_offset_yaw, cmd_offset_yaw)
time_cmd_strokvel_yaw, (cmd_strokvel_yaw,) = crop_data(time_cmd_strokvel_yaw, cmd_strokvel_yaw)
time_linear_acc, (ax_imu, ay_imu, az_imu) = crop_data(time_linear_acc, ax_imu, ay_imu, az_imu)
time_alt, (alt_imu,) = crop_data(time_alt, alt_imu)

###
# ==== Step 7: Plot figures ====
###
plt.style.use("seaborn-v0_8-whitegrid")

###
# Figure 1: Euler angles vs. PWM signals
###
if PLOT_EULER_VS_PWM:
    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # --- Top subplot: Roll angle ---
    axs1[0].plot(time_roll, roll_imu, label="Filtered Roll (IMU)", color="#1f77b4", linewidth=2)
    axs1[0].plot(time_roll, roll_rls, label="Estimated Roll Offset (RLS)", color="#ff7f0e", linewidth=2, linestyle="--")
    # axs1[0].plot(time_roll, roll_target, label="Target Roll", color="#2ca02c", linewidth=2, linestyle=":")
    axs1[0].set_ylabel("Roll Angle (deg)")
    axs1[0].legend(loc="best", frameon=True, framealpha=0.9)
    axs1[0].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Second subplot: Pitch angle ---
    axs1[1].plot(time_pitch, pitch_imu, label="Filtered Pitch (IMU)", color="#9467bd", linewidth=2)
    axs1[1].plot(time_pitch, pitch_rls, label="Estimated Pitch Offset (RLS)", color="#d62728", linewidth=2, linestyle="--")
    axs1[1].plot(time_pitch, pitch_target, label="Target Pitch", color="#17becf", linewidth=2, linestyle=":")
    axs1[1].set_ylabel("Pitch Angle (deg)")
    axs1[1].legend(loc="best", frameon=True, framealpha=0.9)
    axs1[1].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Third subplot: Yaw angle ---
    axs1[2].plot(time_yaw, yaw_imu, label="Filtered Yaw (IMU)", color="#7f7f7f", linewidth=2)
    axs1[2].plot(time_yaw, yaw_target, label="Target Yaw", color="#17becf", linewidth=2, linestyle=":")
    axs1[2].set_ylabel("Yaw Angle (deg)")
    axs1[2].legend(loc="best", frameon=True, framealpha=0.9)
    axs1[2].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    # --- Bottom subplot: PWM signals ---
    axs1[3].plot(time_pwm, pwm1_y, label="PWM1", color="#8c564b", linewidth=2)
    axs1[3].plot(time_pwm, pwm2_y, label="PWM2", color="#e377c2", linewidth=2, linestyle="--")
    axs1[3].set_ylabel("PWM")
    axs1[3].set_xlabel("Time (s)")
    axs1[3].legend(loc="upper right", frameon=True, framealpha=0.9)
    axs1[3].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Title ---
    file_name = os.path.basename(esp_file_path)
    fig1.suptitle(f"Data from {file_name}", y=0.98)

    # Improve ticks
    for ax in axs1:
        ax.tick_params(axis="both")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-res first figure
    save_name1 = os.path.splitext(file_name)[0] + "_euler_vs_pwm.png"
    if SAVE_FIG: 
        plt.savefig(save_name1, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name1}")

###
# Figure 2: Yaw PID tracking
###
if PLOT_YAW_PID:
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot yaw angles on left y-axis
    color_pitch = '#1f77b4'
    color_rls = '#ff7f0e'
    color_target = '#2ca02c'

    ax.plot(time_yaw, yaw_imu, label="Filtered Yaw", color=color_pitch, linewidth=2)
    ax.plot(time_yaw, yaw_target, label="Target Yaw", color=color_target, linewidth=2, linestyle=":")
    ax.set_ylabel("Yaw Angle (deg)", color=color_pitch)
    ax.tick_params(axis='y', labelcolor=color_pitch)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Time (s)")

    # Plot command on right y-axis
    ax_cmd = ax.twinx()
    if 'offset' in esp_file_path.lower():
        ax_cmd.plot(time_cmd_offset_yaw, cmd_offset_yaw, label="Yaw Offset Command", color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.set_ylabel("Offset Command (deg)", color='#d62728')
    elif 'strokvel' in esp_file_path.lower():
        ax_cmd.plot(time_cmd_strokvel_yaw, cmd_strokvel_yaw, label="Yaw Stroke Timing Command", color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.set_ylabel("Stroke Timing Command", color='#d62728')
    else:
        # Plot both if neither is clearly identified
        ax_cmd.plot(time_cmd_offset_yaw, cmd_offset_yaw, label="Yaw Offset Command", color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.plot(time_cmd_strokvel_yaw, cmd_strokvel_yaw, label="Yaw Stroke Timing Command", color='#9467bd', linewidth=2, alpha=0.8, linestyle="--")
        ax_cmd.set_ylabel("Command Value", color='#d62728')
    ax_cmd.grid(True, linestyle="-", linewidth=0.8, alpha=0.7)

    ax_cmd.tick_params(axis='y', labelcolor='#d62728')

    # Combine legends
    lines_yaw, labels_yaw = ax.get_legend_handles_labels()
    lines_cmd, labels_cmd = ax_cmd.get_legend_handles_labels()
    # ax.legend(lines_yaw + lines_cmd, labels_yaw + labels_cmd, loc="upper center", 
                # bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.suptitle(f"Yaw Control Performance with Command Signal from {file_name}", 
                y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-res first figure
    save_name2 = os.path.splitext(file_name)[0] + "_yaw_pid_tracking.png"
    if SAVE_FIG: 
        plt.savefig(save_name2, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name2}")

###
# Figure 3: PWM vs. Power
###
if PLOT_PWM_VS_POWER:
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Upper subplot: PWM ---
    axs2[0].plot(time_pwm, pwm1_y, label="Left Servo", color="#8c564b", linewidth=2)
    axs2[0].plot(time_pwm, pwm2_y, label="Right Servo", color="#e377c2", linewidth=2, linestyle="--")
    axs2[0].set_ylabel("PWM")
    axs2[0].legend(loc="upper right", frameon=True, ncol=2, fontsize=15, framealpha=0.9)
    axs2[0].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Lower subplot: Currents (mA) with total power (W) ---
    V_supply = 8.2  # volts

    ax_curr = axs2[1]
    ax_curr.plot(time_servo_current, left_servo_current, label="Left Servo", color="#1f77b4", linewidth=2)
    ax_curr.plot(time_servo_current, right_servo_current, label="Right Servo", color="#ff7f0e", linewidth=2, linestyle="--")
    ax_curr.set_ylabel("Current (mA)")
    ax_curr.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # Secondary y-axis for total power
    ax_power = ax_curr.twinx()
    ax_power.set_ylabel("Total Power (W)")
    ax_power.grid(True, linestyle="-", linewidth=0.8, alpha=0.7)

    # Calculate total power in watts
    total_power_watts = (left_servo_current + right_servo_current) / 1000.0 * V_supply
    ax_power.plot(time_servo_current, total_power_watts, color="#2ca02c", linestyle=":", linewidth=2, alpha=0.8, label="Total Power")

    # Calculate average power
    avg_power = total_power_watts.mean()
    ax_power.axhline(avg_power, color="#d62728", linestyle="--", linewidth=1.5, label=f"Avg Power = {avg_power:.2f} W")

    # Legends combined
    lines1, labels1 = ax_curr.get_legend_handles_labels()
    lines2, labels2 = ax_power.get_legend_handles_labels()
    ax_curr.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, ncol=2, fontsize=15, framealpha=0.9)

    axs2[1].set_xlabel("Time (s)")

    # --- Title ---
    fig2.suptitle(f"PWM and Servo Currents/Total Power from {file_name}", y=0.98)

    # Improve ticks
    for ax in axs2:
        ax.tick_params(axis="both")
    ax_power.tick_params(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-res second figure
    save_name3 = os.path.splitext(file_name)[0] + "_pwm_vs_power.png"
    if SAVE_FIG: 
        plt.savefig(save_name3, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name3}")


###
# Figure 4: Altitude
###
if PLOT_ALT:
    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # --- Top subplot: Altitude ---
    axs1.plot(time_alt, alt_imu, label="Measured Altitude (IMU)", color="#1f77b4", linewidth=2)
    axs1.set_ylabel("Altitude (m)")
    axs1.legend(loc="best", frameon=True, framealpha=0.9)
    axs1.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Title ---
    file_name = os.path.basename(esp_file_path)
    fig1.suptitle(f"Data from {file_name}", y=0.98)

    # Improve ticks
    ax.tick_params(axis="both")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-res first figure
    save_name1 = os.path.splitext(file_name)[0] + "_altitude.png"
    if SAVE_FIG: 
        plt.savefig(save_name1, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name1}")

###
# Figure 5: Linear Acceleration Analysis
###
if PLOT_LINEAR_ACC:
    fig5, axs5 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Convert acceleration from g to m/s²
    g = 9.81  # m/s² per g
    ax_mps2 = ax_imu * g
    ay_mps2 = ay_imu * g
    az_mps2 = (az_imu - 1) * g
    
    # --- Top subplot: Raw acceleration in g ---
    axs5[0].plot(time_linear_acc, ax_imu, label='$a_x$ (IMU)', color='#1f77b4', linewidth=2, alpha=0.8)
    axs5[0].plot(time_linear_acc, ay_imu, label='$a_y$ (IMU)', color='#d62728', linewidth=2, alpha=0.8)
    axs5[0].plot(time_linear_acc, az_imu, label='$a_z$ (IMU)', color='#2ca02c', linewidth=2, alpha=0.8)
    axs5[0].set_ylabel("Magnitude (g)")
    axs5[0].legend(loc="best", frameon=True, framealpha=0.9)
    axs5[0].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    axs5[0].set_title("Raw Accelerometer Data (Body Frame)")
    
    # --- Bottom subplot: Acceleration in m/s² ---
    axs5[1].plot(time_linear_acc, ax_mps2, label='$a_x$', color='#1f77b4', linewidth=2, alpha=0.8)
    axs5[1].plot(time_linear_acc, ay_mps2, label='$a_y$', color='#d62728', linewidth=2, alpha=0.8)
    axs5[1].plot(time_linear_acc, az_mps2, label='$a_z$', color='#2ca02c', linewidth=2, alpha=0.8)
    axs5[1].set_ylabel("Magnitude (m/s²)")
    axs5[1].set_xlabel("Time (s)")
    axs5[1].legend(loc="upper right", frameon=True, ncol=3, framealpha=0.9)
    axs5[1].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    axs5[1].set_title("Accelerometer Data in m/s²")
    
    # --- Title ---
    file_name = os.path.basename(esp_file_path)
    fig5.suptitle(f"Linear Acceleration Analysis from {file_name}", y=0.98)
    
    # Improve ticks
    for ax in axs5:
        ax.tick_params(axis="both")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save high-res figure
    save_name5 = os.path.splitext(file_name)[0] + "_linear_acceleration.png"
    if SAVE_FIG: 
        plt.savefig(save_name5, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name5}")

###
# Figure 6: Acceleration Magnitude and Statistics
###
if PLOT_LINEAR_ACC:
    # Calculate acceleration magnitude
    acc_magnitude_g = np.sqrt(ax_imu**2 + ay_imu**2 + az_imu**2)
    acc_magnitude_mps2 = acc_magnitude_g * g
    
    fig6, axs6 = plt.subplots(2, 1, figsize=(10, 6))
    
    # --- Top subplot: Acceleration magnitude ---
    axs6[0].plot(time_linear_acc, acc_magnitude_g, label='Total Linear Acceleration', 
                 color='#9467bd', linewidth=2)
    axs6[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Gravity Reference (1g)')
    axs6[0].set_xlabel("Time (s)")
    axs6[0].set_ylabel("Magnitude (g)")
    axs6[0].legend(loc="upper right", frameon=True, ncol=2, framealpha=0.9)
    axs6[0].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    axs6[0].set_title("Acceleration Magnitude")
    
    # --- Bottom subplot: Statistical measures ---
    # Calculate rolling statistics for smoother visualization
    window_size = min(50, len(time_linear_acc) // 10)  # Adaptive window size
    
    if window_size > 1:
        rolling_std = pd.Series(acc_magnitude_g).rolling(window=window_size, center=True).std()
        axs6[1].plot(time_linear_acc, rolling_std, 
                    label=f'Rolling Std Dev (window={window_size})', 
                    color='#e377c2', linewidth=2)
    else:
        axs6[1].plot(time_linear_acc, np.full_like(acc_magnitude_g, acc_magnitude_g.std()), 
                    label='Standard Deviation', color='#e377c2', linewidth=2)
    
    # Add horizontal lines for mean and ±1 std
    mean_acc = acc_magnitude_g.mean()
    std_acc = acc_magnitude_g.std()
    axs6[1].axhline(y=mean_acc, color='blue', linestyle='-', alpha=0.7, linewidth=1.5, 
                   label=f'Mean: {mean_acc:.3f}g')
    axs6[1].axhline(y=mean_acc + std_acc, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                   label=f'Mean ± Std: {mean_acc + std_acc:.3f}g')
    axs6[1].axhline(y=mean_acc - std_acc, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                   label=f'Mean - Std: {mean_acc - std_acc:.3f}g')
    
    axs6[1].set_ylabel("Statistical Measures (g)")
    axs6[1].set_xlabel("Time (s)")
    axs6[1].legend(loc="best", frameon=True, framealpha=0.9)
    axs6[1].grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    axs6[1].set_title("Acceleration Statistics")
    
    # --- Title ---
    fig6.suptitle(f"Acceleration Magnitude and Statistics from {file_name}", 
                 y=0.98)
    
    # Improve ticks
    for ax in axs6:
        ax.tick_params(axis="both")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save high-res figure
    save_name6 = os.path.splitext(file_name)[0] + "_acceleration_stats.png"
    if SAVE_FIG: 
        plt.savefig(save_name6, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name6}")
    
    # Print acceleration statistics
    print(f"\nAcceleration Statistics:")
    print(f"  X-axis: {ax_imu.mean():.3f} ± {ax_imu.std():.3f} g")
    print(f"  Y-axis: {ay_imu.mean():.3f} ± {ay_imu.std():.3f} g")  
    print(f"  Z-axis: {az_imu.mean():.3f} ± {az_imu.std():.3f} g")
    print(f"  Magnitude: {acc_magnitude_g.mean():.3f} ± {acc_magnitude_g.std():.3f} g")
    print(f"  Max magnitude: {acc_magnitude_g.max():.3f} g")
    print(f"  Min magnitude: {acc_magnitude_g.min():.3f} g")



###
# Figure 8: Pitch Control Performance with Command Signal
###
if PLOT_PITCH_CONTROL:
    fig3, ax_pitch = plt.subplots(figsize=(10, 4.5))

    # Plot pitch angles on left y-axis
    color_pitch = '#1f77b4'
    color_rls = '#ff7f0e'
    color_target = '#2ca02c'

    ax_pitch.plot(time_pitch, pitch_imu, label="Filtered Pitch", color=color_pitch, linewidth=2)
    ax_pitch.plot(time_pitch, pitch_rls, label="Estimated Pitch Offset", color=color_rls, linewidth=2, linestyle="--")
    ax_pitch.plot(time_pitch, pitch_target, label="Target Pitch", color=color_target, linewidth=2, linestyle=":")
    ax_pitch.set_ylabel("Pitch Angle (deg)", color=color_pitch)
    ax_pitch.tick_params(axis='y', labelcolor=color_pitch)
    ax_pitch.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_pitch.set_xlabel("Time (s)")

    # Plot command on right y-axis
    ax_cmd = ax_pitch.twinx()
    if 'offset' in esp_file_path.lower():
        ax_cmd.plot(time_cmd_offset_pitch, cmd_offset_pitch, color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.set_ylabel("Offset Command (deg)", color='#d62728')
    elif 'strokvel' in esp_file_path.lower():
        ax_cmd.plot(time_cmd_strokvel_pitch, cmd_strokvel_pitch, color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.set_ylabel("Stroke Timing Command", color='#d62728')
    else:
        # Plot both if neither is clearly identified
        ax_cmd.plot(time_cmd_offset_pitch, cmd_offset_pitch, label="Offset Command (deg)", color='#d62728', linewidth=2, alpha=0.8)
        ax_cmd.plot(time_cmd_strokvel_pitch, cmd_strokvel_pitch, label="Stroke Timing Command", color='#9467bd', linewidth=2, alpha=0.8, linestyle="--")
        ax_cmd.set_ylabel("Command Value", fontsize=13, color='#d62728')

    ax_cmd.tick_params(axis='y', labelcolor='#d62728')

    # Combine legends
    lines_pitch, labels_pitch = ax_pitch.get_legend_handles_labels()
    lines_cmd, labels_cmd = ax_cmd.get_legend_handles_labels()
    # ax_pitch.legend(lines_pitch + lines_cmd, labels_pitch + labels_cmd, fontsize=10, loc="upper center", 
    #             bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    ax_pitch.legend(loc='best', frameon=True, framealpha=0.9)

    fig3.suptitle(f"Pitch Control Performance with Command Signal from {file_name}", 
                y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-res first figure
    save_name1 = os.path.splitext(file_name)[0] + "_pitch_pid_tracking.png"
    if SAVE_FIG: 
        plt.savefig(save_name1, dpi=600, bbox_inches="tight")
        print(f"Figure saved as: {save_name1}")

###
# Figure 7: Acceleration vs Flapping Cycle Phase with Violin Plots
###
if PLOT_LINEAR_ACC:
    # Extract PWM1 signal for flapping cycle detection
    pwm_signal = pwm1_y.values if hasattr(pwm1_y, 'values') else pwm1_y
    time_pwm_array = time_pwm.values if hasattr(time_pwm, 'values') else time_pwm
    
    # Find peaks and troughs in PWM signal (flapping cycles)
    from scipy.signal import find_peaks
    
    # Find peaks (upstroke transitions)
    peaks, _ = find_peaks(pwm_signal, height=np.percentile(pwm_signal, 70), 
                         distance=10)
    
    # Find troughs (downstroke transitions)
    troughs, _ = find_peaks(-pwm_signal, height=np.percentile(-pwm_signal, 70), 
                           distance=10)
    
    # Ensure we have matching cycles (start with trough, end with peak)
    if len(troughs) > 0 and len(peaks) > 0:
        if peaks[0] < troughs[0]:
            peaks = peaks[1:]
        if troughs[-1] > peaks[-1]:
            troughs = troughs[:-1]
        
        # Create lists to store phase and acceleration data
        phase_data = []
        acc_magnitude_data = []
        
        # Calculate acceleration magnitude
        acc_magnitude_g = np.sqrt(ax_imu**2 + ay_imu**2 + az_imu**2)
        
        # Process each complete flapping cycle
        for i in range(min(len(troughs), len(peaks))):
            start_idx = troughs[i]  # Downstroke start
            end_idx = peaks[i]      # Upstroke end
            
            # Check if indices are within bounds of time_pwm_array
            if (end_idx < len(time_pwm_array) and start_idx < len(time_pwm_array) and 
                end_idx - start_idx > 5):
                
                cycle_time = time_pwm_array[end_idx] - time_pwm_array[start_idx]
                
                # Find acceleration data points within this cycle
                cycle_mask = (time_linear_acc >= time_pwm_array[start_idx]) & (time_linear_acc <= time_pwm_array[end_idx])
                cycle_acc_times = time_linear_acc[cycle_mask]
                cycle_acc_magnitude = acc_magnitude_g[cycle_mask]
                
                if len(cycle_acc_times) > 0:
                    # Normalize time within cycle (0 to 1)
                    normalized_times = (cycle_acc_times - time_pwm_array[start_idx]) / cycle_time
                    
                    phase_data.extend(normalized_times)
                    acc_magnitude_data.extend(cycle_acc_magnitude)
        
        # Create the combined scatter + violin plot
        if phase_data and acc_magnitude_data:
            fig_phase, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # Science Robotics Color Palette
            primary_blue = '#1A5276'      # Deep blue for main elements
            accent_orange = '#E67E22'     # Warm orange for highlights
            neutral_gray = '#7F8C8D'      # Gray for secondary elements
            light_blue = '#3498DB'        # Light blue for fills
            dark_blue = '#2C3E50'         # Dark blue for text/borders
            
            # Top: Raw PWM signal with detected cycles
            ax1.plot(time_pwm_array, pwm_signal, label='PWM1 Signal', color=primary_blue, linewidth=2)
            ax1.plot(time_pwm_array[peaks], pwm_signal[peaks], 'o', color=accent_orange, 
                    label='Upstroke Start', markersize=5, markeredgecolor='white', markeredgewidth=1)
            ax1.plot(time_pwm_array[troughs], pwm_signal[troughs], 's', color=primary_blue, 
                    label='Downstroke Start', markersize=5, markeredgecolor='white', markeredgewidth=1)
            ax1.set_ylabel('PWM Value')
            ax1.set_title('PWM Signal with Detected Flapping Cycles')
            ax1.legend(loc='best', frameon=True, framealpha=0.9)
            ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
            
            # Bottom: Scatter plot + Violin plot
            # Create phase bins for violin plots
            n_bins = 20
            phase_bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
            
            # Group acceleration data by phase bins and filter out empty bins
            binned_acc_data = []
            valid_bin_centers = []
            
            for i in range(len(phase_bins)-1):
                # For the last bin, include the upper boundary
                if i == len(phase_bins) - 2:  # Last bin
                    mask = (np.array(phase_data) >= phase_bins[i]) & (np.array(phase_data) <= phase_bins[i+1])
                else:
                    mask = (np.array(phase_data) >= phase_bins[i]) & (np.array(phase_data) < phase_bins[i+1])
                
                binned_acc = np.array(acc_magnitude_data)[mask]
                
                # Only include bins with sufficient data points for violin plot
                if len(binned_acc) >= 3:  # Reduced threshold for better coverage
                    binned_acc_data.append(binned_acc)
                    valid_bin_centers.append(bin_centers[i])
            
            # Create violin plots only for bins with data
            if binned_acc_data and len(binned_acc_data) >= 3:
                violin_parts = ax2.violinplot(binned_acc_data, positions=valid_bin_centers, 
                                             widths=0.035, showmeans=True, showmedians=False)
                
                # Customize violin plots with elegant colors
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(light_blue)
                    pc.set_alpha(0.4)
                    pc.set_edgecolor(primary_blue)
                    pc.set_linewidth(1)
                
                # Style the mean lines
                violin_parts['cmeans'].set_color(primary_blue)
                violin_parts['cmeans'].set_linewidth(2.5)
                violin_parts['cmeans'].set_linestyle('-')
                
                # Add elegant trend line
                bin_means = [np.mean(bin_data) for bin_data in binned_acc_data]
                ax2.plot(valid_bin_centers, bin_means, 
                        color=accent_orange, linewidth=3.5, linestyle='-', 
                        label='Mean Trend', marker='o', markersize=7, 
                        markerfacecolor=accent_orange, markeredgecolor='white', markeredgewidth=1.5)
            else:
                print("Insufficient data for violin plots - using only scatter plot")
                valid_bin_centers = bin_centers
            
            # Overlay scatter points with elegant styling
            scatter = ax2.scatter(phase_data, acc_magnitude_data, 
                                alpha=0.3, s=20, color=neutral_gray, edgecolors='none',
                                label='Individual Measurements')
            
            # Add phase region annotations with elegant styling
            ax2.axvline(x=0.0, color='#111111', linestyle='--', alpha=0.8, linewidth=2.5)
            ax2.axvline(x=0.5, color='#777777', linestyle='--', alpha=0.8, linewidth=2.5)
            ax2.axvline(x=1.0, color='#111111', linestyle='--', alpha=0.8, linewidth=2.5)
            
            # Elegant text annotations
            ax2.text(0.25, ax2.get_ylim()[1]*0.92, 'DOWNSTROKE', ha='center', fontsize=12,
                    fontweight='bold', color='#111111')
            ax2.text(0.75, ax2.get_ylim()[1]*0.92, 'UPSTROKE', ha='center', fontsize=12,
                    fontweight='bold', color='#777777')
            
            ax2.set_xlabel('Normalized Flapping Cycle')
            ax2.set_ylabel('Magnitude (g)')
            ax2.set_title('Acceleration Dynamics Throughout Flapping Cycle', 
                         pad=20)
            
            # Create elegant legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=neutral_gray, markersize=8, 
                       alpha=0.6, label='Measurements'),
                Line2D([0], [0], color=accent_orange, linewidth=3.5, marker='o', markersize=7, 
                       markerfacecolor=accent_orange, markeredgecolor='white', label='Mean Trend'),
            ]
            
            ax2.legend(handles=legend_elements, loc='lower left', frameon=True, ncol=2, fontsize=12,
                      framealpha=0.9, edgecolor=neutral_gray)
            
            ax2.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
            ax2.set_xlim(-0.02, 1.02)
            
            plt.tight_layout()
            
            # Save figure
            save_name_phase = os.path.splitext(file_name)[0] + "_acceleration_vs_flapping_phase_violin.png"
            if SAVE_FIG:
                plt.savefig(save_name_phase, dpi=600, bbox_inches='tight')
                print(f"Figure saved as: {save_name_phase}")
                
            # Print statistical insights
            print(f"\nFlapping Cycle Analysis:")
            print(f"Analyzed {len(peaks)} flapping cycles")
            print(f"Total data points: {len(phase_data)}")
            
            # Calculate average cycle duration safely
            cycle_durations = []
            for i in range(min(len(peaks), len(troughs))):
                if (peaks[i] < len(time_pwm_array) and troughs[i] < len(time_pwm_array)):
                    cycle_durations.append(time_pwm_array[peaks[i]] - time_pwm_array[troughs[i]])
            
            if cycle_durations:
                avg_cycle_duration = np.mean(cycle_durations)
                print(f"Average cycle duration: {avg_cycle_duration*1000:.1f} ms")
            
            print(f"Bins with sufficient data: {len(binned_acc_data)}/{n_bins}")
            
            # Calculate phase-specific statistics
            downstroke_mask = np.array(phase_data) < 0.5
            upstroke_mask = np.array(phase_data) >= 0.5
            
            if np.any(downstroke_mask) and np.any(upstroke_mask):
                downstroke_acc = np.array(acc_magnitude_data)[downstroke_mask]
                upstroke_acc = np.array(acc_magnitude_data)[upstroke_mask]
                
                print(f"Downstroke acceleration: {np.mean(downstroke_acc):.3f} ± {np.std(downstroke_acc):.3f} g (n={len(downstroke_acc)})")
                print(f"Upstroke acceleration: {np.mean(upstroke_acc):.3f} ± {np.std(upstroke_acc):.3f} g (n={len(upstroke_acc)})")
                
                if binned_acc_data and len(binned_acc_data) >= 3:
                    max_acc_bin_idx = np.argmax(bin_means)
                    print(f"Max acceleration typically occurs at phase: {valid_bin_centers[max_acc_bin_idx]:.2f}")
            
        else:
            print("No complete flapping cycles found with acceleration data")
    else:
        print("Could not detect clear flapping cycles in PWM signal")

plt.show()
