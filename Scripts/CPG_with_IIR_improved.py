import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Parameter settings ===
F_servo = 100       # MCU computation frequency, Hz
F_flap = 3.75       # Flapping frequency, Hz
T_total = 2         # Total simulation time (s)
dt = 1 / F_servo    # Time step
time = np.arange(0, T_total + dt, dt)  # Time vector
flapping_amplitude = 1 # np.deg2rad(60)

# Low-pass filter parameters
alpha = 0.18
p_smoothed_p = 0.5   # Filtering p approach
p_smoothed_r = 0.5   # Filtering 1/p approach

# Target p values (sinusoidal + noise)
p_target = 0.5 + 0.2 * np.sin(2 * np.pi * 2 * time) + 0.1 * np.random.randn(len(time))
p_target = np.clip(p_target, 0.3, 0.7)

# === Initialize data storage ===
w_list_p = np.zeros_like(time)
cos_list_p = np.zeros_like(time)
p_smooth_list_p = np.zeros_like(time)
dwdt_list_p = np.zeros_like(time)

w_list_r = np.zeros_like(time)
cos_list_r = np.zeros_like(time)
p_smooth_list_r = np.zeros_like(time)
dwdt_list_r = np.zeros_like(time)

w_unfiltered_list = np.zeros_like(time)
cos_unfiltered_list = np.zeros_like(time)
dwdt_unfiltered_list = np.zeros_like(time)

# === Initialization ===
w_old_p = 0
w_old_r = 0
w_unfiltered_old = 0

w_list_p[0] = w_old_p
cos_list_p[0] = np.cos(w_old_p)
p_smooth_list_p[0] = p_smoothed_p

w_list_r[0] = w_old_r
cos_list_r[0] = np.cos(w_old_r)
p_smooth_list_r[0] = p_smoothed_r

w_unfiltered_list[0] = w_unfiltered_old
cos_unfiltered_list[0] = np.cos(w_unfiltered_old)

# For filtering 1/p
r_target = 1 / p_target
r_smooth = np.zeros_like(r_target)
r_smooth[0] = 1 / p_smoothed_r

# === Main loop ===
for i in range(1, len(time)):
    # --- Method 1: filter p ---
    p_smoothed_p = alpha * p_target[i] + (1 - alpha) * p_smoothed_p
    dw_p = (np.pi * F_flap) / (F_servo * p_smoothed_p)
    w_new_p = w_old_p + dw_p

    w_list_p[i] = np.mod(w_new_p, 2 * np.pi)
    cos_list_p[i] = np.cos(w_list_p[i]) * flapping_amplitude
    p_smooth_list_p[i] = p_smoothed_p
    dwdt_list_p[i] = dw_p * F_servo

    w_old_p = w_new_p

    # --- Method 2: filter 1/p ---
    r_smooth[i] = alpha * r_target[i] + (1 - alpha) * r_smooth[i - 1]
    dw_r = (np.pi * F_flap) / F_servo * r_smooth[i]
    w_new_r = w_old_r + dw_r

    w_list_r[i] = np.mod(w_new_r, 2 * np.pi)
    cos_list_r[i] = np.cos(w_list_r[i]) * flapping_amplitude
    p_smooth_list_r[i] = 1 / r_smooth[i]
    dwdt_list_r[i] = dw_r * F_servo

    w_old_r = w_new_r

    # --- Unfiltered reference ---
    dw_unfiltered = (np.pi * F_flap) / (F_servo * p_target[i])
    w_unfiltered_new = w_unfiltered_old + dw_unfiltered

    w_unfiltered_list[i] = np.mod(w_unfiltered_new, 2 * np.pi)
    cos_unfiltered_list[i] = np.cos(w_unfiltered_list[i])
    dwdt_unfiltered_list[i] = dw_unfiltered * F_servo

    w_unfiltered_old = w_unfiltered_new

# === Compute derivatives ===
dcos_dt_p = np.diff(cos_list_p) / dt
dcos_dt_r = np.diff(cos_list_r) / dt
dcos_dt_unfiltered = np.diff(cos_unfiltered_list) / dt
t_diff = time[1:]

# === Phase error ===
error_p = cos_unfiltered_list - cos_list_p
error_r = cos_unfiltered_list - cos_list_r

# === Plotting ===
mpl.rcParams.update({
    "font.size": 15,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})

fig = plt.figure(figsize=(15, 6))

# Top row: 3 subplots
ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)  # Modulation parameter
ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)  # Angular velocity
ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)  # Error

# Bottom row: 2 subplots 
ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=3)  # Stroke angle
ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=3)  # Stroke velocity

# Adjust spacing
plt.subplots_adjust(wspace=0.8, hspace=0.35, top=0.9, bottom=0.1, left=0.08, right=0.98)

font_size = 16
line_width = 2.0

# Colors 
color_unfiltered = [0.2, 0.2, 0.2, 0.5]      # Dark gray
color_filter_p = [0, 0.447, 0.741]      # Blue
color_filter_r = [0.851, 0.325, 0.098]  # Orange-red

# Global axis style for all subplots
all_axes = [ax1, ax2, ax3, ax4, ax5]
for ax in all_axes:
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Top row: Modulation parameter, Angular velocity, and Error
# Subplot 1: Target and smoothed p (top left)
ax1.plot(time, p_target, '-', lw=1.5, color=color_unfiltered, label='Unfiltered')
ax1.plot(time, p_smooth_list_p, '-', lw=line_width, color=color_filter_p, label=r'Filtered $p$')
ax1.plot(time, p_smooth_list_r, '-', lw=line_width, color=color_filter_r, label=r'Filtered $1/p$')
# ax1.set_ylabel(r"Modulation Parameter $p$", fontsize=font_size)
ax1.set_ylabel(r"Modulation Parameter", fontsize=font_size)
ax1.set_ylim([0.2, 0.8])
ax1.set_xlabel(r"Time $t$ (s)", fontsize=font_size)

# Subplot 2: Angular velocity (top middle)
ax2.plot(time, dwdt_unfiltered_list, '-', lw=1.5, color=color_unfiltered, label='Unfiltered')
ax2.plot(time, dwdt_list_p, '-', lw=line_width, color=color_filter_p)
ax2.plot(time, dwdt_list_r, '-', lw=line_width, color=color_filter_r)
# ax2.set_ylabel(r"Angular Velocity $\dot{\omega}$ (rad/s)", fontsize=font_size)
ax2.set_ylabel(r"Angular Velocity (rad/s)", fontsize=font_size)
ax2.set_xlabel(r"Time $t$ (s)", fontsize=font_size)

# Subplot 3: Phase error (top right)
ax3.plot(time, error_p, '-', lw=line_width, color=color_filter_p)
ax3.plot(time, error_r, '-', lw=line_width, color=color_filter_r)
# ax3.set_ylabel(r"Error $\Delta \cos(\omega)$ (rad)", fontsize=font_size)
ax3.set_ylabel(r"Stroke Angle Error (rad)", fontsize=font_size)
ax3.set_xlabel(r"Time $t$ (s)", fontsize=font_size)

# Bottom row: Stroke angle and Stroke velocity
# Subplot 4: Wing position (bottom left)
ax4.plot(time, cos_unfiltered_list, '-', lw=1.5, color=color_unfiltered, label='Unfiltered')
ax4.plot(time, cos_list_p, '-', lw=line_width, color=color_filter_p)
ax4.plot(time, cos_list_r, '-', lw=line_width, color=color_filter_r)
# ax4.set_ylabel(r"Stroke Angle $\cos(\omega)$ (rad)", fontsize=font_size)
ax4.set_ylabel(r"Stroke Angle (rad)", fontsize=font_size)
ax4.set_ylim([-1.1, 1.1])
ax4.set_xlabel(r"Time $t$ (s)", fontsize=font_size)

# Subplot 5: Wing velocity derivative (bottom right)
ax5.plot(t_diff, dcos_dt_unfiltered, '-', lw=1.5, color=color_unfiltered, label='Unfiltered')
ax5.plot(t_diff, dcos_dt_p, '-', lw=line_width, color=color_filter_p)
ax5.plot(t_diff, dcos_dt_r, '-', lw=line_width, color=color_filter_r)
# ax5.set_ylabel(r"Stroke Velocity $\frac{d}{dt}\cos(\omega)$ (rad/s)", fontsize=font_size)
ax5.set_ylabel(r"Stroke Velocity (rad/s)", fontsize=font_size)
ax5.set_xlabel(r"Time $t$ (s)", fontsize=font_size)

# Final styling
for ax in all_axes:
    ax.tick_params(labelsize=font_size)
    ax.set_xlim([0, T_total])

# Shared legend above the subplots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=3,
           frameon=False, fontsize=font_size-1)

fig.patch.set_facecolor("white")
plt.savefig("fig_cpg_with_iir.tiff", dpi=600, bbox_inches='tight')
plt.show()


# === Performance analysis ===
print("=== PERFORMANCE COMPARISON ===\n")

mean_error_p = np.mean(np.abs(error_p))
mean_error_r = np.mean(np.abs(error_r))
max_error_p = np.max(np.abs(error_p))
max_error_r = np.max(np.abs(error_r))

print("Phase Error Statistics:")
print(f"Mean absolute error (filter p): {mean_error_p:.4f}")
print(f"Mean absolute error (filter 1/p): {mean_error_r:.4f}")
print(f"Max absolute error (filter p): {max_error_p:.4f}")
print(f"Max absolute error (filter 1/p): {max_error_r:.4f}")
print(f"Error reduction: {(mean_error_p - mean_error_r)/mean_error_p*100:.1f}%")

smoothness_p = np.std(np.diff(dwdt_list_p))
smoothness_r = np.std(np.diff(dwdt_list_r))
smoothness_unfiltered = np.std(np.diff(dwdt_unfiltered_list))

print("\nAngular Velocity Smoothness (std of acceleration):")
print(f"Unfiltered: {smoothness_unfiltered:.2f} rad/s²")
print(f"Filter p: {smoothness_p:.2f} rad/s²")
print(f"Filter 1/p: {smoothness_r:.2f} rad/s²")

fc = (alpha * F_servo) / (2 * np.pi * (1 - alpha))
print("\nFilter Characteristics:")
print(f"Cutoff frequency: {fc:.2f} Hz")
print(f"Flapping frequency: {F_flap:.2f} Hz")
print(f"Ratio fc/F_flap: {fc/F_flap:.2f}")
