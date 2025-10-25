import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})
font_size = 16

# === Time & signals ===
flapping_frequency = 3.75  # Hz
cycle_duration = 1 / flapping_frequency
start_time = 0.2
end_time = start_time + 2 * cycle_duration
t = np.linspace(0, end_time, 1000)

# Trimmed window
time_mask = (t >= start_time) & (t <= end_time)
t_trimmed = t[time_mask]

# Signals
flapping = (np.pi/3) * np.sin(2 * np.pi * flapping_frequency * t)
pitch = (-np.pi/6) * np.sin(2 * np.pi * flapping_frequency * (t - 1/50)) - np.pi/6

flapping = np.rad2deg(flapping)
pitch = np.rad2deg(pitch)
flapping_trimmed = flapping[time_mask]
pitch_trimmed = pitch[time_mask]

# HSB into RGB function
def hsb_to_rgb(h, s, v):
    h = h / 360.0; s = s / 100.0; v = v / 100.0
    if s == 0: return (v, v, v)
    i = int(h * 6); f = h * 6 - i
    p = v * (1 - s); q = v * (1 - s * f); t_val = v * (1 - s * (1 - f))
    if i % 6 == 0: r, g, b = v, t_val, p
    elif i % 6 == 1: r, g, b = q, v, p
    elif i % 6 == 2: r, g, b = p, v, t_val
    elif i % 6 == 3: r, g, b = p, q, v
    elif i % 6 == 4: r, g, b = t_val, p, v
    else: r, g, b = v, p, q
    return (r, g, b)

left_wing_color = hsb_to_rgb(217, 57, 69)
right_wing_color = hsb_to_rgb(358, 60, 77)
pitch_color = (148/255, 33/255, 146/255)

# === Stroke reversal detection ===
# Derivative and its sign
df = np.diff(flapping_trimmed)
sign_df = np.sign(df).astype(int)  # values in {-1, 0, +1}

# Replace isolated 0's in sign_df by nearest non-zero to avoid spurious transitions
sd = sign_df.copy()
n = len(sd)
# Forward-fill zeros where possible
for i in range(n):
    if sd[i] == 0:
        j = i + 1
        while j < n and sd[j] == 0:
            j += 1
        if j < n and sd[j] != 0:
            sd[i] = sd[j]
        elif i > 0:
            sd[i] = sd[i-1]
        else:
            sd[i] = 1  # Fallback

# Detect sign changes in derivative
diff_sign = np.diff(sd)  # Change between consecutive derivative signs

# IMPORTANT:
# - maxima (positive slope -> negative slope) => sign change +1 -> -1 => diff == (-2)
# - minima (negative slope -> positive slope) => sign change -1 -> +1 => diff == (+2)
# For "upstroke extreme moving downward" we usually want maxima => use -2.
maxima_idx = np.where(diff_sign == -2)[0] + 1  # indices in trimmed-array where a maximum occurs
minima_idx = np.where(diff_sign == 2)[0] + 1   # indices in trimmed-array where a minimum occurs

# Build downstroke segments from each maximum to the next minimum 
down_segments = []
for m in maxima_idx:
    # Find next minima after this maximum
    next_mins = minima_idx[minima_idx > m]
    if next_mins.size > 0:
        e = next_mins[0]
    else:
        e = len(t_trimmed) - 1  # If no next minima found, clip to end
    s = m
    # Ensure segment is non-empty and within bounds
    if 0 <= s < len(t_trimmed) and 0 <= e < len(t_trimmed) and e > s:
        down_segments.append((s, e))

print("Maxima indices (trimmed):", maxima_idx.tolist())
print("Minima indices (trimmed):", minima_idx.tolist())
print("Downstroke segments:", down_segments)
print("Downstroke times (s):", [(t_trimmed[s], t_trimmed[e]) for s, e in down_segments])

# === Plotting ===
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=300)

# Shade downstroke regions 
for s_idx, e_idx in down_segments:
    ax1.axvspan(t_trimmed[s_idx], t_trimmed[e_idx], alpha=0.6, color='lightgray', zorder=0, edgecolor='none')

# Plot flapping (left/right) and pitch angle
ax1.plot(t_trimmed, flapping_trimmed, label='Left Wing', color=left_wing_color, linewidth=2.5, zorder=3)
ax1.plot(t_trimmed, -flapping_trimmed, label='Right Wing', color=right_wing_color, linewidth=2.5, zorder=3)
ax1.set_ylabel('PWM', fontsize=font_size, labelpad=10)
ax1.tick_params(axis='y', which='both', left=False, labelleft=False)

ax2 = ax1.twinx()
ax2.plot(t_trimmed, pitch_trimmed, label='Pitch Angle', color=pitch_color, linewidth=2.5, linestyle="--", zorder=4)
ax2.set_ylabel('Angle (°)', fontsize=font_size, labelpad=10, color=pitch_color)
ax2.tick_params(axis='y', labelcolor=pitch_color)

ax1.set_title('Wing Kinematics and Body Undulation', fontsize=font_size, pad=20)
ax1.set_xlabel('Time (s)', fontsize=font_size, labelpad=10)
ax1.grid(True, linestyle='--', alpha=0.1)
ax1.set_xlim(start_time, end_time)

# Combined legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=4, framealpha=0.9, edgecolor='none', fontsize=font_size-1)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('fig_body_undulation.png', dpi=600, bbox_inches='tight', transparent=False)
plt.show()
