import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Parameters ===
F_servo = 1000        # MCU calculation frequency, Hz
F_flap = 3.75         # Flapping frequency, Hz
T_total = 1 / F_flap  # Total simulation time (s)
dt = 1 / F_servo      # Time step
time = np.arange(0, T_total + dt, dt)

A_values = [-0.4, -0.2, 0, 0.2, 0.4]  # Different A values
colors = plt.cm.tab10(np.linspace(0, 1, len(A_values)))  # High-contrast colormap

# === Initialize storage ===
cos_all, dcos_all, w_all, p_all = [], [], [], []

# === Main loop ===
for A in A_values:
    w_old = 0
    w_list = np.zeros_like(time)
    cos_list = np.zeros_like(time)
    p_list = np.zeros_like(time)

    w_list[0] = w_old
    cos_list[0] = np.sin(w_old)
    p_list[0] = 0

    for i in range(1, len(time)):
        p_val = A * np.cos(w_old) + 0.5       # Sinusoidal modulation
        dw = (2 * np.pi / F_servo) * F_flap / (2 * p_val)
        w_new = w_old + dw

        w_list[i] = np.mod(w_new, 2 * np.pi)  # Keep in [0, 2π)
        cos_list[i] = np.sin(w_list[i])
        p_list[i] = p_val
        w_old = w_new

    dcos_dt = np.diff(cos_list) / dt

    cos_all.append(cos_list)
    dcos_all.append(dcos_dt)
    w_all.append(w_list)
    p_all.append(p_list)

# === Plotting ===
mpl.rcParams.update({
    "font.size": 15,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})

fontsize = 16
fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharex=False)
fig.subplots_adjust(wspace=0.4, top=0.85, bottom=0.15, left=0.08, right=0.95)

line_width = 2.0

# --- Subplot A: p(w) ---
for idx, A in enumerate(A_values):
    w = w_all[idx]
    p_vals = p_all[idx]

    # Remove the spurious first point if it starts at (0,0)
    if np.isclose(w[0], 0.0) and np.isclose(p_vals[0], 0.0):
        w = w[1:]
        p_vals = p_vals[1:]

    # Find discontinuities when phase wraps from ~2π -> 0
    discontinuities = np.where(np.diff(w) < -5)[0] + 1
    split_indices = np.split(np.arange(len(w)), discontinuities)

    # Plot each continuous segment separately
    for seg in split_indices:
        axs[0].plot(w[seg], p_vals[seg], lw=line_width, color=colors[idx],
                    label=fr"$A={A:.1f}$" if seg is split_indices[0] else "")


axs[0].set_xlabel(r"Phase $\it{w}$ (rad)", fontsize=fontsize)
# axs[0].set_ylabel(r"$p$", fontsize=fontsize, fontweight="bold")
axs[0].set_ylabel(r"Modulation Function $p(w)$", fontsize=fontsize)
# axs[0].set_title(r"Modulation Function $p(w)$", fontsize=fontsize, fontweight="bold")
axs[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
axs[0].set_xlim([0, 2*np.pi])
axs[0].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axs[0].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axs[0].set_ylim([0.1, 0.9])

# --- Subplot B: cos(w) vs time ---
for idx in range(len(A_values)):
    axs[1].plot(time, cos_all[idx], lw=line_width, color=colors[idx])

axs[1].set_xlabel(r"Time $\it{t}$ (s)", fontsize=fontsize)
# axs[1].set_ylabel(r"$\cos(\it{w})$", fontsize=fontsize, fontweight="bold")
axs[1].set_ylabel(r"Stroke Angle (rad)", fontsize=fontsize)
# axs[1].set_title("Flapping Motion", fontsize=fontsize, fontweight="bold")
axs[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
axs[1].set_xlim([0, T_total])
axs[1].set_ylim([-1.1, 1.1])

# --- Subplot C: dcos(w)/dt vs time ---
t_diff = time[1:]
for idx in range(len(A_values)):
    axs[2].plot(t_diff, dcos_all[idx], lw=line_width, color=colors[idx])

axs[2].set_xlabel(r"Time $\it{t}$ (s)", fontsize=15)
# axs[2].set_ylabel(r"$\frac{d}{dt}\cos(\it{w})$", fontsize=15, fontweight="bold")
axs[2].set_ylabel("Stroke Velocity (rad/s)", fontsize=15)
# axs[2].set_title("Time Derivative of Flapping Pattern", fontsize=17, fontweight="bold")
axs[2].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
axs[2].set_xlim([0, T_total])

# --- Shared legend above all subplots ---
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.05), ncol=len(A_values),
           frameon=False, fontsize=fontsize-1)

fig.patch.set_facecolor("white")
plt.savefig("fig_cpg_ablation.tiff", dpi=600, bbox_inches='tight')
plt.show()