import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 15,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})

# Parameters
NUM_CYCLE = 1
fs = 1000
t = np.linspace(0, NUM_CYCLE, int(fs * NUM_CYCLE), endpoint=False)
A1 = 1.0
phase_shift = 0.25  # 90° phase shift in cycles
nominal = A1 * np.sin(2 * np.pi * (t+phase_shift))  # 1 Hz
nominal_ptp = np.ptp(nominal)  # peak-to-peak

# === Method 1: Split-Cycle Function ===
A2_values = [0.1, 0.3, 0.5]
split_signals = []

for A2 in A2_values:
    # Apply 90° phase shift to both harmonics
    split_raw = A1 * np.sin(2 * np.pi * (t + phase_shift)) + A2 * np.sin(2 * np.pi * 2 * (t + phase_shift))

    # Normalize to match nominal peak-to-peak amplitude
    split_ptp = np.ptp(split_raw)
    scaling_factor = nominal_ptp / split_ptp
    split_signals.append(split_raw * scaling_factor)

# === Method 2: Piecewise Cosine Function ===
def phi_asymmetric(t, p, k1=2.0, k2=0.0):
    """Generate asymmetric flapping profile for multiple cycles."""
    phi_vals = np.zeros_like(t)
    t_cyclic = t % 1  # Wrap into [0, 1)
    mask_down = t_cyclic <= p
    t_down = t_cyclic[mask_down] / p
    phi_vals[mask_down] = (k1 / 2) * np.cos(np.pi * t_down) + k2
    mask_up = t_cyclic > p
    t_up = (t_cyclic[mask_up] - p) / (1 - p)
    phi_vals[mask_up] = (k1 / 2) * np.cos(np.pi * (1 + t_up)) + k2
    return phi_vals

p_values = [0.3, 0.4, 0.5, 0.6, 0.7]
cosine_signals = [phi_asymmetric(t, p) for p in p_values]

# === Method 3: Polynomial Time Warping Function ===
def t_star(t, p):
    """Time warping function."""
    numerator1 = (p - 0.5) * t**2
    numerator2 = (0.5 - p**2) * t
    denominator = p - p**2
    return (numerator1 + numerator2) / denominator

def phi_polynomial(t, p, k1=1.0, k2=0.0):
    """Generate flapping profile using polynomial time warping."""
    # Wrap time to [0, 1) for each cycle
    t_cyclic = t % 1
    t_star_vals = t_star(t_cyclic, p)
    
    # Generate the cosine signal with range [-1, 1]
    signal = np.cos(2 * np.pi * t_star_vals)
    
    # Scale and offset to match the range of other methods
    return k1 * signal + k2

# Parameters for polynomial method (adjusted to match range of other methods)
k1_poly = 1.0  # amplitude
k2_poly = 0.0  # offset
p_poly_values = [0.1, 0.3, 0.5, 0.7, 0.9]
polynomial_signals = [phi_polynomial(t, p, k1_poly, k2_poly) for p in p_poly_values]

# === First Derivative Calculation ===
dt = t[1] - t[0]  # time step

nominal_deriv = np.gradient(nominal, dt)
split_derivs = [np.gradient(split, dt) for split in split_signals]
cosine_derivs = [np.gradient(signal, dt) for signal in cosine_signals]
polynomial_derivs = [np.gradient(signal, dt) for signal in polynomial_signals]

# === Create a single figure with subplots ===
fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=300, sharex=True)
fig.subplots_adjust(top=0.92, hspace=0.2, wspace=0.15)

# Define colors for different methods
colors_split = ['#1f77b4', '#ff7f0e', '#ff69b4']  # Blue, Orange, Pink
colors_cosine = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # Green, Purple, Brown, Pink, Gray
colors_poly = ['#d62728', '#17becf', '#bcbd22', '#ff9896', '#c5b0d5']  # Red, Cyan, Olive, Light Red, Light Purple

# === Left column: Stroke Angles ===

# Row 1: Split-Cycle method
ax1 = axes[1, 0]
ax1.plot(t, nominal, 'k-', alpha=0.7, linewidth=5, label='Nominal')
for i, (A2, split) in enumerate(zip(A2_values, split_signals)):
    ax1.plot(t, split, color=colors_split[i], linewidth=2.5, label=f'$A_2$={A2}')
ax1.set_ylabel("Stroke Angle (rad)", fontsize=15)
ax1.set_title("Split-Cycle Method", fontsize=15, fontweight='bold')
ax1.set_ylim(-1.2, 1.2)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right', fontsize=15, frameon=True, facecolor="white", framealpha=0.5)
# ax1.text(0.02, 0.95, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

# Row 2: Piecewise Cosine method
ax2 = axes[2, 0]
ax2.plot(t, nominal, 'k-', alpha=0.7, linewidth=5, label='Nominal')
for i, (p, signal) in enumerate(zip(p_values, cosine_signals)):
    ax2.plot(t, signal, color=colors_cosine[i], linewidth=2.5, label=f'p={p}')
ax2.set_ylabel("Stroke Angle (rad)", fontsize=15)
ax2.set_xlabel("Time (cycles)", fontsize=15)
ax2.set_title("Piecewise Cosine Method", fontsize=15, fontweight='bold')
ax2.set_ylim(-1.2, 1.2)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right', fontsize=15, frameon=True, facecolor="white", framealpha=0.5)
# ax2.text(0.02, 0.95, 'C', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

# Row 3: Polynomial Warping method
ax3 = axes[0, 0]
ax3.plot(t, nominal, 'k-', alpha=0.7, linewidth=5, label='Nominal')
for i, (p, signal) in enumerate(zip(p_poly_values, polynomial_signals)):
    ax3.plot(t, signal, color=colors_poly[i], linewidth=2.5, label=f'p={p}')
ax3.set_ylabel("Stroke Angle (rad)", fontsize=15)
ax3.set_title("Polynomial Warping Method", fontsize=15, fontweight='bold')
ax3.set_ylim(-1.2, 1.2)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend(loc='upper right', fontsize=15, frameon=True, facecolor="white", framealpha=0.5)
# ax3.text(0.02, 0.95, 'E', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')

# === Right column: Derivatives ===

# Row 1: Split-Cycle method derivatives
ax4 = axes[1, 1]
ax4.plot(t, nominal_deriv, 'k-', alpha=0.7, linewidth=3)
for i, (A2, deriv) in enumerate(zip(A2_values, split_derivs)):
    ax4.plot(t, deriv, color=colors_split[i], linewidth=2.5)
ax4.set_title("Split-Cycle Derivatives", fontsize=15, fontweight='bold')
ax4.set_ylabel("Angular Velocity (rad/cycle)", fontsize=15)
ax4.grid(True, linestyle=':', alpha=0.7)
# ax4.text(0.02, 0.95, 'B', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top')

# Row 2: Piecewise Cosine method derivatives
ax5 = axes[2, 1]
ax5.plot(t, nominal_deriv, 'k-', alpha=0.7, linewidth=3)
for i, (p, deriv) in enumerate(zip(p_values, cosine_derivs)):
    ax5.plot(t, deriv, color=colors_cosine[i], linewidth=2.5)
ax5.set_title("Piecewise Cosine Derivatives", fontsize=15, fontweight='bold')
ax5.set_ylabel("Angular Velocity (rad/cycle)", fontsize=15)
ax5.set_xlabel("Time (cycles)", fontsize=15)
ax5.grid(True, linestyle=':', alpha=0.7)
# ax5.text(0.02, 0.95, 'D', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top')

# Row 3: Polynomial Warping method derivatives
ax6 = axes[0, 1]
ax6.plot(t, nominal_deriv, 'k-', alpha=0.7, linewidth=3)
for i, (p, deriv) in enumerate(zip(p_poly_values, polynomial_derivs)):
    ax6.plot(t, deriv, color=colors_poly[i], linewidth=2.5)
ax6.set_title("Polynomial Warping Derivatives", fontsize=15, fontweight='bold')
ax6.set_ylabel("Angular Velocity (rad/cycle)", fontsize=15)
ax6.grid(True, linestyle=':', alpha=0.7)
# ax6.text(0.02, 0.95, 'F', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top')

# Add overall title
fig.suptitle("Comparison of Flapping Wing Stroke Patterns and Their Derivatives", 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
fig.patch.set_facecolor("white")
plt.savefig('fig_flapping_patterns_comparison_complete.tiff', dpi=600, bbox_inches='tight')
plt.show()