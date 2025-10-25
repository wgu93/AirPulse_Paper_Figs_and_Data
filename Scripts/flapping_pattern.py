import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib as mpl

plt.style.use('default')
mpl.rcParams.update({
    "font.size": 16,
    "axes.linewidth": 1.2,
    "font.family": "Arial",
    "mathtext.fontset": "cm",  
})

def create_figure_1():
    """Fixed f=3.75Hz, delta=0: A = [40, 45, 50, 55, 60, 65, 70]"""
    A_values = [40, 45, 50, 55, 60, 65, 70]
    f = 3.75
    delta = 0
    
    file_order = A_values
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_order)))
    color_map = dict(zip(file_order, colors))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t = np.linspace(0, 1/f, 1000)  # One cycle
    
    for A in A_values:
        y = A * np.sin(2 * np.pi * f * t) + delta
        ax.plot(t * 1000, y, label=f'{A}°', color=color_map[A], linewidth=3)
    
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stroke Angle [°]')
    ax.set_title('Flapping Pattern: y = A·sin(2πft) + δ\n(f = 3.75 Hz, δ = 0°)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_xlim(0, 1000/f)
    
    return fig

def create_figure_2():
    """Fixed A=65, delta=0: f = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]"""
    A = 65
    f_values = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    delta = 0
    
    file_order = f_values
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_order)))
    color_map = dict(zip(file_order, colors))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for f_val in f_values:
        t = np.linspace(0, 1/f_val, 1000)  # One cycle
        y = A * np.sin(2 * np.pi * f_val * t) + delta
        ax.plot(t * 1000, y, label=f'{f_val} Hz', color=color_map[f_val], linewidth=3)
    
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stroke Angle [°]')
    ax.set_title('Flapping Pattern: y = A·sin(2πft) + δ\n(A = 65°, δ = 0°)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_xlim(0, 1000/2.5)  # Set xlim based on lowest frequency
    
    return fig

def create_figure_3():
    """Fixed A=65, f=3.75: delta = [-15, -10, -5, 0, 5, 10, 15]"""
    A = 65
    f = 3.75
    delta_values = [-15, -10, -5, 0, 5, 10, 15]
    
    file_order = delta_values
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_order)))
    color_map = dict(zip(file_order, colors))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t = np.linspace(0, 1/f, 1000)  # One cycle
    
    for delta in delta_values:
        y = A * np.sin(2 * np.pi * f * t) + delta
        ax.plot(t * 1000, y, label=f'{delta}°', color=color_map[delta], linewidth=3)
    
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stroke Angle [°]')
    ax.set_title('Flapping Pattern: y = A·sin(2πft) + δ\n(A = 65°, f = 3.75 Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_xlim(0, 1000/f)
    
    return fig

def create_figure_4():
    """Fixed A=65, f=3.75: P = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]"""
    A = 65
    f_nominal = 3.75
    P_values = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    
    file_order = P_values
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_order)))
    color_map = dict(zip(file_order, colors))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for P in P_values:
        # Solve for omega using the differential equation
        t_span = (0, 1/f_nominal)  # One nominal cycle
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # Numerical integration of omega_dot = pi*f/p(omega)
        omega = np.zeros_like(t_eval)
        dt = t_eval[1] - t_eval[0]
        
        for i in range(1, len(t_eval)):
            p_omega = 0.5 + P * np.cos(omega[i-1])
            if abs(p_omega) < 1e-6:  # Avoid division by zero
                p_omega = 1e-6 if p_omega >= 0 else -1e-6
            omega_dot = np.pi * f_nominal / p_omega
            omega[i] = omega[i-1] + omega_dot * dt
        
        y = A * np.sin(omega)
        ax.plot(t_eval * 1000, y, label=f'A = {P}', color=color_map[P], linewidth=3)
    
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stroke Angle [°]')
    ax.set_title('Flapping Pattern: y = A·sin(ω), ω̇ = πf/p(ω), p(ω) = 0.5 + P·cos(ω)\n(A = 65°, f = 3.75 Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_xlim(0, 1000/f_nominal)
    
    return fig

def create_all_figures():
    """Create all four figures and display them"""
    print("Creating Figure 1: Varying Amplitude (A)")
    fig1 = create_figure_1()
    fig1.tight_layout()
    
    print("Creating Figure 2: Varying Frequency (f)")
    fig2 = create_figure_2()
    fig2.tight_layout()
    
    print("Creating Figure 3: Varying Offset (δ)")
    fig3 = create_figure_3()
    fig3.tight_layout()
    
    print("Creating Figure 4: Varying Asymmetry Parameter (P)")
    fig4 = create_figure_4()
    fig4.tight_layout()
    
    return fig1, fig2, fig3, fig4

def create_combined_figure():
    """Create a single figure with all four subplots"""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Figure 1: Varying Amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    A_values = [40, 45, 50, 55, 60, 65, 70]
    f = 3.75
    delta = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(A_values)))
    color_map = dict(zip(A_values, colors))
    
    t = np.linspace(0, 1/f, 1000)
    for A in A_values:
        y = A * np.sin(2 * np.pi * f * t) + delta
        ax1.plot(t * 1000, y, label=f'ζ = {A}°', color=color_map[A], linewidth=3)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Stroke Angle [°]')
    ax1.set_title('(a) Varying Amplitude\nf = 3.75 Hz, δ = 0°')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=15, columnspacing=0.3, handletextpad=0.3, handlelength=0.8)
    ax1.set_xlim(0, 1000/f)
    
    # Figure 2: Varying Frequency
    ax2 = fig.add_subplot(gs[0, 1])
    A = 65
    f_values = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    delta = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(f_values)))
    color_map = dict(zip(f_values, colors))
    
    for f_val in f_values:
        t = np.linspace(0, 1/f_val, 1000)
        y = A * np.sin(2 * np.pi * f_val * t) + delta
        ax2.plot(t * 1000, y, label=f'f = {f_val} Hz', color=color_map[f_val], linewidth=3)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Stroke Angle [°]')
    ax2.set_title('(b) Varying Frequency\nA = 65°, δ = 0°')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=15, columnspacing=0.3, handletextpad=0.3, handlelength=0.8)
    ax2.set_xlim(0, 1000/2.5)
    
    # Figure 3: Varying Offset
    ax3 = fig.add_subplot(gs[1, 0])
    A = 65
    f = 3.75
    delta_values = [-15, -10, -5, 0, 5, 10, 15]
    colors = plt.cm.tab10(np.linspace(0, 1, len(delta_values)))
    color_map = dict(zip(delta_values, colors))
    
    t = np.linspace(0, 1/f, 1000)
    for delta in delta_values:
        y = A * np.sin(2 * np.pi * f * t) + delta
        ax3.plot(t * 1000, y, label=f'δ = {delta}°', color=color_map[delta], linewidth=3)
    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('Stroke Angle [°]')
    ax3.set_title('(c) Varying Offset\nA = 65°, f = 3.75 Hz')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=15, columnspacing=0.3, handletextpad=0.3, handlelength=0.8)
    ax3.set_xlim(0, 1000/f)
    
    # Figure 4: Varying Asymmetry Parameter
    ax4 = fig.add_subplot(gs[1, 1])
    A = 65
    f_nominal = 3.75
    P_values = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    colors = plt.cm.tab10(np.linspace(0, 1, len(P_values)))
    color_map = dict(zip(P_values, colors))
    
    for P in P_values:
        t_span = (0, 1/f_nominal)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        omega = np.zeros_like(t_eval)
        dt = t_eval[1] - t_eval[0]
        
        for i in range(1, len(t_eval)):
            p_omega = 0.5 + P * np.cos(omega[i-1])
            if abs(p_omega) < 1e-6:
                p_omega = 1e-6 if p_omega >= 0 else -1e-6
            omega_dot = np.pi * f_nominal / p_omega
            omega[i] = omega[i-1] + omega_dot * dt
        
        y = A * np.sin(omega)
        ax4.plot(t_eval * 1000, y, label=f'A = {P}', color=color_map[P], linewidth=3)
    
    ax4.set_xlabel('Time [ms]')
    ax4.set_ylabel('Stroke Angle [°]')
    ax4.set_title('(d) Varying Asymmetry Parameter\nA = 65°, f = 3.75 Hz')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=15, ncol=2, columnspacing=0.3, handletextpad=0.3, handlelength=0.8)
    ax4.set_xlim(0, 1000/f_nominal)
    
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fig1, fig2, fig3, fig4 = create_all_figures()
    fig_combined = create_combined_figure()
    # fig1.savefig('figure1_amplitude_variation.png', dpi=300, bbox_inches='tight')
    # fig2.savefig('figure2_frequency_variation.png', dpi=300, bbox_inches='tight')
    # fig3.savefig('figure3_offset_variation.png', dpi=300, bbox_inches='tight')
    # fig4.savefig('figure4_asymmetry_variation.png', dpi=300, bbox_inches='tight')
    fig_combined.savefig('fig_combined_flapping_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()