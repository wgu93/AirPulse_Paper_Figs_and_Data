import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

def setup_plotting_style():
    plt.rcParams.update({
        # Font settings
        'font.family': 'Arial',
        'font.size': 8,
        'font.weight': 'normal',
        'mathtext.fontset': 'stix',
        'mathtext.default': 'regular',
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        # 'axes.labelweight': 'bold',
        # 'axes.titleweight': 'bold',
        
        # Lines and markers
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'errorbar.capsize': 3,
        
        # Legend
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        
        # Figure
        'figure.dpi': 600,
        'figure.figsize': (7.2, 2.5),  
        'savefig.dpi': 600,
        'savefig.format': 'tiff',
        'savefig.bbox': 'tight',
        
        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

def extract_wing_contour(image_path):
    """Extract wing contour from image"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to extract the wing
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the wing
    wing_contour = max(contours, key=cv2.contourArea)
    
    # Convert to a list of (x, y) points
    points = wing_contour.reshape(-1, 2)
    
    # Find the leftmost point (wing root)
    leftmost_idx = np.argmin(points[:, 0])
    wing_root = points[leftmost_idx]
    
    # Adjust coordinates so wing root is at (0, 0)
    adjusted_points = points - wing_root
    
    return adjusted_points, wing_contour, img, wing_root

def chord_length_parameterization(points):
    """
    Assign parameter values by chord length
    
    t_k = (∑_{m=1}^{k-1} ‖D_{m+1} - D_m‖) / (∑_{m=1}^N ‖D_{m+1} - D_m‖)
    t_1 = 0, t_{N+1} = 1
    """
    # Calculate chord lengths between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Cumulative chord lengths
    cumulative_lengths = np.cumsum(distances)
    total_length = cumulative_lengths[-1]
    
    # Parameter values (t_1 = 0, t_N = 1 for closed curve)
    t = np.zeros(len(points))
    t[1:] = cumulative_lengths / total_length
    
    return t

def periodic_b_spline_fit(points, alpha=0.5, num_resample=1000):
    """
    Fit periodic cubic B-spline to digitized wing outline
    """
    # Ensure points are cyclically ordered and closed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    N = len(points)
    t = chord_length_parameterization(points)
    s = alpha * N
    
    # Fit periodic cubic B-spline
    tck, u = splprep([points[:, 0], points[:, 1]], u=t, s=s, per=True, k=3)
    control_points = np.array(tck[1]).T
    
    # Resample at uniformly spaced parameters
    u_resampled = np.linspace(0, 1, num_resample)
    resampled_curve = np.array(splev(u_resampled, tck)).T
    
    # Calculate RMS error
    tree = KDTree(resampled_curve)
    distances, _ = tree.query(points)
    rms_error = np.sqrt(np.mean(distances**2))
    
    return tck, resampled_curve, control_points, rms_error, distances

def plot_spline_results(original_img, original_contour, points, wing_root, 
                       resampled_curve, control_points, rms_error, distances):    
    fig = plt.figure(figsize=(9.2, 2.3)) 
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Panel A: Original image with contour
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    contour_points = original_contour.reshape(-1, 2)
    ax1.plot(contour_points[:, 0], contour_points[:, 1], 
             color='#2E8B57', linewidth=1.5, label='Extracted contour')
    ax1.plot(wing_root[0], wing_root[1], 'r*', markersize=8, 
             markeredgewidth=0.8, markeredgecolor='black', 
             label='Wing root')
    ax1.set_title('Wing Image Analysis', fontsize=9, fontweight='bold', pad=10)
    ax1.axis('off')
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='0.3')
    
    # Panel B: Spline fitting results
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(points[:, 0], -points[:, 1], 'x', markersize=4.5, alpha=0.5,
             label='Sampled points', color='#2E8B57', markeredgewidth=1)
    ax2.plot(resampled_curve[:, 0], -resampled_curve[:, 1], 
             color='#b200ed', linewidth=2, label='B-spline fit')
    ax2.plot(control_points[:, 0], -control_points[:, 1], 's--', 
             markersize=4, alpha=0.9, label='Control points', 
             color='#ff7f0e', markeredgewidth=0.5, linewidth=1)
    
    ax2.set_xlabel('X coordinate (pixels)', labelpad=2)
    ax2.set_ylabel('Y coordinate (pixels)', labelpad=2)
    ax2.set_title(f'B-Spline Parameterization\nRMS error: {rms_error:.3f} pixels', 
                  fontsize=9, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax2.axis('equal')
    ax2.legend(loc='upper left', framealpha=0.95, edgecolor='0.3')
    
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Panel C: Error analysis 
    ax3 = fig.add_subplot(gs[2])
    
    # Use step histogram 
    n, bins, patches = ax3.hist(distances, bins=30, alpha=0.7, 
                               color='#c44e52', edgecolor='black', 
                               linewidth=0.3, density=False)
    
    ax3.axvline(rms_error, color='darkred', linestyle='--', 
                linewidth=1.5, label=f'RMS = {rms_error:.3f} px')
    
    # Add statistical information
    max_error = np.max(distances)
    mean_error = np.mean(distances)
    ax3.axvline(mean_error, color='#c44e52', linestyle=':', 
                linewidth=1.5, label=f'Mean = {mean_error:.3f} px')
    
    ax3.set_xlabel('Fitting Error (pixels)', labelpad=2)
    ax3.set_ylabel('Frequency', labelpad=2)
    ax3.set_title('Error Distribution', fontsize=9, fontweight='bold', pad=10)
    ax3.legend(framealpha=0.95, edgecolor='0.3')
    ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    plt.tight_layout(pad=1.0)
    
    return fig

def save_spline_data(resampled_curve, control_points, rms_error, filename):
    """Save spline data to JSON file"""
    spline_data = {
        'resampled_curve': [[round(float(x), 9), round(float(y), 9)] 
                           for x, y in resampled_curve],
        'control_points': [[round(float(x), 9), round(float(y), 9)] 
                          for x, y in control_points],
        'rms_error': float(rms_error),
        'metadata': {
            'num_resampled_points': len(resampled_curve),
            'num_control_points': len(control_points),
            'spline_type': 'periodic_cubic_b_spline'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(spline_data, f, indent=2)

def print_publication_summary(points, control_points, rms_error, alpha):
    """Print summary statistics"""
    print("=" * 60)
    print("WING CONTOUR PARAMETERIZATION SUMMARY")
    print("=" * 60)
    print(f"Sample: Graphium agetes")
    print(f"Sampled points (N): {len(points):,}")
    print(f"Control points (n): {len(control_points):,}")
    print(f"Data reduction: {100*(1-len(control_points)/len(points)):.1f}%")
    print(f"RMS fitting error: {rms_error:.4f} pixels")
    print(f"Smoothing parameter (α): {alpha}")
    print(f"B-spline degree: 3 (cubic)")
    print(f"Parameterization: Chord-length")
    print("=" * 60)

def print_b_spline_equation(tck, control_points):
    """Print the mathematical representation of the B-spline"""
    print("\n" + "="*70)
    print("B-SPLINE MATHEMATICAL REPRESENTATION")
    print("="*70)
    
    knots, coeffs, degree = tck
    n_control = len(control_points)
    
    print(f"Spline curve: 𝐂(u) = ∑ᵢ Nᵢ,₃(u) 𝐏ᵢ,  u ∈ [0,1]")
    print(f"Degree: p = {degree} (cubic)")
    print(f"Number of control points: n = {n_control}")
    print(f"Control points: 𝐏ᵢ ∈ ℝ², i = 0,...,{n_control-1}")
    print(f"Basis functions: Nᵢ,₃(u) - cubic B-spline (Cox-de Boor recursion)")
    
    print(f"\nKnot vector U (periodic, length {len(knots)}):")
    print("U = [" + ", ".join([f"{k:.3f}" for k in knots[:8]]) + ", ... , " + 
          ", ".join([f"{k:.3f}" for k in knots[-8:]]) + "]")
    
    print(f"\nControl points 𝐏ᵢ (first 10):")
    print("i\tXᵢ\t\t\tYᵢ")
    print("-" * 55)
    for i, (x, y) in enumerate(control_points[:10]):
        print(f"{i}\t{x:12.6f}\t{y:12.6f}")
    
    if len(control_points) > 10:
        print(f"...\t...\t\t\t...")
        print(f"{len(control_points)-1}\t{control_points[-1][0]:12.6f}\t{control_points[-1][1]:12.6f}")
    
    print(f"\nBasis function recursion:")
    print("Nᵢ,₀(u) = 1 if uᵢ ≤ u < uᵢ₊₁, else 0")
    print("Nᵢ,ₚ(u) = (u - uᵢ)/(uᵢ₊ₚ - uᵢ) Nᵢ,ₚ₋₁(u) + (uᵢ₊ₚ₊₁ - u)/(uᵢ₊ₚ₊₁ - uᵢ₊₁) Nᵢ₊₁,ₚ₋₁(u)")
    
    return knots, coeffs, degree

def export_b_spline_parameters(tck, control_points, filename):
    """Export B-spline parameters for external use"""
    knots, coeffs, degree = tck
    
    spline_params = {
        'degree': int(degree),
        'knot_vector': [float(k) for k in knots],
        'control_points': [[float(x), float(y)] for x, y in control_points],
        'coefficients': [[float(cx), float(cy)] for cx, cy in zip(coeffs[0], coeffs[1])],
        'equation': f"𝐂(u) = ∑ᵢ₌₀^{len(control_points)-1} Nᵢ,{degree}(u) 𝐏ᵢ, u ∈ [0,1]",
        'basis_definition': "Cox-de Boor recursion for cubic B-spline basis functions N_i,3(u)"
    }
    
    with open(filename, 'w') as f:
        json.dump(spline_params, f, indent=2)
    
    print(f"B-spline parameters exported to: {filename}")

if __name__ == "__main__":
    # Configuration
    image_path = "data/Graphium_agetes.png"
    alpha = 0.5
    num_resample = 1000
    setup_plotting_style()
    
    # Extract wing contour
    points, contour, original_img, wing_root = extract_wing_contour(image_path)
    
    if points is not None:
        # Fit periodic cubic B-spline
        tck, resampled_curve, control_points, rms_error, distances = periodic_b_spline_fit(
            points, alpha=alpha, num_resample=num_resample)
        
        # Print summary
        print_publication_summary(points, control_points, rms_error, alpha)
        
        # Plot results
        fig = plot_spline_results(original_img, contour, points, wing_root,
                                 resampled_curve, control_points, rms_error, distances)
        
        # Save figure
        plt.savefig('fig_wing_contour_parameterization.tiff', 
                   dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        plt.show()
        
        # Save spline data
        save_spline_data(resampled_curve, control_points, rms_error, 
                        'wing_contour_parameterization_spline_data.json')
        
        print("\nOutput files generated:")
        print("1. fig_wing_contour_parameterization.tiff")
        print("2. wing_contour_parameterization_spline_data.json")
    
    # After spline fitting, add:
    tck, resampled_curve, control_points, rms_error, distances = periodic_b_spline_fit(
        points, alpha=alpha, num_resample=num_resample)
    
    # Print B-spline equation and parameters
    knots, coeffs, degree = print_b_spline_equation(tck, control_points)
    
    # Export for external use
    export_b_spline_parameters(tck, control_points, 
                              'b_spline_parameters.json')
    
    # Additional: Generate LaTeX code for publication
    print(f"\nLaTeX-ready equation:")
    print(r"\mathbf{C}(u) = \sum_{i=0}^{" + f"{len(control_points)-1}" + 
          r"} N_{i,3}(u) \mathbf{P}_i,\quad u\in[0,1]")
    
    print(f"\nKnot vector LaTeX:")
    knot_str = ", ".join([f"{k:.3f}" for k in knots[:4]] + ["\dots"] + [f"{k:.3f}" for k in knots[-4:]])
    print(f"$\\mathbf{{U}} = [{knot_str}]$")