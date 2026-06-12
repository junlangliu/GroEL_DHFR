#!/usr/bin/env python3
"""
Analyze the distribution of MET (M) psi values and test if it can be modeled by 2 Gaussians.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def fit_two_gaussians_constrained(data, initial_means, mean_range=30.0):
    """
    Fit a 2-component Gaussian mixture model with means constrained to regions around initial values.
    
    Args:
        data: array of psi values
        initial_means: initial means for the two Gaussians [mu1, mu2] (from highest bins)
        mean_range: allowed range around initial means (degrees)
    
    Returns:
        weights: [w1, w2] - mixing weights
        means: [mu1, mu2] - optimized means (constrained to regions)
        stds: [sigma1, sigma2] - fitted standard deviations
        log_likelihood: log-likelihood of the fitted model
    """
    data = np.array(data)
    mu1_init, mu2_init = initial_means
    
    def negative_log_likelihood(params):
        """Negative log-likelihood to minimize"""
        w1, mu1, mu2, sigma1, sigma2 = params
        w2 = 1.0 - w1  # Ensure weights sum to 1
        
        if w1 < 0 or w1 > 1 or sigma1 <= 0 or sigma2 <= 0:
            return 1e10  # Penalty for invalid parameters
        
        # Constrain means to be within range of initial values
        if abs(mu1 - mu1_init) > mean_range or abs(mu2 - mu2_init) > mean_range:
            return 1e10  # Penalty for means outside allowed range
        
        # Calculate log-likelihood
        log_prob1 = norm.logpdf(data, mu1, sigma1) + np.log(w1)
        log_prob2 = norm.logpdf(data, mu2, sigma2) + np.log(w2)
        
        # Log-sum-exp trick for numerical stability
        max_log = np.maximum(log_prob1, log_prob2)
        log_likelihood = np.sum(max_log + np.log(np.exp(log_prob1 - max_log) + np.exp(log_prob2 - max_log)))
        
        return -log_likelihood  # Negative because we're minimizing
    
    # Initial guess: equal weights, initial means, reasonable stds
    initial_params = [0.5, mu1_init, mu2_init, np.std(data) * 0.8, np.std(data) * 0.8]
    
    # Bounds: w1 in [0, 1], means within range, sigmas > 0
    bounds = [
        (0.01, 0.99),  # w1
        (mu1_init - mean_range, mu1_init + mean_range),  # mu1
        (mu2_init - mean_range, mu2_init + mean_range),  # mu2
        (1.0, 200.0),  # sigma1
        (1.0, 200.0)   # sigma2
    ]
    
    # Optimize
    result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
    
    w1 = result.x[0]
    w2 = 1.0 - w1
    mu1 = result.x[1]
    mu2 = result.x[2]
    sigma1 = result.x[3]
    sigma2 = result.x[4]
    
    # Calculate final log-likelihood
    log_prob1 = norm.logpdf(data, mu1, sigma1) + np.log(w1)
    log_prob2 = norm.logpdf(data, mu2, sigma2) + np.log(w2)
    max_log = np.maximum(log_prob1, log_prob2)
    log_likelihood = np.sum(max_log + np.log(np.exp(log_prob1 - max_log) + np.exp(log_prob2 - max_log)))
    
    return [w1, w2], [mu1, mu2], [sigma1, sigma2], log_likelihood

def plot_met_psi_analysis(csv_path: str, output_path: str = None):
    """
    Analyze MET psi distribution and test 2-Gaussian model.
    
    Args:
        csv_path: Path to dihedral_angles_pos20.csv
        output_path: Path to save the plot (default: met_psi_analysis.png)
    """
    csv_file = Path(csv_path)
    if output_path is None:
        output_path = csv_file.parent / "met_psi_analysis.png"
    else:
        output_path = Path(output_path)
    
    # Map 1-letter to 3-letter codes
    aa_1_to_3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    # Read MET psi values
    met_psi_values = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psi_str = row.get('Psi', '').strip()
            aa = row.get('Amino_Acid', '').strip().upper()
            found_aa = row.get('Found_Amino_Acid', '').strip().upper()
            
            # Check if it's MET
            is_met = False
            if aa == 'M' or found_aa == 'MET':
                is_met = True
            elif len(aa) == 1 and aa in aa_1_to_3 and aa_1_to_3[aa] == 'MET':
                is_met = True
            
            if is_met and psi_str and psi_str.lower() != 'none':
                try:
                    psi = float(psi_str)
                    met_psi_values.append(psi)
                except ValueError:
                    continue
    
    if len(met_psi_values) < 10:
        print(f"Not enough MET entries ({len(met_psi_values)}) for analysis!")
        return
    
    met_psi_array = np.array(met_psi_values)
    
    # Filter out psi < 50 for fitting
    met_psi_filtered = met_psi_array[met_psi_array >= 50]
    n_filtered = len(met_psi_array) - len(met_psi_filtered)
    
    if len(met_psi_filtered) < 10:
        print(f"Not enough MET entries after filtering (psi >= 50): {len(met_psi_filtered)}")
        return
    
    print(f"Filtered out {n_filtered} entries with psi < 50°")
    print(f"Using {len(met_psi_filtered)} entries for fitting (out of {len(met_psi_array)} total)")
    
    print(f"\n{'='*60}")
    print("MET Psi Distribution Analysis")
    print(f"{'='*60}")
    print(f"Total MET entries: {len(met_psi_array)}")
    print(f"Mean: {np.mean(met_psi_array):.2f}°")
    print(f"Median: {np.median(met_psi_array):.2f}°")
    print(f"Std Dev: {np.std(met_psi_array):.2f}°")
    print(f"Min: {np.min(met_psi_array):.2f}°")
    print(f"Max: {np.max(met_psi_array):.2f}°")
    
    # Find the two highest bins to use as fixed centers (using filtered data)
    # Use more bins for higher resolution
    n_bins = 50  # Increased from default for higher resolution
    hist_counts, bin_edges = np.histogram(met_psi_filtered, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find indices of two highest bars
    top_two_indices = np.argsort(hist_counts)[-2:]
    initial_means = [bin_centers[top_two_indices[0]], bin_centers[top_two_indices[1]]]
    
    print(f"\n{'─'*60}")
    print("Finding Initial Centers from Histogram...")
    print(f"  Using {n_bins} bins for higher resolution")
    print(f"  Highest bin center: {initial_means[0]:.2f}° (count: {hist_counts[top_two_indices[0]]})")
    print(f"  2nd highest bin center: {initial_means[1]:.2f}° (count: {hist_counts[top_two_indices[1]]})")
    
    # Fit 2-Gaussian mixture model with means constrained to regions around highest bins
    mean_range = 30.0  # Allow means to vary ±30 degrees from initial values
    print(f"\n{'─'*60}")
    print(f"Fitting 2-Gaussian Mixture Model (means constrained to ±{mean_range}° around highest bins, psi >= 50 only)...")
    weights, means, stds, log_likelihood = fit_two_gaussians_constrained(met_psi_filtered, initial_means, mean_range=mean_range)
    
    print(f"\nGaussian 1:")
    print(f"  Weight: {weights[0]:.3f} ({weights[0]*100:.1f}%)")
    print(f"  Mean: {means[0]:.2f}° (optimized, initial: {initial_means[0]:.2f}°)")
    print(f"  Std Dev: {stds[0]:.2f}°")
    
    print(f"\nGaussian 2:")
    print(f"  Weight: {weights[1]:.3f} ({weights[1]*100:.1f}%)")
    print(f"  Mean: {means[1]:.2f}° (optimized, initial: {initial_means[1]:.2f}°)")
    print(f"  Std Dev: {stds[1]:.2f}°")
    
    # Calculate AIC and BIC (using filtered sample size)
    # Parameters: 1 weight + 2 means + 2 stds = 5 parameters
    n_params = 5
    n_samples = len(met_psi_filtered)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    
    print(f"\nModel Statistics:")
    print(f"  AIC: {aic:.2f}")
    print(f"  BIC: {bic:.2f}")
    print(f"  Log-likelihood: {log_likelihood:.2f}")
    print(f"  Number of parameters: {n_params} (1 weight + 2 means + 2 stds)")
    
    # Find intersection point of the two Gaussians
    # Solve: weights[0] * norm.pdf(x, means[0], stds[0]) = weights[1] * norm.pdf(x, means[1], stds[1])
    # This simplifies to finding where the log probabilities are equal
    from scipy.optimize import fsolve
    
    def gaussian_diff(x):
        """Difference between the two weighted Gaussians"""
        g1 = weights[0] * norm.pdf(x, means[0], stds[0])
        g2 = weights[1] * norm.pdf(x, means[1], stds[1])
        return g1 - g2
    
    # Find intersection point(s) - try around the midpoint of the two means
    intersection_points = []
    x_search = np.linspace(min(means[0], means[1]) - 50, max(means[0], means[1]) + 50, 1000)
    diff_values = gaussian_diff(x_search)
    
    # Find sign changes (where curves cross)
    for i in range(len(diff_values) - 1):
        if diff_values[i] * diff_values[i+1] < 0:  # Sign change
            # Refine intersection point
            x0 = x_search[i]
            try:
                intersection = fsolve(gaussian_diff, x0)[0]
                if min(means[0], means[1]) - 50 <= intersection <= max(means[0], means[1]) + 50:
                    intersection_points.append(intersection)
            except:
                pass
    
    # Remove duplicates (within tolerance)
    intersection_points = np.unique(np.round(intersection_points, 2))
    
    print(f"\n{'─'*60}")
    print("Gaussian Intersection Point(s):")
    if len(intersection_points) > 0:
        # Filter out intersections where both values are essentially zero
        meaningful_intersections = []
        for intersect in intersection_points:
            g1_val = weights[0] * norm.pdf(intersect, means[0], stds[0])
            g2_val = weights[1] * norm.pdf(intersect, means[1], stds[1])
            # Only consider intersections where at least one Gaussian has meaningful density (> 0.001)
            if g1_val > 0.001 or g2_val > 0.001:
                meaningful_intersections.append((intersect, g1_val, g2_val))
        
        if len(meaningful_intersections) > 0:
            for i, (intersect, g1_val, g2_val) in enumerate(meaningful_intersections):
                print(f"  Intersection {i+1}: psi = {intersect:.2f}°")
                print(f"    Gaussian 1 value: {g1_val:.6f}")
                print(f"    Gaussian 2 value: {g2_val:.6f}")
        else:
            print("  No meaningful intersection found (all intersections at near-zero density)")
    else:
        print("  No intersection found in the search range")
    
    # Create visualization
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot range: focus on psi >= 50, but show all data
    x_plot = np.linspace(max(met_psi_array.min() - 20, 0), met_psi_array.max() + 20, 500)
    
    # Histogram with higher resolution (more bins) - show all data
    n, bins, patches = ax1.hist(met_psi_array, bins=n_bins, density=True, alpha=0.5, 
                                color='steelblue', edgecolor='black', linewidth=0.5,
                                label=f'MET Psi (n={len(met_psi_array)}, all data)')
    
    # 2-Gaussian mixture - make them more visible
    gauss1 = weights[0] * norm.pdf(x_plot, means[0], stds[0])
    gauss2 = weights[1] * norm.pdf(x_plot, means[1], stds[1])
    mixture = gauss1 + gauss2
    
    # Plot individual Gaussians with thicker, more visible lines
    ax1.plot(x_plot, gauss1, color='green', linestyle='-', linewidth=3, alpha=0.8, 
            label=f'Gaussian 1 (w={weights[0]:.2f}, μ={means[0]:.1f}°, σ={stds[0]:.1f}°)')
    ax1.plot(x_plot, gauss2, color='orange', linestyle='-', linewidth=3, alpha=0.8,
            label=f'Gaussian 2 (w={weights[1]:.2f}, μ={means[1]:.1f}°, σ={stds[1]:.1f}°)')
    ax1.plot(x_plot, mixture, color='purple', linewidth=3, alpha=0.9,
            label='2-Gaussian Mixture (means optimized in region of highest bins)')
    
    
    # KDE for comparison (using filtered data)
    try:
        kde = gaussian_kde(met_psi_filtered)
        kde_values = kde(x_plot)
        ax1.plot(x_plot, kde_values, color='brown', linewidth=2, alpha=0.7, linestyle='-.',
                label='KDE (non-parametric, psi >= 50)')
    except:
        pass
    
    ax1.set_xlabel('Psi (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('MET Psi Distribution (fitted using psi >= 50)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)  # Moved to upper left
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*60}\n")
    plt.close()
    
    return {
        'n_samples': len(met_psi_array),
        'mean': np.mean(met_psi_array),
        'std': np.std(met_psi_array),
        'gmm_weights': weights,
        'gmm_means': means,
        'gmm_stds': stds,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MET psi distribution with 2-Gaussian model")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input CSV file (default: dihedral_angles_pos20.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output plot file (default: met_psi_analysis.png)"
    )
    
    args = parser.parse_args()
    
    if args.input is None:
        csv_path = Path(__file__).parent / "dihedral_angles_pos20.csv"
    else:
        csv_path = Path(args.input)
    
    if not csv_path.exists():
        print(f"Error: Input file not found: {csv_path}")
        exit(1)
    
    plot_met_psi_analysis(str(csv_path), args.output)

