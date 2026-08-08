#!/usr/bin/env python3
"""
Plot 2D Ramachandran plot and 1D distributions for M, L, I residues only.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

def plot_ramachandran_mli(csv_path: str, output_path: str = None):
    """
    Plot 2D Ramachandran plot and 1D distributions for M (MET), L (LEU), I (ILE) residues only.
    
    Args:
        csv_path: Path to dihedral_angles_pos20.csv
        output_path: Path to save the plot (default: ramachandran_mli.png)
    """
    csv_file = Path(csv_path)
    if output_path is None:
        output_path = csv_file.parent / "ramachandran_mli.png"
    else:
        output_path = Path(output_path)
    
    # Map 1-letter to 3-letter codes
    aa_1_to_3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    # Target residues: M (MET), L (LEU), I (ILE)
    target_residues = {'M', 'L', 'I', 'MET', 'LEU', 'ILE'}
    
    # Also collect Y, R, F for reference lines
    ref_data = {'Y': [], 'R': [], 'F': []}  # Store (phi, psi) tuples
    
    # Read data - only for M, L, I (and collect Y, R, F for reference)
    phi_values = []
    psi_values = []
    amino_acids = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phi_str = row.get('Phi', '').strip()
            psi_str = row.get('Psi', '').strip()
            aa = row.get('Amino_Acid', '').strip().upper()
            found_aa = row.get('Found_Amino_Acid', '').strip().upper()
            
            # Check if residue is M, L, or I (using either Amino_Acid or Found_Amino_Acid)
            is_target = False
            if aa in target_residues or found_aa in target_residues:
                is_target = True
            elif len(aa) == 1 and aa in aa_1_to_3:
                three_letter = aa_1_to_3[aa]
                if three_letter in target_residues:
                    is_target = True
            elif len(found_aa) == 3 and found_aa in target_residues:
                is_target = True
            
            # Check if it's Y, R, or F for reference lines
            is_ref = False
            ref_aa = None
            if aa in ['Y', 'R', 'F']:
                is_ref = True
                ref_aa = aa
            elif found_aa in ['TYR', 'ARG', 'PHE']:
                is_ref = True
                ref_aa = {'TYR': 'Y', 'ARG': 'R', 'PHE': 'F'}[found_aa]
            
            if is_target and phi_str and psi_str and phi_str.lower() != 'none' and psi_str.lower() != 'none':
                try:
                    phi = float(phi_str)
                    psi = float(psi_str)
                    phi_values.append(phi)
                    psi_values.append(psi)
                    # Normalize to 1-letter code for display
                    if len(aa) == 1:
                        amino_acids.append(aa)
                    elif aa in ['MET', 'LEU', 'ILE']:
                        amino_acids.append({'MET': 'M', 'LEU': 'L', 'ILE': 'I'}[aa])
                    elif found_aa in ['MET', 'LEU', 'ILE']:
                        amino_acids.append({'MET': 'M', 'LEU': 'L', 'ILE': 'I'}[found_aa])
                    else:
                        amino_acids.append(aa[0] if aa else '?')
                except ValueError:
                    continue
            
            # Collect Y, R, F data for reference lines
            if is_ref and phi_str and psi_str and phi_str.lower() != 'none' and psi_str.lower() != 'none':
                try:
                    phi = float(phi_str)
                    psi = float(psi_str)
                    ref_data[ref_aa].append((phi, psi))
                except ValueError:
                    continue
    
    if not phi_values:
        print("No valid dihedral angles found for M, L, I residues!")
        return
    
    # Count entries per residue
    counts = {
        'M': sum(1 for aa in amino_acids if aa == 'M'),
        'L': sum(1 for aa in amino_acids if aa == 'L'),
        'I': sum(1 for aa in amino_acids if aa == 'I')
    }
    
    print(f"Found {len(phi_values)} entries for M, L, I residues")
    print(f"  M (MET): {counts['M']}")
    print(f"  L (LEU): {counts['L']}")
    print(f"  I (ILE): {counts['I']}")
    
    # Color scheme for M, L, I
    aa_colors = {'M': '#1f77b4', 'L': '#ff7f0e', 'I': '#2ca02c'}  # Blue, Orange, Green
    aa_labels = {'M': 'MET', 'L': 'LEU', 'I': 'ILE'}
    
    # Set figure width (all subplots same width)
    fig_width = 8
    # Distribution height is 0.6 of width
    dist_height = fig_width * 0.6
    # Ramachandran is square, so same as width
    ramachandran_height = fig_width
    
    # Total figure height
    total_height = ramachandran_height + dist_height + dist_height + 0.5  # 0.5 for spacing
    
    # Create figure with 3 subplots stacked vertically
    # All subplots will have the same width (same column in gridspec)
    fig = plt.figure(figsize=(fig_width, total_height))
    gs = fig.add_gridspec(3, 1, hspace=0.4, height_ratios=[ramachandran_height, dist_height, dist_height], 
                          width_ratios=[1])  # Explicitly set same width for all
    
    # 1. 2D Ramachandran plot (square, larger)
    ax1 = fig.add_subplot(gs[0, 0])
    for aa in ['M', 'L', 'I']:
        aa_phi = [phi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa]
        aa_psi = [psi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa]
        if aa_phi:
            label = f"{aa_labels[aa]} (n={counts[aa]})"
            ax1.scatter(aa_phi, aa_psi, alpha=0.6, s=30, label=label, 
                       c=aa_colors[aa], edgecolors='black', linewidths=0.3)
    
    ax1.set_xlabel('Phi (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Psi (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Ramachandran Plot (MET, LEU, ILE)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.set_aspect('equal')  # Make it square
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 2. Phi distribution by residue type with KDE (no KDE in legend)
    ax2 = fig.add_subplot(gs[1, 0])
    x_phi = np.linspace(-180, 180, 200)
    # Define bins once for all residues (same bin edges)
    phi_bins = np.linspace(-180, 180, 37)  # 36 bins from -180 to 180
    for aa in ['M', 'L', 'I']:
        aa_phi = np.array([phi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa])
        if len(aa_phi) > 1:
            # Histogram with density=True, using the same bins for all
            ax2.hist(aa_phi, bins=phi_bins, alpha=0.3, density=True, 
                    label=f"{aa_labels[aa]} (n={counts[aa]})", 
                    color=aa_colors[aa], edgecolor='black', linewidth=0.3)
            # KDE fitting (no label, so it won't appear in legend)
            try:
                kde = gaussian_kde(aa_phi)
                kde_values = kde(x_phi)
                ax2.plot(x_phi, kde_values, color=aa_colors[aa], linewidth=2, 
                        linestyle='-', alpha=0.8)
            except:
                pass  # Skip KDE if it fails
    
    # Add reference lines for Y, R, F residues (all individual entries)
    ref_colors = {'Y': 'purple', 'R': 'red', 'F': 'brown'}
    # Get y_max after plotting
    y_max = ax2.get_ylim()[1]
    for ref_aa in ['Y', 'R', 'F']:
        if ref_data[ref_aa]:
            for phi_val, psi_val in ref_data[ref_aa]:
                ax2.axvline(x=phi_val, color=ref_colors[ref_aa], linestyle='--', linewidth=1.5, alpha=0.7)
                ax2.text(phi_val, y_max * 0.95, ref_aa, color=ref_colors[ref_aa], 
                        fontsize=10, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=ref_colors[ref_aa]))
    
    ax2.set_xlabel('Phi (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax2.set_title('Phi Distribution by Residue', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 3. Psi distribution by residue type with KDE (no KDE in legend)
    ax3 = fig.add_subplot(gs[2, 0])
    x_psi = np.linspace(-180, 180, 200)
    # Define bins once for all residues (same bin edges)
    psi_bins = np.linspace(-180, 180, 37)  # 36 bins from -180 to 180
    for aa in ['M', 'L', 'I']:
        aa_psi = np.array([psi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa])
        if len(aa_psi) > 1:
            # Histogram with density=True, using the same bins for all
            ax3.hist(aa_psi, bins=psi_bins, alpha=0.3, density=True, 
                    label=f"{aa_labels[aa]} (n={counts[aa]})", 
                    color=aa_colors[aa], edgecolor='black', linewidth=0.3)
            # KDE fitting (no label, so it won't appear in legend)
            try:
                kde = gaussian_kde(aa_psi)
                kde_values = kde(x_psi)
                ax3.plot(x_psi, kde_values, color=aa_colors[aa], linewidth=2, 
                        linestyle='-', alpha=0.8)
            except:
                pass  # Skip KDE if it fails
    
    # Add reference lines for Y, R, F residues (all individual entries)
    # Get y_max after plotting
    y_max = ax3.get_ylim()[1]
    for ref_aa in ['Y', 'R', 'F']:
        if ref_data[ref_aa]:
            for phi_val, psi_val in ref_data[ref_aa]:
                ax3.axvline(x=psi_val, color=ref_colors[ref_aa], linestyle='--', linewidth=1.5, alpha=0.7)
                ax3.text(psi_val, y_max * 0.95, ref_aa, color=ref_colors[ref_aa], 
                        fontsize=10, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=ref_colors[ref_aa]))
    
    ax3.set_xlabel('Psi (degrees)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax3.set_title('Psi Distribution by Residue', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.legend(loc='upper left', fontsize=9)  # Moved to left
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.suptitle('Ramachandran Analysis: MET, LEU, ILE Only', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot Ramachandran plot for M, L, I residues only")
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
        help="Output plot file (default: ramachandran_mli.png)"
    )
    
    args = parser.parse_args()
    
    if args.input is None:
        csv_path = Path(__file__).parent / "data" / "dihedral_angles_pos20.csv"
    else:
        csv_path = Path(args.input)
    
    if not csv_path.exists():
        print(f"Error: Input file not found: {csv_path}")
        exit(1)
    
    plot_ramachandran_mli(str(csv_path), args.output)
