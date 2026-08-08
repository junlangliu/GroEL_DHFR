#!/usr/bin/env python3
"""
Plot psi distribution for M, I, L amino acids, categorized by ligand pairs:
1) NAP/NDP-FOL
2) NAP/NDP-MTX/TOP
3) All other entries

One subplot per amino acid.
"""

import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu

# Non-DHFR ligands to exclude
WATER_AND_IONS = {
    'HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN',
    'CU', 'NI', 'CO', 'CD', 'HG', 'PB', 'AS', 'SE', 'BR', 'IOD', 'IODIDE',
    'SULFATE', 'PHOSPHATE', 'ACETATE', 'EDTA', 'TRIS', 'HEPES', 'MES', 'MOPS',
    'CB3', 'UMP', 'EOH', 'GLY'
}


def filter_ligands(ligands_str: str) -> list:
    """
    Filter out common ligands (water, ions, buffers, non-DHFR ligands) from ligand string.
    """
    if not ligands_str or not ligands_str.strip():
        return []
    
    ligands = [lig.strip().upper() for lig in ligands_str.split(";") if lig.strip()]
    filtered = [lig for lig in ligands if lig not in WATER_AND_IONS]
    
    return filtered


def has_exactly_cofactor(cofactor_str: str, target_cofactors: set) -> bool:
    """Check if cofactor string contains exactly one of the target cofactors."""
    cofactors = filter_ligands(cofactor_str)
    if len(cofactors) != 1:
        return False
    return cofactors[0] in target_cofactors


def has_exactly_substrate(substrate_str: str, target_substrates: set) -> bool:
    """Check if substrate string contains exactly one of the target substrates."""
    substrates = filter_ligands(substrate_str)
    if len(substrates) != 1:
        return False
    return substrates[0] in target_substrates


def categorize_entry(cofactor_str: str, substrate_str: str) -> str:
    """
    Categorize entry into:
    - 'NAP_NDP_FOL': exactly NAP/NDP and FOL
    - 'NAP_NDP_MTX_TOP': exactly NAP/NDP and MTX/TOP
    - 'other': all other entries
    
    Returns:
        Category string
    """
    has_nap_ndp = has_exactly_cofactor(cofactor_str, {'NAP', 'NDP'})
    
    if has_nap_ndp:
        if has_exactly_substrate(substrate_str, {'FOL'}):
            return 'NAP_NDP_FOL'
        elif has_exactly_substrate(substrate_str, {'MTX', 'TOP'}):
            return 'NAP_NDP_MTX_TOP'
    
    return 'other'


def plot_psi_by_ligand_category(
    classification_csv: str,
    dihedral_csv: str,
    output_plot: str = None,
    psi_threshold: float = 128.39
):
    """
    Plot psi distribution for M, I, L, categorized by ligand pairs.
    
    Args:
        classification_csv: Path to ligand_classification.csv
        dihedral_csv: Path to dihedral_angles_pos20.csv
        output_plot: Path to output plot file
        psi_threshold: Psi threshold for M classification
    """
    class_path = Path(classification_csv)
    dihedral_path = Path(dihedral_csv)
    
    if output_plot is None:
        output_plot = class_path.parent / "psi_distribution_by_ligand_category.png"
    else:
        output_plot = Path(output_plot)
    
    # Load psi values
    psi_values = {}
    with open(dihedral_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            aa = row.get('Amino_Acid', '').strip().upper()
            psi_str = row.get('Psi', '').strip()
            status = row.get('Status', '').strip()
            
            if status == 'Success' and aa in {'M', 'L', 'I'} and psi_str and psi_str.lower() != 'none':
                try:
                    psi = float(psi_str)
                    psi_values[pdb_id] = psi
                except (ValueError, TypeError):
                    pass
    
    # Organize entries by amino acid and category
    amino_acids = ['M', 'L', 'I']  # L in the middle
    categories = ['NAP_NDP_FOL', 'NAP_NDP_MTX_TOP', 'other']
    
    data = {aa: {cat: [] for cat in categories} for aa in amino_acids}
    
    with open(class_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            pos20_aa = row.get('Pos20_Amino_Acid', '').strip().upper()
            cofactor_str = row.get('Cofactor', '').strip()
            substrate_str = row.get('Substrate_Antibiotic', '').strip()
            
            # Determine amino acid group
            aa_group = None
            if pos20_aa == 'M':
                psi = psi_values.get(pdb_id)
                if psi is not None:
                    aa_group = 'M'  # Include all M entries (don't split by psi threshold)
            elif pos20_aa == 'L':
                aa_group = 'L'
            elif pos20_aa == 'I':
                aa_group = 'I'
            # Only consider entries with psi >= 50 (no need to include psi < 50)
            if aa_group and pdb_id in psi_values:
                psi_val = psi_values[pdb_id]
                if psi_val >= 50:
                    category = categorize_entry(cofactor_str, substrate_str)
                    data[aa_group][category].append(psi_val)
    
    # Create plot - 3 subplots in one row (one per amino acid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for categories
    category_colors = {
        'NAP_NDP_FOL': 'steelblue',
        'NAP_NDP_MTX_TOP': 'orange',
        'other': 'gray'
    }
    
    category_labels = {
        'NAP_NDP_FOL': 'NAP/NDP-FOL',
        'NAP_NDP_MTX_TOP': 'NAP/NDP-MTX/TOP',
        'other': 'Other'
    }
    
    # Calculate p-values for all pairwise comparisons
    p_values_dict = {}
    for aa in amino_acids:
        p_values_dict[aa] = {}
        psi_data = {cat: data[aa][cat] for cat in categories if len(data[aa][cat]) > 0}
        cat_list = list(psi_data.keys())
        
        for i in range(len(cat_list)):
            for j in range(i + 1, len(cat_list)):
                cat1 = cat_list[i]
                cat2 = cat_list[j]
                
                vals1 = psi_data[cat1]
                vals2 = psi_data[cat2]
                
                if len(vals1) > 1 and len(vals2) > 1:
                    try:
                        u_stat, p_value = mannwhitneyu(vals1, vals2, alternative='two-sided')
                        pair_key = tuple(sorted([cat1, cat2]))
                        p_values_dict[aa][pair_key] = p_value
                    except:
                        pass
    
    for idx, aa in enumerate(amino_acids):
        ax = axes[idx]
        
        # Prepare data for violin plot - always include all three categories
        plot_data = []
        plot_labels = []
        plot_colors = []
        
        for cat in categories:
            psi_vals = data[aa][cat]
            if psi_vals and len(psi_vals) > 0:
                plot_data.append(psi_vals)
                plot_labels.append(f'{category_labels[cat]}\n(n={len(psi_vals)})')
            else:
                # Add empty list to maintain position
                plot_data.append([])
                plot_labels.append(f'{category_labels[cat]}\n(n=0)')
            plot_colors.append(category_colors[cat])
        
        # Always set x-axis labels for all three categories FIRST
        # This ensures all positions are shown even if empty
        ax.set_xticks([0, 1, 2])  # Explicitly set all three positions
        ax.set_xticklabels(plot_labels, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, 2.5)  # Explicitly set x-axis limits to show all three positions
        
        # Create violin plot - only plot non-empty categories
        plot_data_for_violin = []
        positions_for_violin = []
        
        for i, d in enumerate(plot_data):
            if len(d) > 0:
                plot_data_for_violin.append(d)
                positions_for_violin.append(i)
        
        if plot_data_for_violin:
            # Draw violin plots for psi >= 50
            parts = ax.violinplot(
                plot_data_for_violin,
                positions=positions_for_violin,
                widths=0.6,
                showmeans=False,
                showmedians=False
            )
            
            # Customize violin plot colors (lighter to emphasize summary stats)
            for i, pc in enumerate(parts['bodies']):
                # Find the original position to get the right color
                orig_pos = positions_for_violin[i]
                pc.set_facecolor(plot_colors[orig_pos])
                pc.set_alpha(0.4)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Customize other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1.0)
        
        # Overlay median and 25–75 percentile as summary for each category (psi >= 50)
        for i, cat in enumerate(categories):
            psi_vals = data[aa][cat]
            if not psi_vals:
                continue
            q25 = np.percentile(psi_vals, 25)
            median = np.percentile(psi_vals, 50)
            q75 = np.percentile(psi_vals, 75)
            
            # Horizontal lines for 25, 50 (median), and 75 percentiles
            ax.hlines(
                [q25, median, q75],
                i - 0.18,
                i + 0.18,
                colors='black',
                linewidth=2
            )
        
        # Add p-value annotations on the plot around psi ≈ 35
        y_label = 35  # Base y-position for all labels
        
        # Annotate pairwise comparisons
        if aa in p_values_dict:
            comparisons = []
            for (cat1, cat2), p_val in p_values_dict[aa].items():
                # Find positions of these categories
                pos1 = categories.index(cat1)
                pos2 = categories.index(cat2)
                
                # Format p-value - no stars, just "ns" or p-value
                if p_val >= 0.05:
                    p_str = "ns"
                elif p_val < 0.001:
                    p_str = f"p<0.001"
                else:
                    p_str = f"p={p_val:.3f}"
                
                comparisons.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'p_str': p_str,
                    'p_val': p_val,
                    'span': abs(pos2 - pos1)  # Distance between positions
                })
            
            # Sort by span (wider comparisons first) to minimize overlap
            comparisons.sort(key=lambda x: (-x['span'], x['pos1']))
            
            # Draw brackets and p-values
            # Use different positive y-offsets to prevent overlap (stacked upwards)
            y_offsets = [0, 10, 20]  # Different heights for different comparisons
            
            for idx, comp in enumerate(comparisons):
                pos1 = comp['pos1']
                pos2 = comp['pos2']
                p_str = comp['p_str']
                
                # Use offset to prevent overlap
                bracket_y = y_label + y_offsets[idx % len(y_offsets)]
                
                # Draw bracket pointing upward
                ax.plot([pos1, pos1], [bracket_y, bracket_y + 3], 
                       'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [bracket_y, bracket_y + 3], 
                       'k-', linewidth=1.5)
                ax.plot([pos1, pos2], [bracket_y, bracket_y], 
                       'k-', linewidth=1.5)
                
                # Add p-value text below the bracket (closer to line)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, bracket_y - 2, p_str,
                       ha='center', va='top', fontsize=9, fontweight='bold')
        
        # Add threshold line
        ax.axhline(y=psi_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({psi_threshold}°)', alpha=0.7)
        
        ax.set_xlabel('Ligand Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Psi (degrees)', fontsize=12, fontweight='bold')
        ax.set_title(f'{aa}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        # Show psi range from 0 to 180
        ax.set_ylim(0, 180)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_plot}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for aa in amino_acids:
        print(f"\n{aa}:")
        for cat in categories:
            count = len(data[aa][cat])
            if count > 0:
                psi_vals = data[aa][cat]
                mean_psi = np.mean(psi_vals)
                q25 = np.percentile(psi_vals, 25)
                q75 = np.percentile(psi_vals, 75)
                print(f"  {category_labels[cat]}: {count} entries")
                print(f"    Mean: {mean_psi:.1f}°, Q25: {q25:.1f}°, Q75: {q75:.1f}°")
    print("=" * 60)
    
    # Perform pairwise Mann-Whitney U tests
    print("\n" + "=" * 60)
    print("Pairwise Mann-Whitney U test Results:")
    print("=" * 60)
    
    for aa in amino_acids:
        print(f"\n{aa}:")
        psi_data = {cat: data[aa][cat] for cat in categories if len(data[aa][cat]) > 0}
        
        # Compare all pairs
        cat_list = list(psi_data.keys())
        for i in range(len(cat_list)):
            for j in range(i + 1, len(cat_list)):
                cat1 = cat_list[i]
                cat2 = cat_list[j]
                
                vals1 = psi_data[cat1]
                vals2 = psi_data[cat2]
                
                if len(vals1) > 1 and len(vals2) > 1:
                    u_stat, p_value = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    
                    mean1 = np.mean(vals1)
                    mean2 = np.mean(vals2)
                    median1 = np.median(vals1)
                    median2 = np.median(vals2)
                    
                    # Determine significance
                    if p_value < 0.001:
                        sig = "***"
                    elif p_value < 0.01:
                        sig = "**"
                    elif p_value < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    print(f"  {category_labels[cat1]} vs {category_labels[cat2]}:")
                    print(f"    Mean: {mean1:.1f}° vs {mean2:.1f}° (Δ={abs(mean1-mean2):.1f}°)")
                    print(f"    Median: {median1:.1f}° vs {median2:.1f}° (Δ={abs(median1-median2):.1f}°)")
                    print(f"    U-statistic: {u_stat:.1f}, p-value: {p_value:.4f} {sig}")
                else:
                    print(f"  {category_labels[cat1]} vs {category_labels[cat2]}:")
                    print(f"    Insufficient data (n1={len(vals1)}, n2={len(vals2)})")
    
    print("=" * 60)
    print("Mann-Whitney U test results")
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot psi distribution for M, I, L by ligand category"
    )
    parser.add_argument(
        "-c", "--classification",
        type=str,
        default=None,
        help="Path to ligand_classification.csv"
    )
    parser.add_argument(
        "-d", "--dihedral",
        type=str,
        default=None,
        help="Path to dihedral_angles_pos20.csv"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=128.39,
        help="Psi threshold for M classification (default: 128.39)"
    )
    parser.add_argument(
        "-p", "--plot",
        type=str,
        default=None,
        help="Output plot file (default: psi_distribution_by_ligand_category.png)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.classification is None:
        args.classification = Path(__file__).resolve().parent.parent / "data" / "ligand_classification.csv"
    if args.dihedral is None:
        args.dihedral = Path(__file__).resolve().parent.parent / "data" / "dihedral_angles_pos20.csv"
    
    plot_psi_by_ligand_category(
        str(args.classification),
        str(args.dihedral),
        args.plot,
        args.threshold
    )

