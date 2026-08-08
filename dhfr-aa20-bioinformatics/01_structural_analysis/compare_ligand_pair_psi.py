#!/usr/bin/env python3
"""
Compare psi distribution for entries with exactly specified cofactor-substrate/antibiotic pair.

This script can analyze any combination of cofactors and substrates/antibiotics
to compare their psi angle distributions across different amino acid groups.
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

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
    
    Args:
        ligands_str: Semicolon-separated ligand string
        
    Returns:
        List of filtered ligands
    """
    if not ligands_str or not ligands_str.strip():
        return []
    
    # Split by semicolon, strip whitespace, convert to uppercase
    ligands = [lig.strip().upper() for lig in ligands_str.split(";") if lig.strip()]
    
    # Filter out water and ions
    filtered = [lig for lig in ligands if lig not in WATER_AND_IONS]
    
    return filtered


def has_exactly_cofactor(cofactor_str: str, target_cofactors: set) -> bool:
    """
    Check if cofactor string contains exactly one of the target cofactors (and no other cofactors).
    
    Args:
        cofactor_str: Semicolon-separated cofactor string
        target_cofactors: Set of target cofactor codes (e.g., {'NAP', 'NDP'})
    
    Returns:
        True if exactly one target cofactor is present and no other cofactors
    """
    cofactors = filter_ligands(cofactor_str)
    
    # Must have exactly one cofactor, and it must be in target_cofactors
    if len(cofactors) != 1:
        return False
    
    return cofactors[0] in target_cofactors


def has_exactly_substrate(substrate_str: str, target_substrates: set) -> bool:
    """
    Check if substrate string contains exactly one of the target substrates (and no other substrates/antibiotics).
    
    Args:
        substrate_str: Semicolon-separated substrate string
        target_substrates: Set of target substrate codes (e.g., {'FOL'}, {'MTX', 'TOP'})
    
    Returns:
        True if exactly one target substrate is present and no other substrates
    """
    substrates = filter_ligands(substrate_str)
    
    # Must have exactly one substrate, and it must be in target_substrates
    if len(substrates) != 1:
        return False
    
    return substrates[0] in target_substrates


def compare_ligand_pair_psi_distribution(
    classification_csv: str,
    dihedral_csv: str,
    target_cofactors: set,
    target_substrates: set,
    output_plot: str = None,
    psi_threshold: float = 128.39,
    plot_title_suffix: str = None
):
    """
    Compare psi distribution for entries with exactly specified cofactor-substrate/antibiotic pair.
    
    Args:
        classification_csv: Path to ligand_classification.csv
        dihedral_csv: Path to dihedral_angles_pos20.csv (for psi values)
        target_cofactors: Set of target cofactor codes (e.g., {'NAP', 'NDP'})
        target_substrates: Set of target substrate/antibiotic codes (e.g., {'FOL'}, {'MTX', 'TOP'})
        output_plot: Path to output plot file
        psi_threshold: Psi threshold for M classification
        plot_title_suffix: Optional suffix for plot title (e.g., "NAP/NDP and FOL")
    """
    class_path = Path(classification_csv)
    dihedral_path = Path(dihedral_csv)
    
    # Generate strings for display and filename
    cofactor_str_display = '/'.join(sorted(target_cofactors))
    substrate_str_display = '/'.join(sorted(target_substrates))
    cofactor_str_file = '_'.join(sorted(target_cofactors))
    substrate_str_file = '_'.join(sorted(target_substrates))
    
    if output_plot is None:
        # Generate default filename based on ligands
        filename = f"psi_distribution_{cofactor_str_file}_{substrate_str_file}.png"
        output_plot = class_path.parent / filename
    else:
        output_plot = Path(output_plot)
    
    if plot_title_suffix is None:
        plot_title_suffix = f"{cofactor_str_display} and {substrate_str_display}"
    
    # Load psi values for M and L entries
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
    
    # Count entries by group
    groups = {
        'M_low': [],   # M with psi < threshold
        'M_high': [],  # M with psi > threshold
        'L': [],       # L entries
        'I': []        # I entries
    }
    
    with open(class_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            pos20_aa = row.get('Pos20_Amino_Acid', '').strip().upper()
            cofactor_str = row.get('Cofactor', '').strip()
            substrate_str = row.get('Substrate_Antibiotic', '').strip()
            
            # Determine group
            group = None
            if pos20_aa == 'M':
                psi = psi_values.get(pdb_id)
                if psi is not None:
                    if psi < psi_threshold:
                        group = 'M_low'
                    elif psi > psi_threshold:
                        group = 'M_high'
            elif pos20_aa == 'L':
                group = 'L'
            elif pos20_aa == 'I':
                group = 'I'
            
            if group:
                groups[group].append({
                    'pdb_id': pdb_id,
                    'cofactor_str': cofactor_str,
                    'substrate_str': substrate_str,
                    'psi': psi_values.get(pdb_id)  # Store psi value if available
                })
    
    # Count entries with exactly NAP/NDP and FOL
    group_names = ['M (psi<128.39)', 'M (psi>128.39)', 'L', 'I']
    group_keys = ['M_low', 'M_high', 'L', 'I']
    
    print("\n" + "=" * 60)
    print(f"Count of entries with exactly {cofactor_str_display} and {substrate_str_display}:")
    print("=" * 60)
    
    results = {}
    for gk, group_label in zip(group_keys, group_names):
        count = 0
        matching_pdb_ids = []
        matching_psi_values = []  # Store psi values for matching entries
        
        for entry in groups[gk]:
            has_cofactor = has_exactly_cofactor(entry['cofactor_str'], target_cofactors)
            has_substrate = has_exactly_substrate(entry['substrate_str'], target_substrates)
            
            if has_cofactor and has_substrate:
                count += 1
                matching_pdb_ids.append(entry['pdb_id'])
                if entry['psi'] is not None:
                    matching_psi_values.append(entry['psi'])
        
        # Also collect psi values for entries that do NOT have exactly the target pair
        rest_psi_values = []
        for entry in groups[gk]:
            has_cofactor = has_exactly_cofactor(entry['cofactor_str'], target_cofactors)
            has_substrate = has_exactly_substrate(entry['substrate_str'], target_substrates)
            
            if not (has_cofactor and has_substrate):
                if entry['psi'] is not None:
                    rest_psi_values.append(entry['psi'])
        
        results[gk] = {
            'count': count,
            'pdb_ids': matching_pdb_ids,
            'psi_values': matching_psi_values,
            'rest_psi_values': rest_psi_values,
            'total': len(groups[gk])
        }
        
        print(f"\n{group_label}:")
        print(f"  Total entries: {len(groups[gk])}")
        print(f"  Entries with exactly {cofactor_str_display} and {substrate_str_display}: {count}")
        if count > 0:
            print(f"  PDB IDs: {', '.join(sorted(matching_pdb_ids))}")
    
    print("\n" + "=" * 60)
    
    # Summary table
    print("\nSummary Table:")
    print("-" * 60)
    pair_label = f"{cofactor_str_display}+{substrate_str_display}"
    print(f"{'Group':<20} {'Total':<10} {pair_label:<20} {'Percentage':<10}")
    print("-" * 60)
    for gk, group_label in zip(group_keys, group_names):
        total = results[gk]['total']
        count = results[gk]['count']
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{group_label:<20} {total:<10} {count:<15} {percentage:.1f}%")
    print("-" * 60)
    
    # Create plots - 2x2 grid with bottom spanning both columns
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Top left: count plot with percentage
    ax3 = fig.add_subplot(gs[0, 1])  # Top right: distribution for NAP/NDP+FOL
    ax4 = fig.add_subplot(gs[1, :])  # Bottom: distribution for rest entries (spanning both columns)
    
    # Colors for amino acid groups
    group_colors = {
        'M_low': 'steelblue',
        'M_high': 'lightblue',
        'L': 'orange',
        'I': 'green'
    }
    
    # Extract data for plotting
    totals = [results[gk]['total'] for gk in group_keys]
    counts = [results[gk]['count'] for gk in group_keys]
    percentages = [(count / total * 100) if total > 0 else 0 
                   for count, total in zip(counts, totals)]
    
    x = np.arange(len(group_names))
    width = 0.6
    
    # Plot 1: Counts (absolute numbers)
    bars1 = ax1.bar(x, counts, width, color=[group_colors[gk] for gk in group_keys],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars (with percentage)
    for i, (bar, count, total, pct) in enumerate(zip(bars1, counts, totals, percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Amino Acid Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Entries', fontsize=12, fontweight='bold')
    ax1.set_title(f'Entries with exactly {plot_title_suffix}\n(Count and Percentage)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 10)
    
    # Plot 3: Psi distribution for entries with exactly NAP/NDP and FOL
    # Combine M_low and M_high into single M group
    bins = np.linspace(0, 180, 37)  # 5-degree bins from 0 to 180
    
    # Combine M groups
    m_combined_psi = []
    for gk in ['M_low', 'M_high']:
        m_combined_psi.extend(results[gk]['psi_values'])
    
    if m_combined_psi:
        ax3.hist(m_combined_psi, bins=bins, alpha=0.7, color='steelblue', 
                edgecolor='black', linewidth=0.5, label=f'M (n={len(m_combined_psi)})',
                density=True)
    
    # Plot L and I separately
    for gk, group_label, color in [('L', 'L', 'orange'), ('I', 'I', 'green')]:
        psi_values = results[gk]['psi_values']
        if psi_values:
            ax3.hist(psi_values, bins=bins, alpha=0.7, color=color, 
                    edgecolor='black', linewidth=0.5, label=f'{group_label} (n={len(psi_values)})',
                    density=True)
    
    # Add vertical line at threshold
    ax3.axvline(x=psi_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({psi_threshold}°)', alpha=0.7)
    
    ax3.set_xlabel('Psi (degrees)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax3.set_title(f'With exactly {plot_title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_xlim(0, 180)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Psi distribution for all 4 groups WITHOUT exactly NAP/NDP and FOL
    # Create histogram with smaller bin size, x-axis from -180 to 180
    bins_rest = np.linspace(-180, 180, 73)  # 5-degree bins from -180 to 180
    
    # Create x-axis for KDE
    x_kde = np.linspace(-180, 180, 1000)
    
    for gk, group_label, color in zip(group_keys, group_names, [group_colors[gk] for gk in group_keys]):
        rest_psi_values = results[gk]['rest_psi_values']
        if rest_psi_values:
            # Filter for psi > 50 for KDE fitting
            filtered_psi = [psi for psi in rest_psi_values if psi > 50]
            
            # Plot histogram
            ax4.hist(rest_psi_values, bins=bins_rest, alpha=0.7, color=color, 
                    edgecolor='black', linewidth=0.5, label=f'{group_label} (n={len(rest_psi_values)})',
                    density=True)
            
            # Add KDE fitting (only for psi > 50)
            if len(filtered_psi) > 1:  # Need at least 2 points for KDE
                try:
                    kde = gaussian_kde(filtered_psi)
                    # Only plot KDE in the range where we have data
                    x_kde_filtered = x_kde[(x_kde >= 50) & (x_kde <= max(filtered_psi) + 10)]
                    kde_values = kde(x_kde_filtered)
                    ax4.plot(x_kde_filtered, kde_values, color=color, linewidth=2, 
                            linestyle='-', alpha=0.8)  # No label, so won't appear in legend
                except:
                    pass  # Skip if KDE fails
    
    # Add vertical line at threshold
    ax4.axvline(x=psi_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({psi_threshold}°)', alpha=0.7)
    
    ax4.set_xlabel('Psi (degrees)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax4.set_title('All other entries (KDE fitted using psi > 50)', 
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_xlim(-180, 180)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_plot}")
    plt.close()
    
    return results


# Convenience function for backward compatibility
def count_nap_ndp_fol(
    classification_csv: str,
    dihedral_csv: str,
    output_plot: str = None,
    psi_threshold: float = 128.39
):
    """
    Count entries with exactly NAP/NDP and FOL (backward compatibility wrapper).
    """
    return compare_ligand_pair_psi_distribution(
        classification_csv,
        dihedral_csv,
        target_cofactors={'NAP', 'NDP'},
        target_substrates={'FOL'},
        output_plot=output_plot,
        psi_threshold=psi_threshold,
        plot_title_suffix="NAP/NDP and FOL"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare psi distribution for entries with exactly specified cofactor-substrate pair"
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
        help="Output plot file"
    )
    parser.add_argument(
        "--cofactors",
        type=str,
        nargs='+',
        default=['NAP', 'NDP'],
        help="Target cofactors (default: NAP NDP)"
    )
    parser.add_argument(
        "--substrates",
        type=str,
        nargs='+',
        default=['FOL'],
        help="Target substrates/antibiotics (default: FOL)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.classification is None:
        args.classification = Path(__file__).parent / "data" / "ligand_classification.csv"
    if args.dihedral is None:
        args.dihedral = Path(__file__).parent / "data" / "dihedral_angles_pos20.csv"
    
    # Convert to sets
    target_cofactors = set(c.upper() for c in args.cofactors)
    target_substrates = set(s.upper() for s in args.substrates)
    
    compare_ligand_pair_psi_distribution(
        str(args.classification),
        str(args.dihedral),
        target_cofactors,
        target_substrates,
        args.plot,
        args.threshold
    )
    
    # Also generate plot for NAP/NDP and MTX/TOP
    print("\n" + "=" * 60)
    print("Generating plot for NAP/NDP and MTX/TOP...")
    print("=" * 60)
    
    compare_ligand_pair_psi_distribution(
        str(args.classification),
        str(args.dihedral),
        target_cofactors={'NAP', 'NDP'},
        target_substrates={'MTX', 'TOP'},
        output_plot=None,  # Use default filename
        psi_threshold=args.threshold,
        plot_title_suffix="NAP/NDP and MTX/TOP"
    )

