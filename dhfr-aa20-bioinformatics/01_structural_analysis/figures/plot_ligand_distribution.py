#!/usr/bin/env python3
"""
Analyze and plot ligand distribution for 4 amino acid groups:
- M (psi < 128.39)
- M (psi > 128.39)
- L
- I

Upper plot: Cofactor distribution (NAP, NDP, other, none)
Lower plot: Substrate/Antibiotic distribution (FOL, MTX, TOP, other, none)
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# Cofactor group
COFACTORS = {'NDP', 'NAP', 'ATR'}

# Major substrates/antibiotics
MAJOR_SUBSTRATES = {'FOL', 'MTX', 'TOP'}


def classify_cofactor(cofactor_str: str) -> tuple:
    """
    Classify cofactor into NAP, NDP, other, or none.
    
    Returns:
        (category, is_only_cofactor)
        category: 'NAP', 'NDP', 'other', or 'none'
        is_only_cofactor: True if entry has ONLY cofactor (no substrate/antibiotic)
    """
    if not cofactor_str or not cofactor_str.strip():
        return 'none', False
    
    cofactors = [c.strip().upper() for c in cofactor_str.split(';') if c.strip()]
    
    if not cofactors:
        return 'none', False
    
    # Check if has NAP
    if 'NAP' in cofactors:
        return 'NAP', True  # Will be updated based on substrate presence
    
    # Check if has NDP
    if 'NDP' in cofactors:
        return 'NDP', True  # Will be updated based on substrate presence
    
    # Other cofactor
    return 'other', True  # Will be updated based on substrate presence


def classify_substrate(substrate_str: str) -> tuple:
    """
    Classify substrate/antibiotic into FOL, MTX, TOP, other, or none.
    
    Returns:
        (category, is_only_substrate)
        category: 'FOL', 'MTX', 'TOP', 'other', or 'none'
        is_only_substrate: True if entry has ONLY substrate/antibiotic (no cofactor)
    """
    if not substrate_str or not substrate_str.strip():
        return 'none', False
    
    substrates = [s.strip().upper() for s in substrate_str.split(';') if s.strip()]
    
    if not substrates:
        return 'none', False
    
    # Check for major substrates (priority order)
    if 'FOL' in substrates:
        return 'FOL', True  # Will be updated based on cofactor presence
    
    if 'MTX' in substrates:
        return 'MTX', True  # Will be updated based on cofactor presence
    
    if 'TOP' in substrates:
        return 'TOP', True  # Will be updated based on cofactor presence
    
    # Other substrate
    return 'other', True  # Will be updated based on cofactor presence


def analyze_ligand_distribution(
    classification_csv: str,
    dihedral_csv: str,
    output_plot: str = None,
    output_cofactor_other: str = None,
    output_substrate_other: str = None,
    psi_threshold: float = 128.39
):
    """
    Analyze and plot ligand distribution.
    
    Args:
        classification_csv: Path to ligand_classification.csv
        dihedral_csv: Path to dihedral_angles_pos20.csv (for psi values)
        output_plot: Path to output plot
        output_cofactor_other: Path to output file for "other" cofactors
        output_substrate_other: Path to output file for "other" substrates
        psi_threshold: Psi threshold for M classification
    """
    class_path = Path(classification_csv)
    dihedral_path = Path(dihedral_csv)
    
    if output_plot is None:
        output_plot = class_path.parent / "ligand_distribution.png"
    else:
        output_plot = Path(output_plot)
    
    if output_cofactor_other is None:
        output_cofactor_other = class_path.parent / "cofactor_other_chemicals.csv"
    else:
        output_cofactor_other = Path(output_cofactor_other)
    
    if output_substrate_other is None:
        output_substrate_other = class_path.parent / "substrate_other_chemicals.csv"
    else:
        output_substrate_other = Path(output_substrate_other)
    
    # Load psi values for M entries
    psi_values = {}
    with open(dihedral_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            aa = row.get('Amino_Acid', '').strip().upper()
            psi_str = row.get('Psi', '').strip()
            status = row.get('Status', '').strip()
            
            if status == 'Success' and aa == 'M' and psi_str and psi_str.lower() != 'none':
                try:
                    psi = float(psi_str)
                    psi_values[pdb_id] = psi
                except (ValueError, TypeError):
                    pass
    
    # Load classification data and group entries
    groups = {
        'M_low': [],   # M with psi < threshold
        'M_high': [],  # M with psi > threshold
        'L': [],       # L entries
        'I': []        # I entries
    }
    
    other_cofactors = defaultdict(list)  # group -> list of (pdb_id, chemical_code) tuples
    other_substrates = defaultdict(list)  # group -> list of (pdb_id, chemical_code) tuples
    
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
                # Classify cofactor
                cf_category, _ = classify_cofactor(cofactor_str)
                has_substrate = bool(substrate_str and substrate_str.strip())
                is_only_cofactor = not has_substrate
                
                # Classify substrate
                sub_category, _ = classify_substrate(substrate_str)
                has_cofactor = bool(cofactor_str and cofactor_str.strip())
                is_only_substrate = not has_cofactor
                
                # Collect "other" chemicals with PDB_ID
                if cf_category == 'other':
                    cofactors = [c.strip().upper() for c in cofactor_str.split(';') if c.strip()]
                    for chem in cofactors:
                        other_cofactors[group].append((pdb_id, chem))
                
                if sub_category == 'other':
                    substrates = [s.strip().upper() for s in substrate_str.split(';') if s.strip()]
                    for chem in substrates:
                        other_substrates[group].append((pdb_id, chem))
                
                groups[group].append({
                    'pdb_id': pdb_id,
                    'cofactor_category': cf_category,
                    'is_only_cofactor': is_only_cofactor,
                    'substrate_category': sub_category,
                    'is_only_substrate': is_only_substrate,
                    'cofactor_str': cofactor_str,
                    'substrate_str': substrate_str
                })
    
    # Count cofactors by category
    cofactor_counts = {
        'M_low': {'NAP': {'total': 0, 'only': 0}, 'NDP': {'total': 0, 'only': 0}, 
                  'other': {'total': 0, 'only': 0}, 'none': {'total': 0, 'only': 0}},
        'M_high': {'NAP': {'total': 0, 'only': 0}, 'NDP': {'total': 0, 'only': 0}, 
                   'other': {'total': 0, 'only': 0}, 'none': {'total': 0, 'only': 0}},
        'L': {'NAP': {'total': 0, 'only': 0}, 'NDP': {'total': 0, 'only': 0}, 
              'other': {'total': 0, 'only': 0}, 'none': {'total': 0, 'only': 0}},
        'I': {'NAP': {'total': 0, 'only': 0}, 'NDP': {'total': 0, 'only': 0}, 
              'other': {'total': 0, 'only': 0}, 'none': {'total': 0, 'only': 0}}
    }
    
    # Count substrates by category
    substrate_counts = {
        'M_low': {'FOL': {'total': 0, 'only': 0}, 'MTX': {'total': 0, 'only': 0}, 
                  'TOP': {'total': 0, 'only': 0}, 'other': {'total': 0, 'only': 0}, 
                  'none': {'total': 0, 'only': 0}},
        'M_high': {'FOL': {'total': 0, 'only': 0}, 'MTX': {'total': 0, 'only': 0}, 
                   'TOP': {'total': 0, 'only': 0}, 'other': {'total': 0, 'only': 0}, 
                   'none': {'total': 0, 'only': 0}},
        'L': {'FOL': {'total': 0, 'only': 0}, 'MTX': {'total': 0, 'only': 0}, 
              'TOP': {'total': 0, 'only': 0}, 'other': {'total': 0, 'only': 0}, 
              'none': {'total': 0, 'only': 0}},
        'I': {'FOL': {'total': 0, 'only': 0}, 'MTX': {'total': 0, 'only': 0}, 
              'TOP': {'total': 0, 'only': 0}, 'other': {'total': 0, 'only': 0}, 
              'none': {'total': 0, 'only': 0}}
    }
    
    for group_name, entries in groups.items():
        for entry in entries:
            # Count cofactors
            cf_cat = entry['cofactor_category']
            cofactor_counts[group_name][cf_cat]['total'] += 1
            if entry['is_only_cofactor']:
                cofactor_counts[group_name][cf_cat]['only'] += 1
            
            # Count substrates
            sub_cat = entry['substrate_category']
            substrate_counts[group_name][sub_cat]['total'] += 1
            if entry['is_only_substrate']:
                substrate_counts[group_name][sub_cat]['only'] += 1
    
    # Define group names and keys for later use
    group_names = ['M (psi<128.39)', 'M (psi>128.39)', 'L', 'I']
    group_keys = ['M_low', 'M_high', 'L', 'I']
    
    # Verify counts: "none" in cofactor should equal sum of "only substrate/antibiotic"
    # "none" in substrate should equal sum of "only cofactor"
    print("\nVerifying count consistency:")
    print("=" * 60)
    substrate_categories = ['FOL', 'MTX', 'TOP', 'other', 'none']
    cofactor_categories = ['NAP', 'NDP', 'other', 'none']
    for gk, group_label in zip(group_keys, group_names):
        # Check: "none" in cofactor should equal sum of "only substrate/antibiotic"
        none_cofactor = cofactor_counts[gk]['none']['total']
        sum_only_substrate = sum(substrate_counts[gk][cat]['only'] for cat in substrate_categories)
        print(f"{group_label}:")
        print(f"  'none' in cofactor: {none_cofactor}")
        print(f"  Sum of 'only substrate/antibiotic': {sum_only_substrate}")
        if none_cofactor != sum_only_substrate:
            print(f"  WARNING: Mismatch! Difference: {abs(none_cofactor - sum_only_substrate)}")
        
        # Check: "none" in substrate should equal sum of "only cofactor"
        none_substrate = substrate_counts[gk]['none']['total']
        sum_only_cofactor = sum(cofactor_counts[gk][cat]['only'] for cat in cofactor_categories)
        print(f"  'none' in substrate/antibiotic: {none_substrate}")
        print(f"  Sum of 'only cofactor': {sum_only_cofactor}")
        if none_substrate != sum_only_cofactor:
            print(f"  WARNING: Mismatch! Difference: {abs(none_substrate - sum_only_cofactor)}")
        print()
    print("=" * 60)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Note: group_names and group_keys already defined above
    
    # Colors for amino acid groups
    group_colors = {
        'M_low': 'steelblue',
        'M_high': 'lightblue',
        'L': 'orange',
        'I': 'green'
    }
    
    # Plot 1: Cofactor distribution
    # X-axis: cofactor categories, Y-axis: counts
    # 4 bars per category (one for each amino acid group)
    # Note: cofactor_categories already defined above
    x = np.arange(len(cofactor_categories))
    width = 0.2  # Width for each bar
    
    for i, (gk, group_label) in enumerate(zip(group_keys, group_names)):
        totals = [cofactor_counts[gk][cf_cat]['total'] for cf_cat in cofactor_categories]
        onlys = [cofactor_counts[gk][cf_cat]['only'] for cf_cat in cofactor_categories]
        with_both = [t - o for t, o in zip(totals, onlys)]
        
        # Calculate bar positions (4 bars side by side)
        x_pos = x + (i - 1.5) * width
        
        # Plot "with both" (no hatch, filled) - bottom part
        bars1 = ax1.bar(x_pos, with_both, width,
                       color=group_colors[gk], alpha=0.7, edgecolor='black', linewidth=0.5, label='_nolegend_')
        
        # Plot "only cofactor" (with hatch) - top part (except for "none")
        onlys_plot = [o if cat != 'none' else 0 for o, cat in zip(onlys, cofactor_categories)]
        bars2 = ax1.bar(x_pos, onlys_plot, width, bottom=with_both,
                       color=group_colors[gk], alpha=0.9, hatch='///',
                       edgecolor='black', linewidth=0.5, label='_nolegend_')
    
    # Create custom legend: separate amino acids and patterns
    from matplotlib.patches import Patch
    legend_elements_aa = []
    for gk, group_label in zip(group_keys, group_names):
        n = len(groups[gk])
        legend_elements_aa.append(Patch(facecolor=group_colors[gk], alpha=0.7, edgecolor='black', 
                                      label=f"{group_label} (n={n})"))
    
    # Pattern legend
    legend_elements_pattern = [
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='2+ ligands'),
        Patch(facecolor='none', hatch='///', edgecolor='black', label='Only cofactor')
    ]
    
    # Create two separate legends
    leg1 = ax1.legend(handles=legend_elements_aa, loc='upper left', fontsize=9, title='Amino Acid', title_fontsize=10)
    leg2 = ax1.legend(handles=legend_elements_pattern, loc='upper right', fontsize=9, title='Ligand Type', title_fontsize=10)
    ax1.add_artist(leg1)  # Re-add first legend since second one overwrites it
    
    ax1.set_xlabel('Cofactor Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Entries', fontsize=12, fontweight='bold')
    ax1.set_title('Cofactor Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cofactor_categories)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 2: Substrate/Antibiotic distribution
    # X-axis: substrate categories, Y-axis: counts
    # 4 bars per category (one for each amino acid group)
    # Note: substrate_categories already defined above
    x = np.arange(len(substrate_categories))
    
    for i, (gk, group_label) in enumerate(zip(group_keys, group_names)):
        totals = [substrate_counts[gk][sub_cat]['total'] for sub_cat in substrate_categories]
        onlys = [substrate_counts[gk][sub_cat]['only'] for sub_cat in substrate_categories]
        with_both = [t - o for t, o in zip(totals, onlys)]
        
        # Calculate bar positions (4 bars side by side)
        x_pos = x + (i - 1.5) * width
        
        # Plot "with both" (no hatch, filled) - bottom part
        bars1 = ax2.bar(x_pos, with_both, width,
                       color=group_colors[gk], alpha=0.7, edgecolor='black', linewidth=0.5, label='_nolegend_')
        
        # Plot "only substrate" (with hatch) - top part (except for "none")
        onlys_plot = [o if cat != 'none' else 0 for o, cat in zip(onlys, substrate_categories)]
        bars2 = ax2.bar(x_pos, onlys_plot, width, bottom=with_both,
                       color=group_colors[gk], alpha=0.9, hatch='///',
                       edgecolor='black', linewidth=0.5, label='_nolegend_')
    
    # Create custom legend: separate amino acids and patterns
    from matplotlib.patches import Patch
    legend_elements_aa = []
    for gk, group_label in zip(group_keys, group_names):
        n = len(groups[gk])
        legend_elements_aa.append(Patch(facecolor=group_colors[gk], alpha=0.7, edgecolor='black', 
                                      label=f"{group_label} (n={n})"))
    
    # Pattern legend
    legend_elements_pattern = [
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='2+ ligands'),
        Patch(facecolor='none', hatch='///', edgecolor='black', label='Only substrate/antibiotic')
    ]
    
    # Create two separate legends
    leg1 = ax2.legend(handles=legend_elements_aa, loc='upper left', fontsize=9, title='Amino Acid', title_fontsize=10)
    leg2 = ax2.legend(handles=legend_elements_pattern, loc='upper right', fontsize=9, title='Ligand Type', title_fontsize=10)
    ax2.add_artist(leg1)  # Re-add first legend since second one overwrites it
    
    ax2.set_xlabel('Substrate/Antibiotic Group', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Entries', fontsize=12, fontweight='bold')
    ax2.set_title('Substrate/Antibiotic Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(substrate_categories)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_plot}")
    plt.close()
    
    # Write "other" cofactor chemicals
    with open(output_cofactor_other, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'PDB_ID', 'Cofactor_Chemical'])
        for group_key in group_keys:
            group_label = group_names[group_keys.index(group_key)]
            # Sort by chemical code, then by PDB_ID
            sorted_entries = sorted(set(other_cofactors[group_key]), key=lambda x: (x[1], x[0]))
            for pdb_id, chem in sorted_entries:
                writer.writerow([group_label, pdb_id, chem])
    
    print(f"Other cofactor chemicals saved to: {output_cofactor_other}")
    
    # Write "other" substrate chemicals
    with open(output_substrate_other, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'PDB_ID', 'Substrate_Antibiotic_Chemical'])
        for group_key in group_keys:
            group_label = group_names[group_keys.index(group_key)]
            # Sort by chemical code, then by PDB_ID
            sorted_entries = sorted(set(other_substrates[group_key]), key=lambda x: (x[1], x[0]))
            for pdb_id, chem in sorted_entries:
                writer.writerow([group_label, pdb_id, chem])
    
    print(f"Other substrate/antibiotic chemicals saved to: {output_substrate_other}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    for i, group_key in enumerate(group_keys):
        group_label = group_names[i]
        print(f"\n{group_label} (n={len(groups[group_key])}):")
        print(f"  Cofactors:")
        for cf_cat in cofactor_categories:
            total = cofactor_counts[group_key][cf_cat]['total']
            only = cofactor_counts[group_key][cf_cat]['only']
            print(f"    {cf_cat}: {total} (only: {only})")
        print(f"  Substrates/Antibiotics:")
        for sub_cat in substrate_categories:
            total = substrate_counts[group_key][sub_cat]['total']
            only = substrate_counts[group_key][sub_cat]['only']
            print(f"    {sub_cat}: {total} (only: {only})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot ligand distribution"
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
        "-p", "--plot",
        type=str,
        default=None,
        help="Output plot file (default: ligand_distribution.png)"
    )
    parser.add_argument(
        "-co", "--cofactor-other",
        type=str,
        default=None,
        help="Output file for other cofactors (default: cofactor_other_chemicals.csv)"
    )
    parser.add_argument(
        "-so", "--substrate-other",
        type=str,
        default=None,
        help="Output file for other substrates (default: substrate_other_chemicals.csv)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=128.39,
        help="Psi threshold for M classification (default: 128.39)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.classification is None:
        args.classification = Path(__file__).resolve().parent.parent / "data" / "ligand_classification.csv"
    if args.dihedral is None:
        args.dihedral = Path(__file__).resolve().parent.parent / "data" / "dihedral_angles_pos20.csv"
    
    analyze_ligand_distribution(
        str(args.classification),
        str(args.dihedral),
        args.plot,
        args.cofactor_other,
        args.substrate_other,
        args.threshold
    )

