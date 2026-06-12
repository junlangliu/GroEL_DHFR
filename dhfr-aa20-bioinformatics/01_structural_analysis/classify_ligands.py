#!/usr/bin/env python3
"""
Classify ligands into cofactors and substrates/antibiotics.
Output a CSV file with pdb_id, pos20_amino_acid, cofactor, and substrate/antibiotics.
"""

import csv
import argparse
from pathlib import Path
from collections import Counter

# Common water and ion identifiers to exclude (from cluster_pdb.py)
WATER_AND_IONS = {
    # Water molecules
    "HOH", "WAT", "OH2", "H2O", "DOD",
    # Common ions
    "NA", "CL", "K", "MG", "CA", "MN", "FE", "ZN", "CU", "NI", "CO",
    "SO4", "PO4", "CO3", "NO3", "NH4", "AC", "AG", "AL", "AS", "AU",
    "BA", "BE", "BR", "CD", "CE", "CS", "DY", "ER", "EU", "F", "GD",
    "HE", "HG", "HO", "I", "IN", "IR", "LA", "LI", "LU", "MO", "ND",
    "OS", "PB", "PD", "PR", "PT", "RB", "RE", "RH", "RU", "SB", "SC",
    "SE", "SM", "SR", "TB", "TC", "TE", "TH", "TI", "TL", "TM", "U",
    "V", "W", "Y", "YB", "ZR",
    # Additional common buffer/salt components
    "ACT", "CIT", "EDO", "GOL", "HEP", "MES", "TRS", "TRI", "PEG",
    "MPD", "DMS", "DTT", "BME",
    # Non-DHFR ligands
    "CB3", "UMP", "EOH", "GLY"
}

# Cofactor group
COFACTORS = {'NDP', 'NAP', 'ATR'}

# Substrate/antibiotics group (FOL, MTX, TOP, and all other ligands that are not cofactors)
# This will be determined dynamically - any ligand not in COFACTORS and not in WATER_AND_IONS


def filter_ligands(ligands_str: str) -> list:
    """
    Filter out water, ions, buffers, and non-DHFR ligands.
    
    Args:
        ligands_str: Semicolon-separated ligand string
        
    Returns:
        List of filtered ligands
    """
    if not ligands_str or not ligands_str.strip():
        return []
    
    # Split by semicolon, strip whitespace, convert to uppercase
    ligands = [lig.strip().upper() for lig in ligands_str.split(";") if lig.strip()]
    
    # Filter out water, ions, and non-DHFR ligands
    filtered = [lig for lig in ligands if lig not in WATER_AND_IONS]
    
    return filtered


def classify_ligands(ligands: list) -> tuple:
    """
    Classify ligands into cofactors and substrates/antibiotics.
    
    Args:
        ligands: List of ligand codes
        
    Returns:
        (cofactors_list, substrates_list)
    """
    cofactors = []
    substrates = []
    
    for lig in ligands:
        if lig in COFACTORS:
            cofactors.append(lig)
        else:
            # Everything else (not cofactor, not water/ions) is substrate/antibiotic
            substrates.append(lig)
    
    return sorted(cofactors), sorted(substrates)


def process_entries(
    dihedral_csv: str,
    dhfr_csv: str,
    output_csv: str = None
):
    """
    Process entries and classify ligands.
    
    Args:
        dihedral_csv: Path to dihedral_angles_pos20.csv
        dhfr_csv: Path to dhfr_entries_cleaned.csv
        output_csv: Path to output CSV file
    """
    dihedral_path = Path(dihedral_csv)
    dhfr_path = Path(dhfr_csv)
    
    if output_csv is None:
        output_csv = dihedral_path.parent / "ligand_classification.csv"
    else:
        output_csv = Path(output_csv)
    
    if not dihedral_path.exists():
        print(f"Error: Dihedral CSV not found: {dihedral_path}")
        return None
    
    if not dhfr_path.exists():
        print(f"Error: DHFR CSV not found: {dhfr_path}")
        return None
    
    # Load dihedral angles data
    entries = []
    with open(dihedral_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            amino_acid = row.get('Amino_Acid', '').strip().upper()
            found_aa = row.get('Found_Amino_Acid', '').strip().upper()
            status = row.get('Status', '').strip()
            
            if status != 'Success':
                continue
            
            # Get amino acid (prefer 1-letter code)
            pos20_aa = amino_acid if amino_acid else found_aa
            if len(found_aa) == 3:
                # Convert 3-letter to 1-letter if needed
                aa_3_to_1 = {
                    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                }
                if found_aa in aa_3_to_1:
                    pos20_aa = aa_3_to_1[found_aa]
            
            entries.append({
                'PDB_ID': pdb_id,
                'Pos20_Amino_Acid': pos20_aa
            })
    
    # Load DHFR entries
    dhfr_data = {}
    with open(dhfr_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('pdb_id', '').strip().upper()
            if pdb_id:
                dhfr_data[pdb_id] = row
    
    # Process and classify ligands
    output_rows = []
    for entry in entries:
        pdb_id = entry['PDB_ID']
        pos20_aa = entry['Pos20_Amino_Acid']
        
        if pdb_id not in dhfr_data:
            continue
        
        ligands_str = dhfr_data[pdb_id].get('ligands', '')
        filtered_ligands = filter_ligands(ligands_str)
        cofactors, substrates = classify_ligands(filtered_ligands)
        
        # Format as semicolon-separated strings
        cofactor_str = '; '.join(cofactors) if cofactors else ''
        substrate_str = '; '.join(substrates) if substrates else ''
        
        output_rows.append({
            'PDB_ID': pdb_id,
            'Pos20_Amino_Acid': pos20_aa,
            'Cofactor': cofactor_str,
            'Substrate_Antibiotic': substrate_str
        })
    
    # Write output
    fieldnames = ['PDB_ID', 'Pos20_Amino_Acid', 'Cofactor', 'Substrate_Antibiotic']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"\n{'='*60}")
    print(f"Ligand classification saved to: {output_csv}")
    print(f"  Total entries: {len(output_rows)}")
    
    # Print summary statistics
    cofactor_counts = Counter()
    substrate_counts = Counter()
    entries_with_cofactor = 0
    entries_with_substrate = 0
    entries_with_both = 0
    entries_with_neither = 0
    
    for row in output_rows:
        has_cofactor = bool(row['Cofactor'])
        has_substrate = bool(row['Substrate_Antibiotic'])
        
        if has_cofactor:
            entries_with_cofactor += 1
            for cf in row['Cofactor'].split('; '):
                cofactor_counts[cf] += 1
        
        if has_substrate:
            entries_with_substrate += 1
            for sub in row['Substrate_Antibiotic'].split('; '):
                substrate_counts[sub] += 1
        
        if has_cofactor and has_substrate:
            entries_with_both += 1
        elif not has_cofactor and not has_substrate:
            entries_with_neither += 1
    
    print(f"\nSummary:")
    print(f"  Entries with cofactor: {entries_with_cofactor}")
    print(f"  Entries with substrate/antibiotic: {entries_with_substrate}")
    print(f"  Entries with both: {entries_with_both}")
    print(f"  Entries with neither: {entries_with_neither}")
    
    print(f"\nTop Cofactors:")
    for cf, count in cofactor_counts.most_common():
        print(f"  {cf}: {count}")
    
    print(f"\nTop Substrates/Antibiotics:")
    for sub, count in substrate_counts.most_common(10):
        print(f"  {sub}: {count}")
    
    print(f"{'='*60}\n")
    
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify ligands into cofactors and substrates/antibiotics"
    )
    parser.add_argument(
        "-d", "--dihedral",
        type=str,
        default=None,
        help="Path to dihedral_angles_pos20.csv"
    )
    parser.add_argument(
        "-f", "--dhfr",
        type=str,
        default=None,
        help="Path to dhfr_entries_cleaned.csv"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV file (default: ligand_classification.csv)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.dihedral is None:
        args.dihedral = Path(__file__).parent / "dihedral_angles_pos20.csv"
    if args.dhfr is None:
        args.dhfr = Path(__file__).parent.parent / "pdb_extraction" / "dhfr_entries_cleaned.csv"
    
    process_entries(
        str(args.dihedral),
        str(args.dhfr),
        args.output
    )

