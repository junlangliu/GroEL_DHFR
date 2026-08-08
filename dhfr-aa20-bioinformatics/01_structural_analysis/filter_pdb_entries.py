#!/usr/bin/env python3
"""
Filter dihedral angles CSV to extract entries where chain description
does not contain "Dihydrofolate reductase" or "dhfr", or entries with
amino acid mismatches.
"""

import csv
from pathlib import Path

def filter_amino_acid_mismatch(input_csv: str, output_csv: str = None):
    """
    Filter CSV to extract entries where Amino_Acid doesn't match Found_Amino_Acid.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (default: adds _mismatch suffix)
    """
    input_path = Path(input_csv)
    
    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_mismatch.csv"
    else:
        output_csv = Path(output_csv)
    
    # Map 1-letter codes to 3-letter codes
    aa_1_to_3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    def normalize_aa(aa_str):
        """Convert amino acid to 3-letter code for comparison."""
        if not aa_str or aa_str.strip() == "":
            return None
        aa_str = aa_str.strip().upper()
        if len(aa_str) == 1 and aa_str in aa_1_to_3:
            return aa_1_to_3[aa_str]
        elif len(aa_str) == 3:
            return aa_str
        return aa_str
    
    filtered_rows = []
    total_rows = 0
    
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            expected_aa = row.get('Amino_Acid', '').strip()
            found_aa = row.get('Found_Amino_Acid', '').strip()
            
            # Skip if either is empty
            if not expected_aa or not found_aa:
                continue
            
            # Normalize both to 3-letter codes
            expected_normalized = normalize_aa(expected_aa)
            found_normalized = normalize_aa(found_aa)
            
            # Check for mismatch
            if expected_normalized and found_normalized and expected_normalized != found_normalized:
                filtered_rows.append(row)
    
    # Write filtered results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Input file: {input_path}")
    print(f"Total entries: {total_rows}")
    print(f"Mismatched entries: {len(filtered_rows)}")
    print(f"Output file: {output_csv}")
    
    return output_csv


def filter_non_dhfr(input_csv: str, output_csv: str = None):
    """
    Filter CSV to extract entries where chain description does not contain
    "Dihydrofolate reductase" or "dhfr" (case-insensitive).
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (default: adds _non_dhfr suffix)
    """
    input_path = Path(input_csv)
    
    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_non_dhfr.csv"
    else:
        output_csv = Path(output_csv)
    
    # Keywords to exclude (case-insensitive)
    # Include common typos and variations
    exclude_keywords = [
        'dihydrofolate reductase',
        'dihydrofolate reducatase',  # typo: missing 't'
        'dehydrofolate reductase',   # typo: missing 'i'
        'dhfr'
    ]
    
    filtered_rows = []
    total_rows = 0
    
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            chain_desc = row.get('Chain_Description', '').strip().lower()
            
            # Check if chain description contains any exclude keywords
            contains_dhfr = any(keyword in chain_desc for keyword in exclude_keywords)
            
            if not contains_dhfr:
                filtered_rows.append(row)
    
    # Write filtered results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Input file: {input_path}")
    print(f"Total entries: {total_rows}")
    print(f"Filtered entries (non-DHFR): {len(filtered_rows)}")
    print(f"Excluded entries (DHFR): {total_rows - len(filtered_rows)}")
    print(f"Output file: {output_csv}")
    
    return output_csv


def compare_chain_lengths(
    dihedral_csv: str,
    dhfr_csv: str,
    output_csv: str = None
):
    """
    Compare chain lengths between dihedral_angles_pos20.csv and dhfr_entries_cleaned.csv.
    Reports entries where Chain_Length doesn't match sequence_length.
    
    Args:
        dihedral_csv: Path to dihedral_angles_pos20.csv
        dhfr_csv: Path to dhfr_entries_cleaned.csv
        output_csv: Path to output CSV file (default: adds _length_mismatch suffix)
    """
    dihedral_path = Path(dihedral_csv)
    dhfr_path = Path(dhfr_csv)
    
    if output_csv is None:
        output_csv = dihedral_path.parent / f"{dihedral_path.stem}_length_mismatch.csv"
    else:
        output_csv = Path(output_csv)
    
    if not dihedral_path.exists():
        print(f"Error: Dihedral CSV not found: {dihedral_path}")
        return None
    
    if not dhfr_path.exists():
        print(f"Error: DHFR CSV not found: {dhfr_path}")
        return None
    
    # Load DHFR entries into a dictionary: pdb_id -> sequence_length
    dhfr_lengths = {}
    with open(dhfr_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get('pdb_id', '').strip().upper()
            seq_length_str = row.get('sequence_length', '').strip()
            if pdb_id and seq_length_str:
                try:
                    seq_length = int(seq_length_str)
                    dhfr_lengths[pdb_id] = seq_length
                except ValueError:
                    continue
    
    print(f"Loaded {len(dhfr_lengths)} entries from DHFR CSV")
    
    # Compare with dihedral angles CSV
    mismatch_rows = []
    matches = []
    not_found_in_dhfr = []
    missing_chain_length = []
    
    with open(dihedral_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        # Add comparison columns
        if 'DHFR_Sequence_Length' not in fieldnames:
            fieldnames.append('DHFR_Sequence_Length')
        if 'Length_Match' not in fieldnames:
            fieldnames.append('Length_Match')
        if 'Length_Difference' not in fieldnames:
            fieldnames.append('Length_Difference')
        
        for row in reader:
            pdb_id = row.get('PDB_ID', '').strip().upper()
            chain_length_str = row.get('Chain_Length', '').strip()
            
            if not pdb_id:
                continue
            
            # Get chain length from dihedral CSV
            try:
                chain_length = int(chain_length_str) if chain_length_str else None
            except ValueError:
                chain_length = None
            
            # Get sequence length from DHFR CSV
            dhfr_seq_length = dhfr_lengths.get(pdb_id)
            
            # Add comparison data
            row['DHFR_Sequence_Length'] = str(dhfr_seq_length) if dhfr_seq_length is not None else ''
            
            if chain_length is None:
                missing_chain_length.append(pdb_id)
                row['Length_Match'] = 'Missing Chain_Length'
                row['Length_Difference'] = ''
            elif dhfr_seq_length is None:
                not_found_in_dhfr.append(pdb_id)
                row['Length_Match'] = 'Not found in DHFR CSV'
                row['Length_Difference'] = ''
            elif chain_length == dhfr_seq_length:
                matches.append(pdb_id)
                row['Length_Match'] = 'Match'
                row['Length_Difference'] = '0'
            else:
                mismatch_rows.append(row)
                row['Length_Match'] = 'Mismatch'
                row['Length_Difference'] = str(chain_length - dhfr_seq_length)
    
    # Write mismatches to output CSV
    if mismatch_rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatch_rows)
    else:
        # Create empty file with headers if no mismatches
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Print summary
    total_entries = len(matches) + len(mismatch_rows) + len(not_found_in_dhfr) + len(missing_chain_length)
    print("\n" + "=" * 60)
    print("Chain Length Comparison Summary")
    print("=" * 60)
    print(f"Dihedral CSV: {dihedral_path}")
    print(f"DHFR CSV: {dhfr_path}")
    print(f"\nTotal entries in dihedral CSV: {total_entries}")
    print(f"  Matches: {len(matches)}")
    print(f"  Mismatches: {len(mismatch_rows)}")
    print(f"  Not found in DHFR CSV: {len(not_found_in_dhfr)}")
    print(f"  Missing Chain_Length: {len(missing_chain_length)}")
    
    if mismatch_rows:
        print(f"\nMismatched entries saved to: {output_csv}")
        print("\nSample mismatches:")
        for i, row in enumerate(mismatch_rows[:5]):
            pdb_id = row.get('PDB_ID', '')
            chain_len = row.get('Chain_Length', '')
            dhfr_len = row.get('DHFR_Sequence_Length', '')
            diff = row.get('Length_Difference', '')
            print(f"  {pdb_id}: Chain_Length={chain_len}, DHFR_Length={dhfr_len}, Diff={diff}")
    else:
        print("\n✓ All chain lengths match!")
    
    print("=" * 60)
    
    return output_csv


def check_data_quality(input_csv: str):
    """
    Check data quality of dihedral angles CSV:
    1. Verify all amino acids match (Amino_Acid == Found_Amino_Acid)
    2. Verify all phi/psi angles are calculated
    
    Args:
        input_csv: Path to dihedral_angles_pos20.csv
    """
    input_path = Path(input_csv)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Map 1-letter codes to 3-letter codes
    aa_1_to_3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    def normalize_aa(aa_str):
        """Convert amino acid to 3-letter code for comparison."""
        if not aa_str or aa_str.strip() == "":
            return None
        aa_str = aa_str.strip().upper()
        # Handle MSE (selenomethionine) as MET
        if aa_str == 'MSE':
            return 'MET'
        if len(aa_str) == 1 and aa_str in aa_1_to_3:
            return aa_1_to_3[aa_str]
        elif len(aa_str) == 3:
            return aa_str
        return aa_str
    
    total_entries = 0
    success_entries = 0
    amino_acid_matches = 0
    amino_acid_mismatches = 0
    missing_amino_acid = 0
    phi_calculated = 0
    psi_calculated = 0
    both_calculated = 0
    missing_phi = 0
    missing_psi = 0
    missing_both = 0
    
    mismatch_details = []
    
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_entries += 1
            status = row.get('Status', '').strip()
            
            # Check status
            if status == 'Success':
                success_entries += 1
            
            # Check amino acid match
            expected_aa = row.get('Amino_Acid', '').strip()
            found_aa = row.get('Found_Amino_Acid', '').strip()
            
            if not expected_aa or not found_aa:
                missing_amino_acid += 1
            else:
                expected_normalized = normalize_aa(expected_aa)
                found_normalized = normalize_aa(found_aa)
                
                if expected_normalized and found_normalized:
                    if expected_normalized == found_normalized:
                        amino_acid_matches += 1
                    else:
                        amino_acid_mismatches += 1
                        mismatch_details.append({
                            'PDB_ID': row.get('PDB_ID', ''),
                            'Expected': expected_aa,
                            'Found': found_aa,
                            'Status': status
                        })
            
            # Check phi/psi calculation
            phi = row.get('Phi', '').strip()
            psi = row.get('Psi', '').strip()
            
            has_phi = phi and phi != '' and phi.lower() != 'none'
            has_psi = psi and psi != '' and psi.lower() != 'none'
            
            if has_phi:
                phi_calculated += 1
            else:
                missing_phi += 1
            
            if has_psi:
                psi_calculated += 1
            else:
                missing_psi += 1
            
            if has_phi and has_psi:
                both_calculated += 1
            elif not has_phi and not has_psi:
                missing_both += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Data Quality Check Summary")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"\nTotal entries: {total_entries}")
    print(f"Successful entries (Status='Success'): {success_entries}/{total_entries}")
    
    print(f"\n{'─' * 60}")
    print("Amino Acid Matching:")
    print(f"  ✓ Matches: {amino_acid_matches}/{total_entries}")
    print(f"  ✗ Mismatches: {amino_acid_mismatches}/{total_entries}")
    print(f"  ? Missing data: {missing_amino_acid}/{total_entries}")
    
    if amino_acid_mismatches > 0:
        print(f"\n  Mismatch details (first 10):")
        for i, detail in enumerate(mismatch_details[:10]):
            print(f"    {detail['PDB_ID']}: Expected {detail['Expected']}, Found {detail['Found']} (Status: {detail['Status']})")
        if len(mismatch_details) > 10:
            print(f"    ... and {len(mismatch_details) - 10} more")
    
    print(f"\n{'─' * 60}")
    print("Dihedral Angle Calculation:")
    print(f"  ✓ Phi calculated: {phi_calculated}/{total_entries}")
    print(f"  ✓ Psi calculated: {psi_calculated}/{total_entries}")
    print(f"  ✓ Both calculated: {both_calculated}/{total_entries}")
    print(f"  ✗ Missing Phi: {missing_phi}/{total_entries}")
    print(f"  ✗ Missing Psi: {missing_psi}/{total_entries}")
    print(f"  ✗ Missing both: {missing_both}/{total_entries}")
    
    print(f"\n{'─' * 60}")
    print("Overall Status:")
    all_match = amino_acid_mismatches == 0 and missing_amino_acid == 0
    all_calculated = missing_both == 0 and both_calculated == success_entries
    
    if all_match:
        print("  ✓ All amino acids match!")
    else:
        print(f"  ✗ {amino_acid_mismatches + missing_amino_acid} entries have amino acid issues")
    
    if all_calculated:
        print("  ✓ All phi/psi angles are calculated for successful entries!")
    else:
        print(f"  ✗ {missing_both} entries are missing both phi and psi")
        if success_entries > both_calculated:
            print(f"     Note: {success_entries - both_calculated} successful entries are missing angles")
    
    print("=" * 60)
    
    return {
        'total': total_entries,
        'success': success_entries,
        'aa_matches': amino_acid_matches,
        'aa_mismatches': amino_acid_mismatches,
        'aa_missing': missing_amino_acid,
        'phi_calculated': phi_calculated,
        'psi_calculated': psi_calculated,
        'both_calculated': both_calculated,
        'missing_both': missing_both
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter CSV to extract non-DHFR entries"
    )
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
        help="Output CSV file"
    )
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="Filter for entries with amino acid mismatches (Amino_Acid != Found_Amino_Acid)"
    )
    parser.add_argument(
        "--compare-lengths",
        action="store_true",
        help="Compare chain lengths between dihedral_angles_pos20.csv and dhfr_entries_cleaned.csv"
    )
    parser.add_argument(
        "--dhfr-csv",
        type=str,
        default=None,
        help="Path to dhfr_entries_cleaned.csv (default: ../pdb_extraction/dhfr_entries_cleaned.csv)"
    )
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Check data quality: amino acid matches and phi/psi calculation status"
    )
    
    args = parser.parse_args()
    
    if args.input is None:
        input_csv = Path(__file__).parent / "data" / "dihedral_angles_pos20.csv"
    else:
        input_csv = Path(args.input)
    
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        exit(1)
    
    if args.check_quality:
        check_data_quality(str(input_csv))
    elif args.compare_lengths:
        if args.dhfr_csv is None:
            # NOTE: dhfr_entries_cleaned.csv has no equivalent in the packaged
            # data/ directory; this default only applies under --compare-lengths
            # and will not resolve unless --dhfr-csv is passed explicitly.
            dhfr_csv = Path(__file__).parent.parent / "pdb_extraction" / "dhfr_entries_cleaned.csv"
        else:
            dhfr_csv = Path(args.dhfr_csv)
        compare_chain_lengths(str(input_csv), str(dhfr_csv), args.output)
    elif args.mismatch:
        filter_amino_acid_mismatch(str(input_csv), args.output)
    else:
        filter_non_dhfr(str(input_csv), args.output)

