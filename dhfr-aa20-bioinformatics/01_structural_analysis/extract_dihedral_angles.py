#!/usr/bin/env python3
"""
Extract dihedral angles (phi and psi) for a specified residue number (label_seq_id)
from mmCIF files.

Inputs:
  - CSV with columns: PDB_ID, Original_Position, Amino_Acid (optionally Chain)
  - Directory of mmCIF files named like: <PDB_ID>.cif

Outputs:
  - dihedral_angles_pos20.csv

Notes:
  - Uses mmCIF atom_site label fields (label_seq_id, label_asym_id) to match MSA numbering.
  - Falls back to auth fields (auth_seq_id, auth_asym_id) if label fields are not available.
  - If chain is not provided (or doesn't match), attempts to infer a chain that
    contains the requested residue number.
  - Computes phi/psi if atoms are present:
      phi: C(i-1)-N(i)-CA(i)-C(i)
      psi: N(i)-CA(i)-C(i)-N(i+1)
"""

import csv
import math
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
from collections import Counter

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.n = 0
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def __iter__(self):
            return iter(self.iterable) if self.iterable else range(self.total)
        
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc} {self.n}/{self.total}", end="", flush=True)
        
        def set_description(self, desc):
            self.desc = desc


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Robust dihedral calculation (degrees) using 4 points.
    Returns np.nan if geometry is degenerate.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p4 = np.asarray(p4, dtype=float)

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # normals to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)

    if n1_norm < 1e-10 or n2_norm < 1e-10 or b2_norm < 1e-10:
        return float("nan")

    n1 /= n1_norm
    n2 /= n2_norm
    b2u = b2 / b2_norm

    x = float(np.dot(n1, n2))
    x = max(-1.0, min(1.0, x))

    # signed angle
    m1 = np.cross(n1, n2)
    y = float(np.dot(m1, b2u))

    angle = math.degrees(math.atan2(y, x))
    return angle


def _get_atom_site_column(d: Dict[str, Any], key: str) -> Optional[List[str]]:
    """Return a mmCIF atom_site column list if present, else None."""
    return d.get(key, None)


def get_chain_entity_id(d: Dict[str, Any], chain_id: str) -> Optional[str]:
    """
    Get the entity_id for a given chain (auth_asym_id).
    Maps chain -> entity_id via _struct_asym.
    """
    try:
        struct_asym_id = d.get("_struct_asym.id", [])
        struct_asym_entity_id = d.get("_struct_asym.entity_id", [])
        
        if not struct_asym_id or not struct_asym_entity_id:
            return None
        
        for i, asym_id in enumerate(struct_asym_id):
            if asym_id == chain_id and i < len(struct_asym_entity_id):
                try:
                    return str(struct_asym_entity_id[i])
                except Exception:
                    continue
        
        return None
    except Exception:
        return None


def get_chain_entity_description(d: Dict[str, Any], chain_id: str) -> Optional[str]:
    """
    Get the entity description for a given chain (auth_asym_id).
    Maps chain -> entity_id via _struct_asym, then gets _entity.pdbx_description.
    """
    try:
        entity_id = get_chain_entity_id(d, chain_id)
        if entity_id is None:
            return None
        
        # Get entity description
        entity_ids = d.get("_entity.id", [])
        entity_descriptions = d.get("_entity.pdbx_description", [])
        
        if not entity_ids or not entity_descriptions:
            return None
        
        for i, eid in enumerate(entity_ids):
            try:
                if str(eid) == str(entity_id) and i < len(entity_descriptions):
                    desc = entity_descriptions[i]
                    return desc if desc and desc != "?" else None
            except Exception:
                continue
        
        return None
    except Exception:
        return None


def get_chain_length(d: Dict[str, Any], chain_id: str) -> Optional[int]:
    """
    Get the full sequence length of a chain using entity-level sequence tables.
    
    This uses _entity_poly_seq which contains the complete polymer sequence,
    not just residues with modeled coordinates. This is more accurate than
    using max(label_seq_id) from _atom_site, which only includes residues
    that have atoms in the structure (and will be shorter if residues are
    disordered/omitted at the termini).
    
    Args:
        d: mmCIF dictionary
        chain_id: auth_asym_id of the chain
        
    Returns:
        Full sequence length of the chain, or None if not found
    """
    try:
        # Get entity_id for this chain
        entity_id = get_chain_entity_id(d, chain_id)
        if not entity_id:
            return None
        
        # Get sequence length from _entity_poly_seq (full sequence, not just modeled)
        # This table contains all residues in the polymer sequence, including
        # those that may be missing from the structure
        entity_poly_seq_entity_id = d.get("_entity_poly_seq.entity_id", [])
        if entity_poly_seq_entity_id:
            # Count residues in this entity
            count = sum(1 for eid in entity_poly_seq_entity_id if str(eid) == entity_id)
            if count > 0:
                return count
        
        # Fallback: try _entity_poly.pdbx_seq_one_letter_code
        entity_poly_entity_id = d.get("_entity_poly.entity_id", [])
        entity_poly_seq = d.get("_entity_poly.pdbx_seq_one_letter_code", [])
        
        for i in range(len(entity_poly_entity_id)):
            if i < len(entity_poly_seq) and str(entity_poly_entity_id[i]) == entity_id:
                seq_str = entity_poly_seq[i]
                if seq_str:
                    # Remove whitespace and count
                    seq_clean = seq_str.replace("\n", "").replace(" ", "").replace("?", "")
                    return len(seq_clean) if seq_clean else None
        
        return None
    except Exception:
        return None


def find_all_chains_with_residue(d: Dict[str, Any], residue_num: int, preferred_chain: Optional[str] = None, use_label_seq: bool = True) -> List[str]:
    """
    Find all chains that contain the requested residue number.
    If use_label_seq=True (default), uses label_seq_id/label_asym_id and does NOT fall back to auth_seq_id.
    If use_label_seq=False, uses auth_seq_id/auth_asym_id.
    If preferred_chain is provided, it will be first in the list if found.
    Returns a list of chain IDs (may be empty).
    """
    # Try label fields first (for MSA numbering)
    label_asym = _get_atom_site_column(d, "_atom_site.label_asym_id")
    label_seq = _get_atom_site_column(d, "_atom_site.label_seq_id")
    
    # Fallback to auth fields
    auth_asym = _get_atom_site_column(d, "_atom_site.auth_asym_id")
    auth_seq = _get_atom_site_column(d, "_atom_site.auth_seq_id")
    
    chains_found = set()
    preferred_found = False
    
    if use_label_seq:
        # Use label_seq_id only - do not fall back to auth_seq_id
        # This ensures we're matching the MSA numbering correctly
        if label_asym is not None and label_seq is not None:
            for i in range(len(label_asym)):
                try:
                    if int(label_seq[i]) == residue_num:
                        chain_id = label_asym[i]
                        # Map to auth_asym_id for consistency
                        if auth_asym is not None and i < len(auth_asym):
                            chain_id = auth_asym[i]
                        chains_found.add(chain_id)
                        if preferred_chain is not None and chain_id == preferred_chain:
                            preferred_found = True
                except Exception:
                    continue
    else:
        # Use auth_seq_id only
        if auth_asym is not None and auth_seq is not None:
            for i in range(len(auth_asym)):
                try:
                    if int(auth_seq[i]) == residue_num:
                        chain_id = auth_asym[i]
                        chains_found.add(chain_id)
                        if preferred_chain is not None and chain_id == preferred_chain:
                            preferred_found = True
                except Exception:
                    continue
    
    # Return list with preferred chain first if it was found
    chains_list = list(chains_found)
    if preferred_chain and preferred_found:
        chains_list.remove(preferred_chain)
        chains_list.insert(0, preferred_chain)
    else:
        chains_list.sort()  # Sort alphabetically if no preference

    return chains_list


def get_residue_amino_acid(d: Dict[str, Any], chain_id: str, residue_num: int) -> Optional[str]:
    """
    Get the amino acid type (3-letter code) for a specific residue.
    Uses label_seq_id/label_comp_id first (for MSA numbering), falls back to auth fields.
    """
    try:
        # Try label fields first
        label_asym = d.get("_atom_site.label_asym_id", [])
        label_seq = d.get("_atom_site.label_seq_id", [])
        label_comp = d.get("_atom_site.label_comp_id", [])
        auth_asym = d.get("_atom_site.auth_asym_id", [])
        
        comp_ids = []
        
        # Try label fields first
        if label_asym and label_seq and label_comp:
            for i in range(len(label_asym)):
                if i >= len(label_seq) or i >= len(label_comp):
                    continue
                try:
                    if int(label_seq[i]) == residue_num:
                        # Check if auth_asym_id matches (to ensure correct chain)
                        if auth_asym and i < len(auth_asym) and auth_asym[i] == chain_id:
                            comp_ids.append(label_comp[i])
                except (ValueError, TypeError):
                    continue
        
        # Fallback to auth fields if label didn't work
        if not comp_ids:
            auth_asym = d.get("_atom_site.auth_asym_id", [])
            auth_seq = d.get("_atom_site.auth_seq_id", [])
            auth_comp = d.get("_atom_site.auth_comp_id", [])
            
            if auth_asym and auth_seq and auth_comp:
                for i in range(len(auth_asym)):
                    if i >= len(auth_seq) or i >= len(auth_comp):
                        continue
                    try:
                        if auth_asym[i] == chain_id and int(auth_seq[i]) == residue_num:
                            comp_ids.append(auth_comp[i])
                    except (ValueError, TypeError):
                        continue
        
        if not comp_ids:
            return None
        
        # Return the most common comp_id (should all be the same, but just in case)
        most_common = Counter(comp_ids).most_common(1)
        return most_common[0][0] if most_common else None
    except Exception:
        return None


def get_neighbor_coords_auth(
    cif_path: Path,
    residue_num: int,
    chain_id: Optional[str] = None
) -> Tuple[List[str], Optional[Dict[int, Dict[str, np.ndarray]]], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Extract coordinates for residues (i-1, i, i+1) using label_seq_id (MSA numbering).
    Maps to auth_seq_id for coordinate extraction.

    Returns:
      (all_chains_list, residues_dict, resolved_chain_id, chain_description, chain_length, found_amino_acid) where:
         - all_chains_list: list of all chains containing the residue
         - residues_dict maps: auth_seq_id -> { atom_name -> coord }
         - resolved_chain_id: the chain used for extraction (first preferred or first found)
         - chain_description: entity description for the resolved chain
         - chain_length: number of residues in the resolved chain
         - found_amino_acid: 3-letter amino acid code found in the structure
       or ([], None, None, None, None, None) if not possible.
    """
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception:
        return ([], None, None, None, None, None)

    # Required columns
    required = [
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.label_atom_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    if not all(k in d for k in required):
        return ([], None, None, None, None, None)

    all_chains = find_all_chains_with_residue(d, residue_num, preferred_chain=chain_id, use_label_seq=True)
    if not all_chains:
        return ([], None, None, None, None, None)
    
    # Map label_seq_id to auth_seq_id for the target residue
    label_seq = d.get("_atom_site.label_seq_id", [])
    auth_seq = d.get("_atom_site.auth_seq_id", [])
    label_asym = d.get("_atom_site.label_asym_id", [])
    auth_asym = d.get("_atom_site.auth_asym_id", [])
    
    # Find auth_seq_id corresponding to label_seq_id = residue_num
    target_auth_seq_ids = set()
    for i in range(len(label_seq)):
        try:
            if int(label_seq[i]) == residue_num:
                if auth_asym and i < len(auth_asym) and auth_asym[i] in all_chains:
                    if auth_seq and i < len(auth_seq):
                        target_auth_seq_ids.add(int(auth_seq[i]))
        except (ValueError, TypeError, IndexError):
            continue
    
    # If no mapping found, try using residue_num directly as auth_seq_id (fallback)
    if not target_auth_seq_ids:
        target_auth_seq_ids.add(residue_num)
    
    # Use the first mapped auth_seq_id (or the original if no mapping)
    target_auth_seq_id = min(target_auth_seq_ids) if target_auth_seq_ids else residue_num

    # Always use entity 1 if available
    # First, find all chains that belong to entity 1 and contain the residue
    resolved_chain = None
    chain_description = None
    
    # Get all chains in entity 1
    struct_asym_id = d.get("_struct_asym.id", [])
    struct_asym_entity_id = d.get("_struct_asym.entity_id", [])
    entity_1_chains = set()
    for i, asym_id in enumerate(struct_asym_id):
        if i < len(struct_asym_entity_id) and str(struct_asym_entity_id[i]) == "1":
            entity_1_chains.add(asym_id)
    
    # Check if any entity 1 chains contain the residue
    for chain_candidate in all_chains:
        if chain_candidate in entity_1_chains:
            resolved_chain = chain_candidate
            chain_description = get_chain_entity_description(d, chain_candidate)
            break
    
    # If no entity 1 chain found among chains with the residue, use the first available chain
    if resolved_chain is None:
        resolved_chain = all_chains[0]
        chain_description = get_chain_entity_description(d, resolved_chain)
    
    # Get chain length
    chain_length = get_chain_length(d, resolved_chain)
    
    # Get the amino acid type for the target residue (using label_seq_id)
    found_amino_acid = get_residue_amino_acid(d, resolved_chain, residue_num)

    asym = d["_atom_site.auth_asym_id"]
    seq  = d["_atom_site.auth_seq_id"]
    atom = d["_atom_site.label_atom_id"]
    xcol = d["_atom_site.Cartn_x"]
    ycol = d["_atom_site.Cartn_y"]
    zcol = d["_atom_site.Cartn_z"]

    # Map neighbors: find auth_seq_id for residue_num-1, residue_num, residue_num+1
    neighbor_auth_seq_ids = {}
    for offset in [-1, 0, 1]:
        neighbor_label_seq = residue_num + offset
        for i in range(len(label_seq)):
            try:
                if int(label_seq[i]) == neighbor_label_seq:
                    if auth_asym and i < len(auth_asym) and auth_asym[i] == resolved_chain:
                        if auth_seq and i < len(auth_seq):
                            neighbor_auth_seq_ids[neighbor_label_seq] = int(auth_seq[i])
                            break
            except (ValueError, TypeError, IndexError):
                continue
    
    # If mapping failed, use direct offset from target_auth_seq_id
    if 0 not in neighbor_auth_seq_ids:
        neighbor_auth_seq_ids[0] = target_auth_seq_id
    if -1 not in neighbor_auth_seq_ids:
        neighbor_auth_seq_ids[-1] = target_auth_seq_id - 1
    if 1 not in neighbor_auth_seq_ids:
        neighbor_auth_seq_ids[1] = target_auth_seq_id + 1
    
    targets = {neighbor_auth_seq_ids.get(-1, target_auth_seq_id - 1), 
               neighbor_auth_seq_ids.get(0, target_auth_seq_id), 
               neighbor_auth_seq_ids.get(1, target_auth_seq_id + 1)}
    residues: Dict[int, Dict[str, np.ndarray]] = {}

    for i in range(len(asym)):
        if asym[i] != resolved_chain:
            continue

        try:
            rnum = int(seq[i])
        except Exception:
            continue

        if rnum not in targets:
            continue

        aname = atom[i]
        try:
            coord = np.array([float(xcol[i]), float(ycol[i]), float(zcol[i])], dtype=float)
        except Exception:
            continue

        residues.setdefault(rnum, {})[aname] = coord

    if not residues:
        return (all_chains, None, resolved_chain, chain_description, chain_length, found_amino_acid)

    return (all_chains, residues, resolved_chain, chain_description, chain_length, found_amino_acid)


def calculate_phi_psi(
    cif_path: Path,
    residue_num: int,
    chain_id: Optional[str] = None,
    expected_amino_acid: Optional[str] = None
) -> Tuple[Optional[float], Optional[float], List[str], Optional[str], Optional[str], Optional[int], Optional[str], str]:
    """
    Calculate phi and psi for residue residue_num using label_seq_id numbering (MSA numbering).

    Args:
        cif_path: Path to mmCIF file
        residue_num: Residue number (label_seq_id, matches CSV Original_Position)
        chain_id: Optional preferred chain ID
        expected_amino_acid: Optional expected amino acid (1-letter or 3-letter code) for validation

    Returns:
      (phi, psi, all_chains_list, resolved_chain_id, chain_description, chain_length, found_amino_acid, status)
    """
    all_chains, residues, resolved_chain, chain_description, chain_length, found_amino_acid = get_neighbor_coords_auth(cif_path, residue_num, chain_id=chain_id)
    if residues is None:
        return (None, None, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, "Residue/chain not found or missing atom_site fields")
    
    # Map residue_num (label_seq_id) to auth_seq_id for residue lookup in residues dict
    # The residues dict uses auth_seq_id as keys
    try:
        d = MMCIF2Dict(str(cif_path))
        label_seq = d.get("_atom_site.label_seq_id", [])
        auth_seq = d.get("_atom_site.auth_seq_id", [])
        auth_asym = d.get("_atom_site.auth_asym_id", [])
        
        target_auth_seq_id = None
        for i in range(len(label_seq)):
            try:
                if int(label_seq[i]) == residue_num:
                    if auth_asym and i < len(auth_asym) and auth_asym[i] == resolved_chain:
                        if auth_seq and i < len(auth_seq):
                            target_auth_seq_id = int(auth_seq[i])
                            break
            except (ValueError, TypeError, IndexError):
                continue
        
        # Fallback: use residue_num directly if mapping failed
        if target_auth_seq_id is None:
            target_auth_seq_id = residue_num
    except Exception:
        target_auth_seq_id = residue_num
    
    # Validate amino acid if expected_amino_acid is provided
    if expected_amino_acid and found_amino_acid:
        # Convert both to uppercase for comparison
        expected_aa = expected_amino_acid.upper().strip()
        found_aa = found_amino_acid.upper().strip()
        
        # Map 1-letter codes to 3-letter codes
        aa_1_to_3 = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        # Convert expected_aa if it's 1-letter
        if len(expected_aa) == 1 and expected_aa in aa_1_to_3:
            expected_aa = aa_1_to_3[expected_aa]
        
        # Normalize MSE (selenomethionine) to MET for comparison
        # MSE is a modified form of MET where selenium replaces sulfur
        if found_aa == "MSE":
            found_aa_normalized = "MET"
        else:
            found_aa_normalized = found_aa
        
        if expected_aa == "MSE":
            expected_aa_normalized = "MET"
        else:
            expected_aa_normalized = expected_aa
        
        # Compare
        if found_aa_normalized != expected_aa_normalized:
            return (None, None, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, 
                   f"Amino acid mismatch: expected {expected_aa}, found {found_aa}")

    if target_auth_seq_id not in residues:
        return (None, None, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, "Target residue not found in extracted neighbor window")

    curr = residues[target_auth_seq_id]
    # backbone atoms required in current residue
    for a in ("N", "CA", "C"):
        if a not in curr:
            return (None, None, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, f"Missing backbone atom {a} in target residue")

    n = curr["N"]
    ca = curr["CA"]
    c = curr["C"]

    # phi needs previous residue C (use auth_seq_id)
    phi = None
    prev_auth_seq = target_auth_seq_id - 1
    if prev_auth_seq in residues and "C" in residues[prev_auth_seq]:
        c_prev = residues[prev_auth_seq]["C"]
        val = calculate_dihedral(c_prev, n, ca, c)
        if not math.isnan(val):
            phi = val

    # psi needs next residue N (use auth_seq_id)
    psi = None
    next_auth_seq = target_auth_seq_id + 1
    if next_auth_seq in residues and "N" in residues[next_auth_seq]:
        n_next = residues[next_auth_seq]["N"]
        val = calculate_dihedral(n, ca, c, n_next)
        if not math.isnan(val):
            psi = val

    if phi is None and psi is None:
        return (None, None, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, "Could not compute phi/psi (missing neighbor atoms)")

    return (phi, psi, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, "Success")


def main(
    csv_path: Optional[str] = None,
    cif_dir: Optional[str] = None,
    test_pdb: Optional[str] = None,
    input_non_dhfr_csv: Optional[str] = None
) -> None:
    # Set up paths
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "msa" / "pdb_aa_list_pos20.csv"
    else:
        csv_path = Path(csv_path)

    if cif_dir is None:
        cif_dir = Path(__file__).parent.parent / "pdb_download" / "cif_files"
    else:
        cif_dir = Path(cif_dir)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    if not cif_dir.exists():
        print(f"Error: CIF directory not found: {cif_dir}")
        return

    if input_non_dhfr_csv:
        print(f"Re-processing entries from: {input_non_dhfr_csv}\n")
        csv_path = Path(input_non_dhfr_csv)
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            entries = list(reader)
        output_path = Path(__file__).parent / "dihedral_angles_pos20_non_dhfr_updated.csv"
    else:
        print(f"Reading positions from: {csv_path}")
        print(f"CIF files directory: {cif_dir}\n")
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            entries = list(reader)
        output_path = Path(__file__).parent / "dihedral_angles_pos20.csv"

    if test_pdb:
        entries = [e for e in entries if e.get("PDB_ID", "").upper() == test_pdb.upper()]
        print(f"Testing with PDB ID: {test_pdb.upper()}\n")

    print(f"Processing {len(entries)} entries...\n")
    fieldnames = ["PDB_ID", "Amino_Acid", "Original_Position", "All_Chains", "Resolved_Chain", "Chain_Description", "Chain_Length", "Found_Amino_Acid", "Phi", "Psi", "Status"]
    
    # Initialize output file with header
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    results = []
    save_interval = 20  # Save every 20 analyses
    
    # Create progress bar
    pbar = tqdm(total=len(entries), desc="Processing", unit="entry")
    
    try:
        for i, entry in enumerate(entries, 1):
            pdb_id = entry.get("PDB_ID", "").upper().strip()
            original_pos = entry.get("Original_Position", "").strip()
            amino_acid = entry.get("Amino_Acid", "").strip()
            chain = entry.get("Chain", "").strip() or None  # optional column

            if not pdb_id or not original_pos:
                pbar.set_description(f"Skipping {pdb_id} (missing PDB_ID or Original_Position)")
                pbar.update(1)
                continue

            # Skip entries with empty or missing amino acid
            if not amino_acid or amino_acid == "-" or amino_acid.strip() == "":
                pbar.set_description(f"Skipping {pdb_id} (missing amino acid)")
                pbar.update(1)
                continue

            cif_path = cif_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                pbar.set_description(f"{pdb_id}: CIF file not found")
                results.append({
                    "PDB_ID": pdb_id,
                    "Amino_Acid": amino_acid,
                    "Original_Position": original_pos,
                    "All_Chains": "",
                    "Resolved_Chain": "",
                    "Chain_Description": "",
                    "Chain_Length": "",
                    "Phi": None,
                    "Psi": None,
                    "Status": "CIF file not found"
                })
                pbar.update(1)
                continue

            try:
                residue_num = int(original_pos)
            except ValueError:
                pbar.set_description(f"{pdb_id}: invalid Original_Position")
                results.append({
                    "PDB_ID": pdb_id,
                    "Amino_Acid": amino_acid,
                    "Original_Position": original_pos,
                    "All_Chains": "",
                    "Resolved_Chain": "",
                    "Chain_Description": "",
                    "Chain_Length": "",
                    "Phi": None,
                    "Psi": None,
                    "Status": "Invalid Original_Position"
                })
                pbar.update(1)
                continue

            pbar.set_description(f"Processing {pdb_id}")

            # Get expected amino acid from CSV (can be 1-letter or 3-letter)
            expected_aa = amino_acid.strip() if amino_acid and amino_acid != "-" else None
            
            phi, psi, all_chains, resolved_chain, chain_description, chain_length, found_amino_acid, status = calculate_phi_psi(
                cif_path, residue_num, chain_id=chain, expected_amino_acid=expected_aa
            )

            all_chains_str = ",".join(all_chains) if all_chains else ""

            # Skip entries with "Residue/chain not found or missing atom_site fields" status
            if status == "Residue/chain not found or missing atom_site fields":
                pbar.set_description(f"{pdb_id}: Skipped (not found)")
                pbar.update(1)
                continue

            # Update progress bar description with result
            if status == "Success":
                phi_str = f"{phi:.2f}" if phi is not None else ""
                psi_str = f"{psi:.2f}" if psi is not None else ""
                pbar.set_description(f"{pdb_id}: Success (Phi={phi_str or 'NA'}, Psi={psi_str or 'NA'})")
            else:
                pbar.set_description(f"{pdb_id}: {status}")

            result = {
                "PDB_ID": pdb_id,
                "Amino_Acid": amino_acid,
                "Original_Position": residue_num,
                "All_Chains": all_chains_str,
                "Resolved_Chain": resolved_chain or "",
                "Chain_Description": chain_description or "",
                "Chain_Length": chain_length if chain_length is not None else "",
                "Found_Amino_Acid": found_amino_acid or "",
                "Phi": f"{phi:.2f}" if phi is not None else None,
                "Psi": f"{psi:.2f}" if psi is not None else None,
                "Status": status
            }
            results.append(result)
            
            # Save every save_interval entries
            if i % save_interval == 0:
                with open(output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerows(results)
                results = []  # Clear written results
            
            pbar.update(1)
    
    finally:
        pbar.close()
    
    # Final save of any remaining results
    if results:
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(results)

    # Count successful entries from the saved file
    successful = 0
    total = 0
    if output_path.exists():
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get("Status") == "Success":
                    successful += 1
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Successful: {successful}/{total}")
    print("=" * 60)


def inspect_chains(cif_path: Path, residue_num: int) -> None:
    """
    Inspect all chains in a CIF file and show which contain the target residue.
    """
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception as e:
        print(f"Error reading CIF file: {e}")
        return
    
    # Get all chains and their entity assignments
    struct_asym_id = d.get("_struct_asym.id", [])
    struct_asym_entity_id = d.get("_struct_asym.entity_id", [])
    entity_id = d.get("_entity.id", [])
    entity_desc = d.get("_entity.pdbx_description", [])
    
    # Map entity_id to description
    entity_map = {}
    for i, eid in enumerate(entity_id):
        if i < len(entity_desc):
            entity_map[str(eid)] = entity_desc[i]
    
    # Get all chains in the structure
    print(f"\nAll chains in structure:")
    print(f"{'Chain':<10} {'Entity ID':<12} {'Entity Description':<60}")
    print(f"{'-'*10} {'-'*12} {'-'*60}")
    
    chain_entity_map = {}
    for i, asym_id in enumerate(struct_asym_id):
        if i < len(struct_asym_entity_id):
            eid = str(struct_asym_entity_id[i])
            desc = entity_map.get(eid, 'N/A')
            chain_entity_map[asym_id] = (eid, desc)
            print(f"{asym_id:<10} {eid:<12} {desc[:58]:<60}")
    
    # Check which chains contain the target residue
    print(f"\nChains containing residue {residue_num}:")
    asym_col = d.get("_atom_site.auth_asym_id", [])
    seq_col = d.get("_atom_site.auth_seq_id", [])
    chains_with_res = {}
    for i in range(len(asym_col)):
        try:
            if int(seq_col[i]) == residue_num:
                chain = asym_col[i]
                if chain not in chains_with_res:
                    eid, desc = chain_entity_map.get(chain, ('?', 'N/A'))
                    chains_with_res[chain] = (eid, desc)
        except:
            pass
    
    if chains_with_res:
        print(f"{'Chain':<10} {'Entity ID':<12} {'Entity Description':<60}")
        print(f"{'-'*10} {'-'*12} {'-'*60}")
        for chain in sorted(chains_with_res.keys()):
            eid, desc = chains_with_res[chain]
            marker = " <-- Entity 1" if eid == "1" else ""
            print(f"{chain:<10} {eid:<12} {desc[:58]:<60}{marker}")
    else:
        print(f"  No chains found containing residue {residue_num}")
    
    # Check entity 1
    entity_1_chains = [ch for ch, (eid, _) in chain_entity_map.items() if eid == "1"]
    entity_1_with_residue = [ch for ch in chains_with_res.keys() if chain_entity_map.get(ch, ('?', ''))[0] == "1"]
    
    print(f"\nEntity 1 analysis:")
    print(f"  All entity 1 chains: {sorted(entity_1_chains) if entity_1_chains else 'None'}")
    print(f"  Entity 1 chains with residue {residue_num}: {sorted(entity_1_with_residue) if entity_1_with_residue else 'None'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract phi/psi dihedral angles for auth_seq_id positions from mmCIF files"
    )
    parser.add_argument(
        "-c", "--csv",
        type=str,
        default=None,
        help="Path to pdb_aa_list_pos20.csv"
    )
    parser.add_argument(
        "-d", "--cif-dir",
        type=str,
        default=None,
        help="Directory containing CIF files"
    )
    parser.add_argument(
        "-t", "--test",
        type=str,
        default=None,
        help="Test with a specific PDB ID (e.g., 1AI9)"
    )
    parser.add_argument(
        "-i", "--input-non-dhfr-csv",
        type=str,
        default=None,
        help="Input CSV for re-processing non-DHFR entries (e.g., dihedral_angles_pos20_non_dhfr.csv)"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        nargs=2,
        metavar=("PDB_ID", "RESIDUE_NUM"),
        help="Inspect chains for a specific PDB ID and residue number (e.g., --inspect 4IXE 25)"
    )

    args = parser.parse_args()
    
    if args.inspect:
        pdb_id, residue_num = args.inspect[0], int(args.inspect[1])
        cif_dir = Path(__file__).parent.parent / "pdb_download" / "cif_files"
        if args.cif_dir:
            cif_dir = Path(args.cif_dir)
        cif_path = cif_dir / f"{pdb_id}.cif"
        if cif_path.exists():
            print(f"Inspecting {pdb_id} for residue {residue_num}:")
            inspect_chains(cif_path, residue_num)
        else:
            print(f"CIF file not found: {cif_path}")
    else:
        main(csv_path=args.csv, cif_dir=args.cif_dir, test_pdb=args.test, input_non_dhfr_csv=args.input_non_dhfr_csv)