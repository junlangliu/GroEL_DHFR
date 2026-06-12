# 01 — Structural Analysis: DHFR Position 20 in PDB Structures

This module extracts and analyzes dihedral angles (φ/ψ) and ligand context at
residue position 20 (AA20) from DHFR crystal structures deposited in the
Protein Data Bank (PDB).

## Overview

591 DHFR-family PDB entries were identified by HMMER search against Pfam model
PF00186, then filtered for presence of an annotated residue at the E. coli
DHFR reference position 20 (Met20 in E. coli). Dihedral angles were extracted
directly from mmCIF files. Ligands were classified as cofactors (NADPH/NADP⁺)
or substrates/antifolates (DHF, THF, MTX, etc.) from RCSB ligand records.

## Input Data (`data/`)

| File | Description |
|------|-------------|
| `pdb_aa_list_pos20_591.csv` | 591 PDB entries with AA identity, chain, dihedral angles, and status |
| `dihedral_angles_pos20.csv` | Extracted φ/ψ angles per structure |
| `ligand_classification.csv` | Per-structure ligand category assignments |
| `dhfr_sequences_hmmer_aligned.fasta` | HMMER-aligned DHFR domain sequences used for MSA |
| `cofactor_other_chemicals.csv` | Reference list of cofactor ligand codes |
| `substrate_other_chemicals.csv` | Reference list of substrate/antibiotic ligand codes |

> **Source data not included.** PDB mmCIF files are not redistributed here.
> Download each structure listed in `data/pdb_aa_list_pos20_591.csv` from
> [RCSB PDB](https://www.rcsb.org/) (e.g., via
> `wget https://files.rcsb.org/download/<PDB_ID>.cif`).
> The `dihedral_angles_pos20.csv` and `ligand_classification.csv` tables
> in `data/` are our computed outputs and are already included.

## Scripts

### Processing pipeline (run in order)

1. **`extract_dihedral_angles.py`** — Extracts φ/ψ angles at the target residue
   from mmCIF files. Takes a CSV with `PDB_ID`, `Original_Position`,
   `Amino_Acid` columns and a directory of downloaded `.cif` files. Outputs
   `dihedral_angles_pos20.csv`.

2. **`classify_ligands.py`** — Classifies ligands in each structure as cofactor,
   substrate/antibiotic, or other. Uses the reference lists in `data/`. Outputs
   `ligand_classification.csv`.

3. **`filter_pdb_entries.py`** — Applies quality and residue-type filters to the
   raw PDB entry list before downstream analysis.

4. **`analyze_met_psi.py`** — Computes summary statistics for ψ-angle
   distributions stratified by AA20 identity (Met/Leu/Ile).

5. **`compare_ligand_pair_psi.py`** — Tests pairwise differences in ψ-angle
   distributions between ligand categories using KS and Mann-Whitney tests.

### Visualization scripts

- **`plot_ramachandran.py`** — 2D Ramachandran plot and 1D φ/ψ distributions
  for M, L, I residues only.
- **`plot_summary_panel.py`** — Combined multi-panel figure (species pie chart,
  AA bar, Ramachandran, ψ violins). Predecessor to the final panel scripts.

### Final figure scripts (`figures/`)

| Script | Output |
|--------|--------|
| `plot_ab_updated.py` | Species pie chart (panel A) + AA bar chart (panel B) |
| `plot_domain_panels.py` | Domain-level multi-panel layout |
| `plot_ligand_distribution.py` | Ligand category distribution bar chart |
| `plot_psi_by_ligand_category.py` | ψ-angle violin plots by ligand category |
| `plot_summary_panel_v2.py` | Final combined summary panel (panels C/D) |

## Output Figures (`figures/`)

| File | Description |
|------|-------------|
| `summary_panel_v2.{png,pdf}` | Full summary panel (all organisms) |
| `summary_panel_eukaryotes_v2.{png,pdf}` | Summary panel — eukaryotes only |
| `summary_panel_prokaryotes_v2.{png,pdf}` | Summary panel — prokaryotes only |

## Dependencies

See top-level `requirements.txt`. Key packages: `biopython`, `numpy`,
`matplotlib`, `scipy`.
