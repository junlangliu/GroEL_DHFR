# 02 — Sequence Database Analysis: AA20 Usage in folA vs dfrA Families

This module estimates amino-acid usage at the E. coli DHFR reference position 20
(P0ABQ4, residue 20) in two DHFR-related sequence sets: the broad folA/DHFR
background and the dfrA-preselected acquired-resistance family.

## Overview

Sequences were obtained from UniProtKB (public database, 2024 release) and
filtered through Pfam DHFR domain detection (PF00186). AA20 was mapped to
P0ABQ4 reference coordinates via MAFFT multiple sequence alignment. Proportion
tests were used to assess whether the I/M/L distribution at AA20 differs
significantly between the two families.

**The raw sequence FASTA files and large MSA files are not included** due to
size. The per-sequence count tables and summary statistics produced by our
analysis are included in each subdirectory. To regenerate from scratch, follow
the pipeline described below and in `docs/methods.md`.

**Data sources:**
- UniProtKB: https://www.uniprot.org/ (query: `gene:dfr* OR gene:dhfr* OR protein_name:"trimethoprim-resistant dihydrofolate reductase" OR protein_name:"DfrA family"`)
- Pfam model PF00186: https://www.ebi.ac.uk/interpro/entry/pfam/PF00186/
- Reference sequence P0ABQ4 (*E. coli* DHFR): https://www.uniprot.org/uniprotkb/P0ABQ4

## Pipeline Summary

```
UniProtKB query (gene:dfr* OR gene:dhfr* OR protein_name:...)
        ↓
  ~125,000 raw records → deduplicate → 124,064 unique sequences
        ↓
  HMMER PF00186 domain extraction → 2,331 dfrA-family domains
        ↓
  MAFFT alignment with P0ABQ4 reference
        ↓
  AA20 counting (P0ABQ4 coordinate) per family
        ↓
  Proportion tests (I vs M vs L between folA and dfrA)
```

folA background: Pfam-clean DHFR sequences clustered with MMseqs2 at threshold
0.5 (1,354 representative sequences + P0ABQ4 reference).

## Submodule Structure

### `folA_background/`

Counts AA20 residues in the folA/DHFR background set (MMseqs2 clustered
representatives from Pfam-annotated DHFR sequences, non-dfrA headers).

| File | Description |
|------|-------------|
| `compute_folA_background_p0abq4_aa20.py` | Extracts AA20 from folA MSA, outputs counts |
| `folA_background_full_p0abq4_aa20_counts.csv` | Per-sequence AA20 identity table |
| `folA_background_full_p0abq4_aa20_summary.tsv` | Aggregate counts (L/M/I/other) |

**Input MSA required:** `dhfr_mmseqs_thr0.5_mafft_with_p0abq4.fasta`
(MMseqs2 representatives + P0ABQ4; not included due to size — regenerate from
Pfam DHFR seed + MMseqs2 at threshold 0.5).

### `dfrA_preselected/`

Counts AA20 in the dfrA preselected set (UniProtKB records with `dfrA` in
header, passing PF00186 HMMER filter).

| File | Description |
|------|-------------|
| `compute_dfrA_preselected_p0abq4_aa20.py` | Extracts AA20 from dfrA MSA, outputs counts |
| `dfrA_preselected_p0abq4_aa20_counts.csv` | Per-sequence AA20 identity table |
| `dfrA_preselected_p0abq4_aa20_summary.tsv` | Aggregate counts (L/M/I/other) |
| `dfrA_post_msa_regex_loss.tsv` | Sequences lost during MAFFT header truncation |

**Input MSA required:** `dfrA_plus_folA_ref_mafft.fasta`
(dfrA preselected + P0ABQ4; not included due to size — regenerate from
`dfrA_plus_folA_ref.fasta` via `mafft --auto`).

### `combined_comparison/`

Side-by-side comparison of AA20 in folA vs dfrA, and statistical tests for
I/M/L proportion differences.

| File | Description |
|------|-------------|
| `compute_combined_comparison.py` | Joins folA and dfrA AA20 tables, computes combined summary |
| `compute_iml_proportion_tests.py` | Chi-squared and Fisher's exact tests for I/M/L proportions |
| `plot_aa20_folA_vs_dfrA.py` | Bar chart comparing AA20 usage (P0ABQ4 reference) |
| `aa20_folA_vs_dfrA_P0ABQ4ref.csv` | Combined per-sequence table with family labels |
| `aa20_folA_vs_dfrA_P0ABQ4ref.png` | Output figure |
| `iml_proportion_tests.csv` | Statistical test results |

## Key Result

| Family | L | I | M |
|--------|---|---|---|
| folA | 64.3% (830/1290) | 13.9% (179/1290) | 14.9% (192/1290) |
| dfrA | 60.6% (413/682) | 28.3% (193/682) | 10.3% (70/682) |

The dfrA-preselected family shows a significantly higher proportion of Ile at
position 20 compared with the broad folA background, consistent with Ile-20
being associated with the acquired trimethoprim-resistance phenotype.

## External Tools Required

- **HMMER** (v3.x): `hmmsearch` with Pfam model `PF00186.hmm`
- **MAFFT** (v7.x): `mafft --auto --thread -1`
- **MMseqs2** (optional): for regenerating the folA representative set at
  identity threshold 0.5

## Dependencies

See top-level `requirements.txt`.
