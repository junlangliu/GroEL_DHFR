# DHFR Position-20 Bioinformatics Analysis

Bioinformatics code and processed data for the analysis of residue position 20
(AA20) in dihydrofolate reductase (DHFR) and its relationship to trimethoprim
(TMP) resistance. Part of the `GroEL_DHFR` repository accompanying the
preprint:

> **Chaperonin recognition of protein dynamics drives drug resistance**
> Junlang Liu et al. — bioRxiv (2026)
> https://www.biorxiv.org/content/10.64898/2026.06.03.729952v1

See also [`../kinetic_model/`](../kinetic_model/) for the analytical kinetic
model of GroEL/S rescue used elsewhere in the paper.

---

## Repository Structure

```
dhfr-aa20-bioinformatics/
├── 01_structural_analysis/   # PDB dihedral angles and ligand context at AA20
├── 02_sequence_database/     # Genome-wide folA vs dfrA divergence (Fisher exact + BH-FDR)
├── 03_fitness_analysis/      # TMP-fitness from broad mutational scanning data
└── docs/
    └── methods.md            # Complete bioinformatics methods (supplementary)
```

## Scientific Overview

Residue 20 of *E. coli* DHFR (Met20, UniProt P0ABQ4) sits in a flexible loop
that gates access to the active site. This study characterizes the sequence,
structural, and functional diversity at this position across the DHFR protein
family using three complementary approaches:

| Module | Question | Data Source |
|--------|----------|-------------|
| [01 Structural](01_structural_analysis/) | How do φ/ψ angles and ligand context at AA20 vary with residue identity (M/L/I)? | 591 PDB crystal structures |
| [02 Sequence Database](02_sequence_database/) | Does the natural dfr resistance family already carry the residues — above all M20I — that lab evolution selects, genome-wide across every aligned position? | NCBI AMRFinderPlus (60 dfr representatives) + UniProt (804 chromosomal folA) |
| [03 Fitness](03_fitness_analysis/) | What are the TMP-resistance fitness consequences of M/L/I at AA20 across diverse DHFR backgrounds? | Romanowicz et al. 2025 BMS data |

---

## Source Data (not included — must be downloaded)

### Module 03: BMS barcode fitness data

The fitness analysis (Module 03) uses article-derived processed data tables
from the following paper. **These original tables are not redistributed in this
repository.** Download them directly from the sources below before running any
script in `03_fitness_analysis/`.

> Romanowicz JM, Resnick SJ, Hinton ER, Plesa C.
> "Exploring Antibiotic Resistance in Diverse Homologs of the Dihydrofolate
> Reductase Protein Family through Broad Mutational Scanning."
> *Science Advances* 2025. DOI: [10.1126/sciadv.adw9178](https://doi.org/10.1126/sciadv.adw9178)

**Download locations:**

| Source | URL | Content |
|--------|-----|---------|
| NCBI BioProject | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1189478 | Raw sequencing reads |
| PlesaLab GitHub | https://github.com/PlesaLab/DHFR | Article-derived processed data tables |
| Supplementary PDF | https://doi.org/10.1126/sciadv.adw9178 | Supplementary methods and tables |

**Specific files used from the PlesaLab GitHub repository (`data/` folder):**

| File | Used in | Description |
|------|---------|-------------|
| `Count/D01_collapse_d1.tsv` – `D11_collapse_d1.tsv` | Module 03 | lib15 (Codon 1) per-condition barcode count tables |
| `Count/E01_collapse_d1.tsv` – `E06_collapse_d1.tsv` | Module 03 | lib16 (Codon 2) per-condition barcode count tables |
| `Mapping/map_files_formatted/BCinfo15.csv` | Module 03 | lib15 barcode-to-variant mapping |
| `Mapping/map_files_formatted/BCs_mutID_15.csv` | Module 03 | lib15 barcode-to-mutID mapping |
| `Mapping/map_files_formatted/mutIDinfo15.csv` | Module 03 | lib15 variant annotation (IDalign, mutations, sequence) |
| `Mapping/map_files_formatted/mutIDinfo16.csv` | Module 03 | lib16 variant annotation |

Place the downloaded files in a local `data_raw/DHFR/` directory. Default
paths expected by the scripts are documented in each module's README.

### Module 01: PDB crystal structures

Download mmCIF files for the 591 PDB entries listed in
`01_structural_analysis/data/pdb_aa_list_pos20_591.csv` from
[RCSB PDB](https://www.rcsb.org/) before running `extract_dihedral_angles.py`.

### Module 02: dfr resistance family + chromosomal folA background

The dfr resistance-enzyme set was built from the **NCBI AMRFinderPlus**
reference DB 4.2 (release 2026-05-15.1; `AMRProt.fa` +
`ReferenceGeneCatalog.txt`, downloaded live by the script), filtered to
`subtype=AMR` with `subclass` containing TRIMETHOPRIM and the dfrB clade
excluded, then collapsed to 60 representatives at ≥90% identity.

The chromosomal folA background was queried from **UniProt** across Bacteria
(`gene:folA OR protein_name:"dihydrofolate reductase"`), filtered for label
purity (headers mentioning dfr/trimethoprim dropped), dereplicated with
`cd-hit` (90% identity → safeguard filter against the dfr representatives →
50% identity for broad, non-redundant sampling; 817 final sequences, 804 of
which are distinct from the folA profile MSA).

Both sets are mapped onto a fixed folA profile MSA anchored to E. coli DHFR
(P0ABQ4) coordinates via `mafft --keeplength --add`. See
`02_sequence_database/README.md` for the full protocol and
`02_sequence_database/01_build_sequence_sets.py` for exact commands.

---

## What This Repository Contains

- **Analysis scripts** (Python): all code to reproduce figures and result tables
  from the downloaded source data
- **Result tables** (CSV/TSV): our computed outputs — fitness values, AA
  assignments, regression coefficients, proportion tests, dihedral angles,
  ligand classifications
- **Figures** (PNG/PDF/SVG): final publication-quality panels

---

## Quick Start

```bash
# From the GroEL_DHFR repository root
cd dhfr-aa20-bioinformatics

# Install dependencies
pip install -r requirements.txt

# Reproduce figures directly from included result tables
cd 01_structural_analysis/figures
python plot_summary_panel_v2.py

cd ../../02_sequence_database
python 03_fisher_and_figures.py

cd ../03_fitness_analysis/iml_violin_trend
python plot_iml_violin_trend.py
```

> Scripts that recompute fitness values or extract dihedral angles require
> downloading the source data first (see module READMEs).

---

## Dependencies

Python ≥ 3.9. See `requirements.txt` for the full package list.

External tools (for rerunning the full pipeline from raw inputs):
- [HMMER](http://hmmer.org/) ≥ 3.3 — with Pfam model `PF00186.hmm` (module 01, PDB entry filtering)
- [MAFFT](https://mafft.cbrc.jp/alignment/software/) ≥ 7.5 — for E. coli-anchored alignment (module 02)
- [cd-hit](https://github.com/weizhongli/cdhit) — for folA/dfr sequence dereplication (module 02)

---

## Methods

Full bioinformatics methods (pipeline steps, parameters, filtering criteria)
are in [`docs/methods.md`](docs/methods.md).

---

## Citation

If you use this code or the processed result tables, please cite:
> Liu J, et al. "Chaperonin recognition of protein dynamics drives drug
> resistance." bioRxiv (2026).
> https://www.biorxiv.org/content/10.64898/2026.06.03.729952v1

Also cite the original BMS dataset used in Module 03:
> Romanowicz JM, Resnick SJ, Hinton ER, Plesa C.
> *Science Advances* 2025. DOI: 10.1126/sciadv.adw9178

## License

MIT — see the [`LICENSE`](../LICENSE) file at the repository root.
