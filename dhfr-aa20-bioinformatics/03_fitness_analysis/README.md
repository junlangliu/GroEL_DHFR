# 03 — Fitness Analysis: TMP-Response Fitness of DHFR Homologs

This module analyzes trimethoprim (TMP) resistance fitness of DHFR homologs
and mutants from the broad mutational scanning (BMS) dataset published by
Romanowicz, Resnick, Hinton, and Plesa (Science Advances 2025,
BioProject PRJNA1189478).

## Source Data (not included — must be downloaded)

All fitness values are computed from article-derived processed data tables
published alongside:

> Romanowicz JM, Resnick SJ, Hinton ER, Plesa C.
> "Exploring Antibiotic Resistance in Diverse Homologs of the Dihydrofolate
> Reductase Protein Family through Broad Mutational Scanning."
> *Science Advances* 2025. DOI: [10.1126/sciadv.adw9178](https://doi.org/10.1126/sciadv.adw9178)

**These original tables are not redistributed here.**
Download them from the PlesaLab GitHub repository:
https://github.com/PlesaLab/DHFR

Specific files required (from the `data/` folder of that repository):

| File | Description |
|------|-------------|
| `Count/D01_collapse_d1.tsv` | lib15 LB overnight barcode counts |
| `Count/D03_collapse_d1.tsv` | lib15 M9 full-supplement baseline |
| `Count/D05_collapse_d1.tsv` | lib15 no-drug complementation (0 µg/mL TMP) — **fitness baseline** |
| `Count/D06_collapse_d1.tsv` | lib15 0.058 µg/mL TMP |
| `Count/D07_collapse_d1.tsv` | lib15 0.5 µg/mL TMP |
| `Count/D08_collapse_d1.tsv` | lib15 1 µg/mL TMP |
| `Count/D09_collapse_d1.tsv` | lib15 10 µg/mL TMP |
| `Count/D10_collapse_d1.tsv` | lib15 50 µg/mL TMP |
| `Count/D11_collapse_d1.tsv` | lib15 200 µg/mL TMP |
| `Count/E01_collapse_d1.tsv` – `E06_collapse_d1.tsv` | lib16 TMP conditions (not used in final figures) |
| `Mapping/map_files_formatted/BCinfo15.csv` | lib15 barcode metadata |
| `Mapping/map_files_formatted/BCs_mutID_15.csv` | lib15 barcode → mutID mapping |
| `Mapping/map_files_formatted/mutIDinfo15.csv` | lib15 variant annotation (IDalign, mutations count, protein sequence) |

Place the downloaded files such that `../../../data_raw/DHFR/DHFR/Count/` and
`../../../data_raw/DHFR/DHFR/Mapping/map_files_formatted/` resolve correctly
relative to each script, or update the `DATA_ROOT` path constant at the top
of each script.

No raw FASTQ reprocessing was performed. Barcode collapsing was taken as-is
from the article-derived tables.

**Fitness definition:** log₂ fold-change of depth-normalized barcode counts
relative to the no-drug M9 baseline (D05, lib15). All analyses use the lib15
(Codon 1) library and `mutations == 0` perfect-assembly homologs unless
otherwise noted.

## Submodule Structure

### `aa20_aa28_joint/` — Joint AA20 × AA28 Counts and TMP-Response Regression

Characterizes the joint distribution of residue identities at positions 20 and
28 across 961 perfect-assembly homologs, and fits per-homolog log₁₀(TMP)
regressions to summarize TMP-response shape.

| File | Description |
|------|-------------|
| `compute_regression_per_point.py` | Fits F = β₀ + β₁·log₁₀(TMP) per homolog; outputs per-IDalign coefficients |
| `plot_aa20_aa28_joint_counts.py` | Heatmap of joint AA20 × AA28 counts |
| `plot_beta_scatter_iqr.py` | Scatter plot of β₀ vs β₁ with IQR error bars, colored by AA20/AA28 combo |
| `aa20_aa28_joint_counts.csv` | Joint count table (961 homologs) |
| `regression_per_idalign_mut0.csv` | Per-homolog β₀/β₁ values |
| `regression_beta_iqr_per_combo_mut0.csv` | Combination-level medians and IQRs |
| `aa20_aa28_joint_counts_heatmap.png` | Output figure |
| `beta_scatter_mut0_beta_iqr_count.png` | β₀/β₁ scatter with IQR |
| `beta_scatter_mut0_count_mono_quantile_labeled.png` | β₀/β₁ scatter with count labels |

### `fitness_ll_subset/` — Fitness Curves for L/L Homologs (AA20=Leu, AA28=Leu)

Plots TMP-fitness curves for the 480 `mutations == 0` homologs with Leu at
both positions 20 and 28, and generates heatmap grids comparing AA20/AA28
combinations.

| File | Description |
|------|-------------|
| `plot_fitness_20L_28L_mut0.py` | Fitness vs TMP line plot for 480 L/L homologs with median curve |
| `plot_beta_example_single.py` | Example single-homolog β₀/β₁ regression illustration |
| `plot_fitness_mut0_pairs_heatmap_layout_2x4.py` | 2×4 heatmap grid of AA20/AA28 pair fitness |
| `plot_fitness_mut0_pairs_heatmap_layout_3x3.py` | 3×3 heatmap grid variant |
| `aa20_aa28_fitness_all.csv` | Fitness table for all homologs with AA20/AA28 annotations (4.8 MB) |
| `fitness_20L28L_mut0.png` | L/L fitness line plot |
| `fitness_20L28L_beta_example_single.{png,svg}` | Example regression illustration |
| `fitness_20L28L_mut0_pairs_grid_heatmap_layout.{png,svg,pdf}` | Final heatmap grid figure |
| `fitness_fitting_example_single.png` | Curve-fitting illustration |

### `iml_violin_trend/` — IML Fitness Distribution and Trend by AA20 State

Violin plots showing TMP-fitness distribution for I, M, L homologs at each
TMP concentration, with PCHIP-interpolated median trends.

| File | Description |
|------|-------------|
| `plot_iml_violin_trend.py` | Violin + jitter + PCHIP median trend plots per AA20 state |
| `aa20_aa28_fitness_lib15.csv` | Fitness table filtered to lib15 I/M/L homologs (107 KB) |
| `fitness_by_aa20_IML_violin_all_tmp_aligned_lib15_trend_aligned.png` | Output figure |

### `mut1_pos20/` — Single-Mutant Analysis at Position 20

Identifies lib15 single-amino-acid mutants whose substitution maps to E. coli
reference position 20, and compares their TMP-fitness curves with their matched
designed reference homologs.

| File | Description |
|------|-------------|
| `plot_mut1_pos20_fitness_facets.py` | Facet panel: mutant vs reference fitness per mutant/reference pair |
| `mut1_pos20_fitness_delta_refmedian_7.csv` | 7 retained mutant/reference pairs with Δfitness |
| `mut1_pos20_fitness_vs_ref_facets_changecolors.png` | Output figure |

Filtering summary: 5,429 `mutations == 1` mutants → 7 with substitution at
E. coli AA20 and complete TMP-fitness data.

### `aa20_iml_bar/` — AA20 Residue Distribution Bar Chart

Bar chart of AA20 identity frequencies across the lib15 perfect-assembly
homolog set, with I/M/L highlighted.

| File | Description |
|------|-------------|
| `plot_aa20_counts_highlight_iml.py` | Bar chart with I/M/L highlighted |
| `aa20_counts_paper_like_zero_mut_lib15_filtered.csv` | Per-homolog AA20 identity table |
| `aa20_counts_highlight_IML_bar.png` | Output figure |

## Dependencies

See top-level `requirements.txt`. Key packages: `pandas`, `numpy`, `scipy`,
`matplotlib`, `seaborn`, `biopython`.
