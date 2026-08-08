# 02 — Sequence Database Analysis: folA vs dfrA Genome-Wide Divergence

This module tests whether the residues that lab evolution selects at the
*E. coli* DHFR (folA) active site — above all **M20I** — mirror the residues
that naturally horizontally-transferred trimethoprim-resistance enzymes (the
**dfr** family) already carry. It compares per-residue amino-acid composition,
position by position across the full DHFR fold, between:

- **804 chromosomal folA / DHFR** sequences (broad bacterial sampling)
- **60 dfr resistance-enzyme** representatives (dfrB clade excluded)

both mapped onto **E. coli DHFR P0ABQ4** numbering (Met20 = profile column 77).

## Key Result

Position 20 shows the largest effect size of any residue in the enzyme:
**Ile is +48.8 percentage points enriched** in the dfr resistance family
(two-sided Fisher exact p = 2.1×10⁻¹⁸, BH q = 6.2×10⁻¹⁶), while the wild-type
**Met is −25.7 pp depleted** (q = 3.5×10⁻⁶). The single lab-evolved
substitution M20I recapitulates the dominant natural resistance state at that
position. Across all 159 aligned positions, 30 show both a significant loss
and a significant gain (q < 0.05); D27E is the next-largest such swap after
M20I.

## Pipeline (run in order)

| Step | Script | Consumes | Produces |
|---|---|---|---|
| 1 | `01_build_sequence_sets.py` | NCBI AMRFinderPlus DB + UniProt (live download) | `data/00_dfr_all_catalog.faa`, `data/02_dfr_representatives_60.faa` (folA final product is shipped; see script notes) |
| 2 | `02_align_to_ecoli.py` | step-1 outputs + `data/03_folA_profile_msa_134x296.faa` | `data/04_folA804_aligned_to_profile.faa`, `data/04_dfr60_aligned_to_profile.faa` |
| 3 | `03_fisher_and_figures.py` | the two `data/04_*` alignments + profile | `data/05_fisher_results_perresidue.csv`, all figures in `figures/` |

```bash
python 01_build_sequence_sets.py     # optional: re-derives dfr seqs from current DBs
python 02_align_to_ecoli.py          # MAFFT --keeplength --add onto the profile
python 03_fisher_and_figures.py      # Fisher exact + BH-FDR + figures
```

The frozen sequence files in `data/` (`00_*`, `01_*`, `02_*`, `03_*`, `04_*`)
are shipped so step 3 alone regenerates the result table and figures without
re-downloading or re-aligning. Step 1 documents how the dfr set was built;
because sequence databases drift, re-running it re-derives an *equivalent*
(not byte-identical) set from whatever release is current. The folA build
steps are documented in comments in `01_build_sequence_sets.py` (the UniProt
download itself is not re-executed by the script).

## Files

### `data/` — sequence sets and alignments

| File | Description |
|------|-------------|
| `00_dfr_all_catalog.faa` | All 113 dfr entries from the NCBI AMRFinderPlus TRIMETHOPRIM catalog, dfrB excluded (pre-dereplication) |
| `00_folA_broad_derep90_9665.faa` | folA after 90% cd-hit dereplication + dfr-safeguard filter (9,665 seqs); direct input to the 50% clustering in step 1 |
| `01_folA_chromosomal_c50_817.faa` | Chromosomal folA, cd-hit-clustered at 50% identity (817 seqs; 804 are added onto the profile in step 2, 13 overlap it already) |
| `02_dfr_representatives_60.faa` | dfr resistance enzymes, collapsed to 60 representatives at ≥90% identity (dfrB clade excluded) |
| `03_folA_profile_msa_134x296.faa` | 134-row folA profile MSA (296 columns) that defines the coordinate system; contains the P0ABQ4 anchor |
| `04_folA804_aligned_to_profile.faa` | 134 profile rows + 804 added folA rows (951 rows), E. coli-numbered |
| `04_dfr60_aligned_to_profile.faa` | 134 profile rows + 60 added dfr rows (194 rows), E. coli-numbered |
| `05_fisher_results_perresidue.csv` | Every (position, residue) tested: counts and fractions in each family (`n_dfr`, `frac_dfr`, `n_folA`, `frac_folA`), Δ (percentage points), odds ratio, Fisher p, BH q-value, FDR-0.05 significance flag. 2,669 rows |

In the two `04_*_aligned_to_profile.faa` files, rows whose id is *not* one of
the 134 profile ids are the analysis sequences being compared; the profile
rows are carried only to hold the alignment columns.

### Scripts

- `01_build_sequence_sets.py` — builds the dfr and folA sequence sets (see
  pipeline table above).
- `02_align_to_ecoli.py` — projects both families onto the fixed folA profile
  MSA with `mafft --keeplength --add`, anchoring everything to E. coli P0ABQ4
  coordinates.
- `03_fisher_and_figures.py` — two-sided Fisher exact test per (position,
  residue) with Benjamini–Hochberg FDR across all 2,669 tests, then generates
  all three figures below.

### `figures/`

| File | Description |
|------|-------------|
| `folA_vs_dfr_volcano.{png,svg}` | Volcano of Δfraction vs −log₁₀q, every residue×position; labels highlight experimental-hit positions among the top 1% most-enriched / most-depleted sites |
| `folA_vs_dfr_volcano_all_enriched.{png,svg}` | Same volcano with all significantly enriched experimental-hit positions labeled |
| `recap_pairs_perposition.{png,svg}` | For each of the 30 positions with both a significant loss and gain, the dominant lost↔gained residue swap; M20I is the largest single gain, D27E is also highlighted |

## Methods Notes

- **dfr set:** NCBI AMRFinderPlus reference DB 4.2 (release 2026-05-15.1);
  `subtype=AMR`, `subclass` contains TRIMETHOPRIM; **dfrB family excluded**
  (structurally unrelated type-II DHFR). Collapsed to 60 representatives at
  ≥90% identity (BLOSUM62 global alignment).
- **folA set:** chromosomal folA / DHFR from UniProt across Bacteria, filtered
  for label purity (headers mentioning dfr/trimethoprim dropped),
  dereplicated at 90% identity (`cd-hit -c 0.90 -aS 0.8`), a sequence-level
  safeguard removing anything ≥60% identical to any dfr representative, then
  clustered at 50% identity for broad, non-redundant sampling.
- **Alignment:** MAFFT `--keeplength --add` projects each family onto the
  fixed folA profile MSA, preserving E. coli P0ABQ4 coordinates. Met20 =
  profile column 77 (`col_of[20] == 77`, asserted in the script).
- **Statistics:** two-sided Fisher exact per (position, residue) on the 2×2
  table `[[dfr with aa, dfr without], [folA with aa, folA without]]`;
  Benjamini–Hochberg FDR across all 2,669 tests. Δ = fraction_dfr −
  fraction_folA (percentage points).

Full narrative methods (data sources, filtering, alignment coordinates) are
also documented in the top-level [`docs/methods.md`](../docs/methods.md).

## Dependencies

See top-level `requirements.txt`. Key packages: `biopython`, `pandas`,
`numpy`, `scipy`, `statsmodels`, `matplotlib`, `adjustText`.

External tools: `mafft` (≥7.5); `cd-hit` (only needed to re-run the folA
build notes in `01_build_sequence_sets.py`).
