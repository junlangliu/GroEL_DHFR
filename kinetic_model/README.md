# Kinetic model of chaperonin-assisted DHFR rescue

Analytical kinetic model of GroEL/S rescue of TMP-inhibited M20I DHFR via
conformational gating of the M20 loop. Code reproduces the publication figures
listed below (PNG only).

Part of the code release for:

> **Chaperonin recognition of protein dynamics drives drug resistance**  
> https://www.biorxiv.org/content/10.64898/2026.06.03.729952v1

This GitHub repository ([junlangliu/GroEL_DHFR](https://github.com/junlangliu/GroEL_DHFR)) hosts the paper’s bioinformatics and modeling code. Experimental raw data and data-processing code are deposited on Zenodo and will be made public upon publication.

## Model

The model is built on five premises:

1. M20I DHFR exists in distinct open and closed states defined by the M20 loop.
2. Open/closed exchange kinetics and equilibrium populations are parameterized
   from NMR CPMG data (`k_OC`, `ΔG_closed-open` / open-state population).
3. TMP can bind both conformations.
4. GroEL selectively recognizes the open state, driving an ATP-dependent
   unfolding–refolding cycle that removes TMP and releases active DHFR.
5. WT DHFR is unrecognized by GroEL under NADPH/TMP conditions because it lacks
   an open state.

A central aim is to ask, across different conditions, whether **open/closed
exchange kinetics** (`k_OC`) or **equilibrium populations** (`ΔG` /
open-state fraction) dominate GroEL recognition and rescue. Phase diagrams
map **THF flux increase** (M20I/WT) and **IC₅₀ increase** (M20I/WT) over
conformational parameters (`k_OC`, `ΔG`) and GroEL–DHFR binding affinity
(`K_d`). The landscape partitions into GroEL-insensitive, kinetic-gating, and
equilibrium-/`K_d`-controlled regimes; CPMG-derived M20I parameters fall in
the kinetic-gating regime.

The two curve figures reproduce experimental observations: the TMP
dose–response under GroEL/S rescue, and M20I activity recovery as a function
of GroEL/S availability.

Shared metric evaluation is in `model.phase_diagram_metrics`. Defaults
(concentrations, catalytic parameters, CPMG-derived M20I values, scan ranges)
are in `constants.py`. Full derivations and parameter justifications are in
the manuscript Methods and Supplementary Table 2.

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 recommended.

## Quick start

```bash
python run_all.py
```

Figures are written to `figures/`. Or run individual scripts:

```bash
python plot_heatmap_kOC_deltaG.py
python plot_heatmap_kOC_kd.py
python plot_heatmap_GroEL_kd.py
python plot_curve_thf_flux_vs_GroEL_ratio.py
python plot_curve_norm_flux_vs_TMP.py
```

## File naming

- heatmaps: `heatmap_{metric}_{x}_vs_{y}.png`
  - `vvwt` = THF flux increase (M20I/WT)
  - `ic50` = IC₅₀ increase (M20I/WT)
- curves: `curve_{y}_vs_{x}.png`

## Publication figures

| Output PNG | Script | Description |
|------------|--------|-------------|
| `figures/heatmap_vvwt_kOC_vs_deltaG.png` | `plot_heatmap_kOC_deltaG.py` | THF flux increase over `k_OC` vs ΔG |
| `figures/heatmap_ic50_kOC_vs_deltaG.png` | `plot_heatmap_kOC_deltaG.py` | IC₅₀ increase over `k_OC` vs ΔG |
| `figures/heatmap_vvwt_kOC_vs_kd.png` | `plot_heatmap_kOC_kd.py` | THF flux increase over `k_OC` vs `K_d` |
| `figures/heatmap_ic50_kOC_vs_kd.png` | `plot_heatmap_kOC_kd.py` | IC₅₀ increase over `k_OC` vs `K_d` |
| `figures/heatmap_ic50_GroEL_vs_kd.png` | `plot_heatmap_GroEL_kd.py` | IC₅₀ increase over [GroEL] vs `K_d` |
| `figures/heatmap_vvwt_GroEL_vs_kd.png` | `plot_heatmap_GroEL_kd.py` | THF flux increase over [GroEL] vs `K_d` |
| `figures/curve_thf_flux_vs_GroEL_DHFR_ratio.png` | `plot_curve_thf_flux_vs_GroEL_ratio.py` | THF flux vs GroEL/DHFR ratio |
| `figures/curve_norm_flux_vs_TMP.png` | `plot_curve_norm_flux_vs_TMP.py` | Normalized dose–response vs [TMP] |

## Package layout

```
kinetic_model/
├── README.md
├── requirements.txt
├── constants.py          # physical / kinetic / grid parameters
├── model.py              # shared rate and IC50 equations
├── plot_utils.py         # shared plotting helpers
├── plot_heatmap_*.py     # one script per heatmap axis pair (both metrics)
├── plot_curve_*.py       # line plots
├── run_all.py
└── figures/              # PNG outputs
```
