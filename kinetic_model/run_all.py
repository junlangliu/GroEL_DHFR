#!/usr/bin/env python3
"""Generate all publication PNG figures for the kinetic model."""

from __future__ import annotations

import plot_curve_norm_flux_vs_TMP
import plot_curve_thf_flux_vs_GroEL_ratio
import plot_heatmap_GroEL_kd
import plot_heatmap_kOC_deltaG
import plot_heatmap_kOC_kd


def main():
    plot_heatmap_kOC_deltaG.main()
    plot_heatmap_kOC_kd.main()
    plot_heatmap_GroEL_kd.main()
    plot_curve_thf_flux_vs_GroEL_ratio.main()
    plot_curve_norm_flux_vs_TMP.main()
    print("All figures written to figures/")


if __name__ == "__main__":
    main()
