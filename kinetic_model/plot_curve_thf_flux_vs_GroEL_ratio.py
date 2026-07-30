#!/usr/bin/env python3
"""curve_thf_flux_vs_GroEL_DHFR_ratio.png — THF flux vs GroEL/DHFR ratio."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from constants import get_all_constants
from model import compute_k_on, m20i_cpmg_k_co, v_m20i, v_m20i_no_rescue, v_wt
from plot_utils import (
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    LINE_FIGSIZE,
    save_png,
    set_frame_style,
)


def main():
    all_constants = get_all_constants()
    layout = all_constants["layout"]

    font_label, font_tick = FONT_LABEL, FONT_TICK
    k_d_fixed = 0.03
    tmp_fixed = 1.0  # uM
    dhfr_total_uM = 0.03  # 30 nM
    k_oc_fixed, k_co_fixed = m20i_cpmg_k_co(all_constants)
    k_on = compute_k_on(all_constants)

    gr_dhfr_ratio = np.array([0.0, 8.0, 80.0], dtype=float)
    gr_vals = gr_dhfr_ratio * dhfr_total_uM
    plot_x_ratio = np.where(gr_dhfr_ratio == 0.0, 0.1, gr_dhfr_ratio)

    v_m20i_vals = np.zeros_like(gr_vals)
    v_m20i_vals[0] = v_m20i_no_rescue(tmp_fixed, dhfr_total_uM, all_constants)
    for idx in range(1, len(gr_vals)):
        v_m20i_vals[idx] = v_m20i(
            k_oc_fixed,
            k_co_fixed,
            k_d_fixed,
            gr_vals[idx],
            tmp_fixed,
            dhfr_total_uM,
            all_constants,
            k_on,
        )

    v_wt_val = v_wt(tmp_fixed, dhfr_total_uM, all_constants)
    v_wt_vals = np.full_like(gr_vals, v_wt_val, dtype=float)

    # uM s^-1 -> nM s^-1
    v_m20i_nm = 1000.0 * v_m20i_vals
    v_wt_nm = 1000.0 * v_wt_vals

    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    ax.plot(
        plot_x_ratio,
        v_m20i_nm,
        color="#d62728",
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="M20I",
    )
    ax.plot(
        plot_x_ratio,
        v_wt_nm,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="WT",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.08, 100.0)
    ax.set_xticks([0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_xlabel("GroEL/S₁₄ : DHFR molar ratio", fontsize=font_label)
    ax.set_ylabel(r"THF flux (nM s$^{-1}$) at [TMP]=1000nM", fontsize=font_label)
    ax.set_ylim(0.0, 20.0)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.minorticks_off()
    ax.tick_params(labelsize=font_tick)
    set_frame_style(ax, layout["line_width_frame"])
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    save_png(fig, "curve_thf_flux_vs_GroEL_DHFR_ratio.png")


if __name__ == "__main__":
    main()
