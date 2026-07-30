#!/usr/bin/env python3
"""GroEL × k_d heatmaps:

- heatmap_ic50_GroEL_vs_kd.png (normalized IC50)
- heatmap_vvwt_GroEL_vs_kd.png (V / V_WT)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from constants import get_all_constants
from model import (
    build_v_function_wt,
    compute_k_on,
    ic50_wt,
    m20i_cpmg_k_co,
    phase_diagram_metrics,
)
from plot_utils import (
    FONT_LABEL,
    FONT_TICK,
    add_contours,
    contour_levels_in_range,
    heatmap_axes,
    positive_log_limits,
    save_png,
    set_frame_style,
)


def main():
    all_constants = get_all_constants()
    gr_scan = all_constants["gr_scan"]
    kd_scan = all_constants["kd_scan"]
    layout = all_constants["layout"]

    k_oc_fixed, k_co_fixed = m20i_cpmg_k_co(all_constants)
    k_on = compute_k_on(all_constants)
    v_wt_ref = build_v_function_wt(all_constants)
    ic50_ref = ic50_wt(all_constants)

    gr_vals = np.logspace(
        np.log10(gr_scan["gr_min"]),
        np.log10(gr_scan["gr_max"]),
        int(gr_scan["n_gr"]),
    )
    kd_vals = np.logspace(
        np.log10(kd_scan["kd_min"]),
        np.log10(kd_scan["kd_max"]),
        int(kd_scan["n_kd"]),
    )

    v_norm = np.zeros((len(kd_vals), len(gr_vals)))
    ic50_norm = np.zeros((len(kd_vals), len(gr_vals)))
    for i, k_d in enumerate(kd_vals):
        for j, gr in enumerate(gr_vals):
            v_norm[i, j], ic50_norm[i, j] = phase_diagram_metrics(
                k_oc_fixed,
                k_co_fixed,
                k_d,
                gr,
                all_constants,
                k_on=k_on,
                v_wt_ref=v_wt_ref,
                ic50_ref=ic50_ref,
            )

    # --- Normalized IC50 ---
    vmin_ic50, vmax_ic50 = positive_log_limits(ic50_norm)
    fig = plt.figure(figsize=layout["figsize"])
    ax, cbar_pos = heatmap_axes(fig)
    mesh = ax.pcolormesh(
        gr_vals,
        kd_vals,
        ic50_norm,
        shading="auto",
        cmap="Oranges",
        norm=colors.LogNorm(vmin=vmin_ic50, vmax=vmax_ic50),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    levels = contour_levels_in_range([2.0, 5.0, 20.0, 100.0], vmin_ic50, vmax_ic50)
    if levels:
        add_contours(ax, gr_vals, kd_vals, ic50_norm, levels=levels)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"[GroEL$_{14}$] ($\mu$M)", fontsize=FONT_LABEL)
    ax.set_ylabel(
        "GroEL$_{14}$-DHFR binding\naffinity Kd ($\\mu$M)", fontsize=FONT_LABEL
    )
    ax.set_box_aspect(1)
    ax.minorticks_off()
    ax.tick_params(labelsize=FONT_TICK)
    set_frame_style(ax, layout["line_width_frame"])

    cax = fig.add_axes(cbar_pos)
    cb = fig.colorbar(mesh, cax=cax, label=r"IC$_{50}$ increase (M20I/WT)")
    cb.ax.yaxis.label.set_size(FONT_LABEL)
    cb.outline.set_linewidth(layout["line_width_frame"])
    cax.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig, "heatmap_ic50_GroEL_vs_kd.png")

    # --- V / V_WT ---
    vmax_v = max(np.nanmax(v_norm), 1.000001)
    fig2 = plt.figure(figsize=layout["figsize"])
    ax2, cbar_pos2 = heatmap_axes(fig2)
    mesh_v = ax2.pcolormesh(
        gr_vals,
        kd_vals,
        v_norm,
        shading="auto",
        cmap="Blues",
        norm=colors.LogNorm(vmin=1.0, vmax=vmax_v),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    v_levels = contour_levels_in_range([2.0, 5.0, 20.0, 100.0], 1.0, vmax_v)
    if v_levels:
        add_contours(ax2, gr_vals, kd_vals, v_norm, levels=v_levels)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"[GroEL$_{14}$] ($\mu$M)", fontsize=FONT_LABEL)
    ax2.set_ylabel(
        "GroEL$_{14}$-DHFR binding\naffinity Kd ($\\mu$M)", fontsize=FONT_LABEL
    )
    ax2.set_box_aspect(1)
    ax2.minorticks_off()
    ax2.tick_params(labelsize=FONT_TICK)
    set_frame_style(ax2, layout["line_width_frame"])

    cax_v = fig2.add_axes(cbar_pos2)
    cb_v = fig2.colorbar(mesh_v, cax=cax_v, label="THF flux increase (M20I/WT)")
    cb_v.ax.yaxis.label.set_size(FONT_LABEL)
    cb_v.outline.set_linewidth(layout["line_width_frame"])
    cax_v.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig2, "heatmap_vvwt_GroEL_vs_kd.png")


if __name__ == "__main__":
    main()
