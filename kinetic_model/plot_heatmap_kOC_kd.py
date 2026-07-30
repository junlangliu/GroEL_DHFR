#!/usr/bin/env python3
"""k_OC vs k_d heatmaps:

- heatmap_vvwt_kOC_vs_kd.png (V / V_WT)
- heatmap_ic50_kOC_vs_kd.png (normalized IC50)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker

from constants import get_all_constants
from model import build_v_function_wt, compute_k_on, ic50_wt, phase_diagram_metrics
from plot_utils import (
    FONT_LABEL,
    FONT_TICK,
    add_contours,
    contour_levels_in_range,
    derivative_boundaries,
    heatmap_axes,
    positive_log_limits,
    save_png,
    set_frame_style,
    smooth_boundary_logx,
)


def _smoothed_boundaries(matrix, k_oc_vals, boundary):
    first, second = derivative_boundaries(
        matrix,
        k_oc_vals,
        int(boundary["consecutive_n"]),
        boundary["threshold_fraction"],
    )
    first_raw = first.copy()
    first = smooth_boundary_logx(first, int(boundary["smooth_window"]))
    second = smooth_boundary_logx(second, int(boundary["smooth_window"]))
    first[np.isclose(first_raw, k_oc_vals[0])] = np.nan
    return first, second


def _style_koc_kd_axes(ax, layout):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.set_xticklabels([r"$10^0$", r"$10^2$", r"$10^4$", r"$10^6$"])
    ax.set_xlabel(r"Open-to-closed rate $k_{OC}$ (s$^{-1}$)", fontsize=FONT_LABEL)
    ax.set_ylabel(
        "GroEL$_{14}$-DHFR binding\naffinity Kd ($\\mu$M)", fontsize=FONT_LABEL
    )
    ax.set_box_aspect(1)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(labelsize=FONT_TICK)
    set_frame_style(ax, layout["line_width_frame"])


def _draw_boundaries(ax, first, second, y_vals, layout):
    ax.plot(
        first,
        y_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
    )
    ax.plot(
        second,
        y_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
    )


def main():
    all_constants = get_all_constants()
    system = all_constants["system"]
    kd_scan = all_constants["kd_scan"]
    boundary = all_constants["boundary"]
    layout = all_constants["layout"]
    m20i_cpmg = all_constants["m20i_cpmg"]

    kd_vals = np.logspace(
        np.log10(kd_scan["kd_min"]),
        np.log10(kd_scan["kd_max"]),
        int(kd_scan["n_kd"]),
    )
    k_oc_vals = np.logspace(
        np.log10(kd_scan["k_oc_min"]),
        np.log10(kd_scan["k_oc_max"]),
        int(kd_scan["n_k_oc"]),
    )
    ratio_m20i = (1.0 - m20i_cpmg["p_open_m20i"]) / m20i_cpmg["p_open_m20i"]
    k_on = compute_k_on(all_constants)
    v_wt_ref = build_v_function_wt(all_constants)
    ic50_ref = ic50_wt(all_constants)

    v_norm = np.zeros((len(kd_vals), len(k_oc_vals)))
    ic50_norm = np.zeros((len(kd_vals), len(k_oc_vals)))
    for i, k_d in enumerate(kd_vals):
        for j, k_oc in enumerate(k_oc_vals):
            v_norm[i, j], ic50_norm[i, j] = phase_diagram_metrics(
                k_oc,
                ratio_m20i * k_oc,
                k_d,
                system["gr"],
                all_constants,
                k_on=k_on,
                v_wt_ref=v_wt_ref,
                ic50_ref=ic50_ref,
            )

    first_v, second_v = _smoothed_boundaries(v_norm, k_oc_vals, boundary)
    first_ic50, second_ic50 = _smoothed_boundaries(ic50_norm, k_oc_vals, boundary)

    # --- V / V_WT ---
    vmax_v = max(np.nanmax(v_norm), 1.000001)
    fig = plt.figure(figsize=layout["figsize"])
    ax, cbar_pos = heatmap_axes(fig)
    mesh = ax.pcolormesh(
        k_oc_vals,
        kd_vals,
        v_norm,
        shading="auto",
        cmap="Greens",
        norm=colors.LogNorm(vmin=1.0, vmax=vmax_v),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    _draw_boundaries(ax, first_v, second_v, kd_vals, layout)
    add_contours(
        ax,
        k_oc_vals,
        kd_vals,
        v_norm,
        levels=contour_levels_in_range([2, 5, 20, 100], 1.0, vmax_v),
        place_at_right=True,
    )
    _style_koc_kd_axes(ax, layout)

    cax = fig.add_axes(cbar_pos)
    cb = fig.colorbar(mesh, cax=cax, label="THF flux increase (M20I/WT)")
    cb.ax.yaxis.label.set_size(FONT_LABEL)
    cb.outline.set_linewidth(layout["line_width_frame"])
    cax.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig, "heatmap_vvwt_kOC_vs_kd.png")

    # --- Normalized IC50 ---
    vmin_ic50, vmax_ic50 = positive_log_limits(ic50_norm)
    fig2 = plt.figure(figsize=layout["figsize"])
    ax2, cbar_pos2 = heatmap_axes(fig2)
    mesh2 = ax2.pcolormesh(
        k_oc_vals,
        kd_vals,
        ic50_norm,
        shading="auto",
        cmap="Greens",
        norm=colors.LogNorm(vmin=vmin_ic50, vmax=vmax_ic50),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    _draw_boundaries(ax2, first_ic50, second_ic50, kd_vals, layout)
    levels = contour_levels_in_range([2, 5, 20, 100], vmin_ic50, vmax_ic50)
    if levels:
        add_contours(
            ax2,
            k_oc_vals,
            kd_vals,
            ic50_norm,
            levels=levels,
            place_at_right=True,
        )
    _style_koc_kd_axes(ax2, layout)

    cax2 = fig2.add_axes(cbar_pos2)
    cb2 = fig2.colorbar(mesh2, cax=cax2, label=r"IC$_{50}$ increase (M20I/WT)")
    cb2.ax.yaxis.label.set_size(FONT_LABEL)
    cb2.outline.set_linewidth(layout["line_width_frame"])
    cax2.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig2, "heatmap_ic50_kOC_vs_kd.png")


if __name__ == "__main__":
    main()
