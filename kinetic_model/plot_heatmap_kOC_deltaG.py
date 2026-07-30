#!/usr/bin/env python3
"""k_OC vs delta_G heatmaps:

- heatmap_vvwt_kOC_vs_deltaG.png (V / V_WT)
- heatmap_ic50_kOC_vs_deltaG.png (normalized IC50)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from constants import get_all_constants
from model import build_v_function_wt, compute_k_on, ic50_wt, phase_diagram_metrics
from plot_utils import (
    FONT_ANNOT,
    FONT_LABEL,
    FONT_TICK,
    add_contours,
    contour_levels_in_range,
    derivative_boundaries,
    heatmap_axes,
    parse_ln_expression,
    positive_log_limits,
    save_png,
    set_frame_style,
    smooth_boundary_logx,
)


def _add_m20i_marker(ax, m20i_cpmg, text_color):
    x_m20i = m20i_cpmg["k_oc_m20i"]
    y_m20i = np.log((1 - m20i_cpmg["p_open_m20i"]) / m20i_cpmg["p_open_m20i"])
    ax.plot(x_m20i, y_m20i, "D", ms=9, mfc="#F5A623", mec="white", mew=1.0, zorder=10)
    ax.text(
        x_m20i * 1.03,
        y_m20i + 0.10,
        "M20I",
        fontsize=FONT_ANNOT,
        color=text_color,
        zorder=11,
    )


def _add_p_open_axis(ax, layout):
    def y_to_p_percent(y):
        ratio = np.exp(y)
        return 100.0 * ratio / (1.0 + ratio)

    def p_percent_to_y(p_percent):
        p = np.clip(np.asarray(p_percent) / 100.0, 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    ax2 = ax.secondary_yaxis("right", functions=(y_to_p_percent, p_percent_to_y))
    ax2.set_ylabel("Open-state population (%)", fontsize=FONT_LABEL)
    ax2.set_yticks([1.0, 2.0, 5.0, 10.0])
    ax2.set_yticklabels(["1.0%", "2.0%", "5.0%", "10.0%"])
    set_frame_style(ax2, layout["line_width_frame"])
    ax2.tick_params(labelsize=FONT_TICK)


def _style_koc_delta_g_axes(ax, layout):
    ax.set_xscale("log")
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.set_xticklabels([r"$10^0$", r"$10^2$", r"$10^4$", r"$10^6$"])
    ax.set_xlabel(r"Open-to-closed rate $k_{OC}$ (s$^{-1}$)", fontsize=FONT_LABEL)
    ax.set_ylabel(r"M20I $\Delta G_{closed-open}$ ($k_B T$)", fontsize=FONT_LABEL)
    ax.set_box_aspect(1)
    set_frame_style(ax, layout["line_width_frame"])
    ax.tick_params(labelsize=FONT_TICK)
    _add_p_open_axis(ax, layout)


def main():
    all_constants = get_all_constants()
    system = all_constants["system"]
    main_grid = all_constants["main_grid"]
    boundary = all_constants["boundary"]
    layout = all_constants["layout"]
    m20i_cpmg = all_constants["m20i_cpmg"]

    k_d_main = 0.1
    k_on = compute_k_on(all_constants)
    v_wt_ref = build_v_function_wt(all_constants)
    ic50_ref = ic50_wt(all_constants)

    k_oc_vals = np.logspace(
        np.log10(main_grid["k_oc_min"]),
        np.log10(main_grid["k_oc_max"]),
        int(main_grid["n_k_oc"]),
    )
    ln_ratio_vals = np.linspace(
        parse_ln_expression(main_grid["ln_ratio_min"]),
        parse_ln_expression(main_grid["ln_ratio_max"]),
        int(main_grid["n_ln_ratio"]),
    )

    v_norm = np.zeros((len(ln_ratio_vals), len(k_oc_vals)))
    ic50_norm = np.zeros((len(ln_ratio_vals), len(k_oc_vals)))
    for i, ln_ratio in enumerate(ln_ratio_vals):
        ratio = np.exp(ln_ratio)
        for j, k_oc in enumerate(k_oc_vals):
            v_norm[i, j], ic50_norm[i, j] = phase_diagram_metrics(
                k_oc,
                ratio * k_oc,
                k_d_main,
                system["gr"],
                all_constants,
                k_on=k_on,
                v_wt_ref=v_wt_ref,
                ic50_ref=ic50_ref,
            )

    first_v, second_v = derivative_boundaries(
        v_norm,
        k_oc_vals,
        int(boundary["consecutive_n"]),
        boundary["threshold_fraction"],
    )
    first_v = smooth_boundary_logx(first_v, int(boundary["smooth_window"]))
    second_v = smooth_boundary_logx(second_v, int(boundary["smooth_window"]))

    first_ic50, second_ic50 = derivative_boundaries(
        ic50_norm,
        k_oc_vals,
        int(boundary["consecutive_n"]),
        boundary["threshold_fraction"],
    )
    first_ic50 = smooth_boundary_logx(first_ic50, int(boundary["smooth_window"]))
    second_ic50 = smooth_boundary_logx(second_ic50, int(boundary["smooth_window"]))

    # --- V / V_WT ---
    vmax_v = max(np.nanmax(v_norm), 1.000001)
    fig = plt.figure(figsize=layout["figsize"])
    ax, cbar_pos = heatmap_axes(fig)
    mesh = ax.pcolormesh(
        k_oc_vals,
        ln_ratio_vals,
        v_norm,
        shading="auto",
        cmap="Purples",
        norm=colors.LogNorm(vmin=1.0, vmax=vmax_v),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    ax.plot(
        first_v,
        ln_ratio_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
    )
    ax.plot(
        second_v,
        ln_ratio_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
    )
    add_contours(
        ax,
        k_oc_vals,
        ln_ratio_vals,
        v_norm,
        levels=contour_levels_in_range([2, 5, 20, 100], 1.0, vmax_v),
        place_at_right=True,
    )
    _add_m20i_marker(ax, m20i_cpmg, text_color="black")
    _style_koc_delta_g_axes(ax, layout)

    cax = fig.add_axes(cbar_pos)
    cb = fig.colorbar(mesh, cax=cax, label="THF flux increase (M20I/WT)")
    cb.ax.yaxis.label.set_size(FONT_LABEL)
    cb.outline.set_linewidth(layout["line_width_frame"])
    cax.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig, "heatmap_vvwt_kOC_vs_deltaG.png")

    # --- Normalized IC50 ---
    vmin_ic50, vmax_ic50 = positive_log_limits(ic50_norm)
    fig2 = plt.figure(figsize=layout["figsize"])
    ax2, cbar_pos2 = heatmap_axes(fig2)
    mesh2 = ax2.pcolormesh(
        k_oc_vals,
        ln_ratio_vals,
        ic50_norm,
        shading="auto",
        cmap="Purples",
        norm=colors.LogNorm(vmin=vmin_ic50, vmax=vmax_ic50),
        edgecolors="face",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    ax2.plot(
        first_ic50,
        ln_ratio_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
        zorder=8,
    )
    ax2.plot(
        second_ic50,
        ln_ratio_vals,
        color="gray",
        linestyle="--",
        linewidth=layout["line_width_boundary"],
        zorder=8,
    )
    levels = contour_levels_in_range([5.0, 20.0, 100.0], vmin_ic50, vmax_ic50)
    if levels:
        add_contours(ax2, k_oc_vals, ln_ratio_vals, ic50_norm, levels=levels)
    _add_m20i_marker(ax2, m20i_cpmg, text_color="white")
    _style_koc_delta_g_axes(ax2, layout)

    cax2 = fig2.add_axes(cbar_pos2)
    cb2 = fig2.colorbar(mesh2, cax=cax2, label=r"IC$_{50}$ increase (M20I/WT)")
    cb2.ax.yaxis.label.set_size(FONT_LABEL)
    cb2.outline.set_linewidth(layout["line_width_frame"])
    cax2.tick_params(width=layout["line_width_frame"], labelsize=FONT_TICK)
    save_png(fig2, "heatmap_ic50_kOC_vs_deltaG.png")


if __name__ == "__main__":
    main()
