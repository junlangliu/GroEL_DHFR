#!/usr/bin/env python3
"""curve_norm_flux_vs_TMP.png — normalized V(TMP) for WT / M20I (± GroEL rescue)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from constants import get_all_constants
from model import (
    compute_k_on,
    find_ic50_crossing,
    m20i_cpmg_k_co,
    v_m20i,
    v_m20i_no_rescue,
    v_wt,
)
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
    system = all_constants["system"]
    tmp_scan = all_constants["tmp_scan"]
    layout = all_constants["layout"]

    font_label, font_tick = FONT_LABEL, FONT_TICK
    k_d_fixed = 0.1
    k_on = compute_k_on(all_constants)
    k_oc_m20i, k_co_m20i = m20i_cpmg_k_co(all_constants)

    tmp_curve = np.logspace(
        np.log10(tmp_scan["tmp_min"]), np.log10(tmp_scan["tmp_max"]), 400
    )

    v0_m20i = v_m20i(
        k_oc_m20i, k_co_m20i, k_d_fixed, system["gr"], 0.0, system["d_total"], all_constants, k_on
    )
    v0_wt = v_wt(0.0, system["d_total"], all_constants)
    v0_no_rescue = v_m20i_no_rescue(0.0, system["d_total"], all_constants)

    v_m20i_curve = np.array(
        [
            v_m20i(
                k_oc_m20i,
                k_co_m20i,
                k_d_fixed,
                system["gr"],
                tmp_uM,
                system["d_total"],
                all_constants,
                k_on,
            )
            for tmp_uM in tmp_curve
        ]
    )
    v_wt_curve = np.array([v_wt(tmp_uM, system["d_total"], all_constants) for tmp_uM in tmp_curve])
    v_no_rescue = np.array(
        [v_m20i_no_rescue(tmp_uM, system["d_total"], all_constants) for tmp_uM in tmp_curve]
    )

    v_m20i_norm = v_m20i_curve / v0_m20i if v0_m20i > 0 else v_m20i_curve
    v_wt_norm = v_wt_curve / v0_wt if v0_wt > 0 else v_wt_curve
    v_no_rescue_norm = v_no_rescue / v0_no_rescue if v0_no_rescue > 0 else v_no_rescue

    ic50_m20i = find_ic50_crossing(tmp_curve, v_m20i_norm)
    ic50_wt = find_ic50_crossing(tmp_curve, v_wt_norm)
    ic50_no_rescue = find_ic50_crossing(tmp_curve, v_no_rescue_norm)
    print(f"M20I IC50={ic50_m20i:.6g} uM")
    print(f"WT IC50={ic50_wt:.6g} uM")
    print(f"M20I(no rescue) IC50={ic50_no_rescue:.6g} uM")

    def ic50_label(name, ic50_uM):
        if not np.isfinite(ic50_uM):
            return name
        return f"{name} (IC$_{{50}}$ = {1000.0 * ic50_uM:.3g} nM)"

    tmp_nm = 1000.0 * tmp_curve
    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    ax.plot(
        tmp_nm,
        v_wt_norm,
        color="#1f77b4",
        linewidth=2.2,
        label=ic50_label("WT", ic50_wt),
    )
    ax.plot(
        tmp_nm,
        v_m20i_norm,
        color="#d62728",
        linewidth=2.2,
        label=ic50_label("M20I, with GroEL/S rescue", ic50_m20i),
    )
    ax.plot(
        tmp_nm,
        v_no_rescue_norm,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=ic50_label("M20I, without GroEL/S rescue", ic50_no_rescue),
    )
    ax.axhline(0.5, color="gray", linewidth=1.2, linestyle="--", alpha=0.8)

    if np.isfinite(ic50_m20i):
        ax.axvline(1000.0 * ic50_m20i, color="#d62728", linewidth=1.4, linestyle="--", alpha=0.9)
    if np.isfinite(ic50_wt):
        ax.axvline(1000.0 * ic50_wt, color="#1f77b4", linewidth=1.4, linestyle="--", alpha=0.9)
    if np.isfinite(ic50_no_rescue):
        ax.axvline(1000.0 * ic50_no_rescue, color="#d62728", linewidth=1.4, linestyle=":", alpha=0.9)

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 1.45)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-1}$", r"$10^{1}$", r"$10^{3}$", r"$10^{5}$"])
    ax.set_xlabel("[TMP] (nM)", fontsize=font_label)
    ax.set_ylabel("Normalized THF flux (d[THF]/dt)", fontsize=font_label)
    ax.tick_params(labelsize=font_tick)
    set_frame_style(ax, layout["line_width_frame"])
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper center", handlelength=2.4)
    fig.tight_layout()
    save_png(fig, "curve_norm_flux_vs_TMP.png")


if __name__ == "__main__":
    main()
