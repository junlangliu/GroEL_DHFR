"""Shared kinetic model equations for DHFR–TMP–GroEL/S."""

from __future__ import annotations

import numpy as np


def diffusion_coefficient(radius_m, temperature_k, viscosity_pa_s, k_b):
    return k_b * temperature_k / (6 * np.pi * viscosity_pa_s * radius_m)


def k_diffusion_limited(d_a, d_b, r_a, r_b, n_a):
    return 4 * np.pi * (d_a + d_b) * (r_a + r_b) * n_a * 1e3


def compute_k_on(all_constants):
    """Diffusion-limited association rate between DHFR and GroEL (uM^-1 s^-1 scale as used in model)."""
    system = all_constants["system"]
    fundamental = all_constants["fundamental"]
    d_dhfr = diffusion_coefficient(
        system["r_dhfr"],
        system["temperature_k"],
        system["viscosity_pa_s"],
        fundamental["k_b"],
    )
    d_groel = diffusion_coefficient(
        system["r_groel"],
        system["temperature_k"],
        system["viscosity_pa_s"],
        fundamental["k_b"],
    )
    return k_diffusion_limited(
        d_dhfr,
        d_groel,
        system["r_dhfr"],
        system["r_groel"],
        fundamental["n_a"],
    )


def co_ratio_apparent(k_oc, k_co, k_d, gr_uM, all_constants, k_on):
    system = all_constants["system"]
    m20i = all_constants["m20i"]
    k_off = k_on * k_d
    return (
        k_oc
        * (
            m20i["k_diss_m20i"]
            + k_on * gr_uM / (k_off + system["k_g"])
            + k_oc
            + k_co
        )
        / (k_co**2 + k_oc * k_co + k_co * m20i["k_diss_m20i"])
    )


def v_m20i(k_oc, k_co, k_d, gr_uM, tmp_uM, dhfr_total_uM, all_constants, k_on):
    """M20I THF flux (uM s^-1) with GroEL rescue."""
    system = all_constants["system"]
    m20i = all_constants["m20i"]
    co = co_ratio_apparent(k_oc, k_co, k_d, gr_uM, all_constants, k_on)

    if tmp_uM <= 0:
        kinetic_term = 1.0
    else:
        k_off = k_on * k_d
        da_dio_ratio = (
            m20i["k_diss_m20i"]
            + m20i["k_diss_m20i"] * co
            + k_on * gr_uM / (k_off + system["k_g"])
        ) / (system["k_ass"] * tmp_uM)
        kinetic_term = da_dio_ratio / (1 + co + da_dio_ratio)

    return (
        m20i["k_cat_m20i"]
        * system["dhf"]
        * dhfr_total_uM
        / (m20i["k_m_m20i"] + system["dhf"])
        * kinetic_term
    )


def v_wt(tmp_uM, dhfr_total_uM, all_constants):
    """WT THF flux (uM s^-1); no GroEL rescue term."""
    system = all_constants["system"]
    wt = all_constants["wt"]
    if tmp_uM <= 0:
        apparent_ratio = 1.0
    else:
        apparent_ratio = wt["k_diss_wt"] / (system["k_ass"] * tmp_uM + wt["k_diss_wt"])
    return wt["k_cat_wt"] * system["dhf"] * dhfr_total_uM * apparent_ratio / (
        wt["k_m_wt"] + system["dhf"]
    )


def v_m20i_no_rescue(tmp_uM, dhfr_total_uM, all_constants):
    """M20I parameters in WT-style form (no GroEL rescue)."""
    system = all_constants["system"]
    m20i = all_constants["m20i"]
    if tmp_uM <= 0:
        apparent_ratio = 1.0
    else:
        apparent_ratio = m20i["k_diss_m20i"] / (
            system["k_ass"] * tmp_uM + m20i["k_diss_m20i"]
        )
    return (
        m20i["k_cat_m20i"]
        * system["dhf"]
        * dhfr_total_uM
        * apparent_ratio
        / (m20i["k_m_m20i"] + system["dhf"])
    )


def build_v_function(k_d, all_constants):
    """Return V(k_oc, k_co) at system GroEL/TMP/DHFR for heatmaps."""
    system = all_constants["system"]
    k_on = compute_k_on(all_constants)

    def v(k_oc, k_co):
        return v_m20i(
            k_oc,
            k_co,
            k_d,
            system["gr"],
            system["tmp"],
            system["d_total"],
            all_constants,
            k_on,
        )

    return v


def build_v_function_wt(all_constants):
    system = all_constants["system"]
    return v_wt(system["tmp"], system["d_total"], all_constants)


def ic50_wt(all_constants):
    """Analytical WT IC50 (uM) for normalized V(TMP)/V(0)."""
    system = all_constants["system"]
    wt = all_constants["wt"]
    return wt["k_diss_wt"] / system["k_ass"]


def ic50_m20i(k_oc, k_co, k_d, gr_uM, all_constants, k_on):
    """Analytical M20I IC50 (uM)."""
    system = all_constants["system"]
    m20i = all_constants["m20i"]
    k_off = k_on * k_d
    co = co_ratio_apparent(k_oc, k_co, k_d, gr_uM, all_constants, k_on)
    c_term = (
        m20i["k_diss_m20i"] * (1.0 + co) + k_on * gr_uM / (k_off + system["k_g"])
    ) / system["k_ass"]
    return c_term / (1.0 + co)


def phase_diagram_metrics(
    k_oc,
    k_co,
    k_d,
    gr_uM,
    all_constants,
    k_on=None,
    v_wt_ref=None,
    ic50_ref=None,
):
    """Return (V/V_WT, IC50_M20I/IC50_WT) at one phase-diagram point.

    Shared by all heatmaps whose color encodes either flux or IC50 rescue.
    """
    system = all_constants["system"]
    if k_on is None:
        k_on = compute_k_on(all_constants)
    if v_wt_ref is None:
        v_wt_ref = build_v_function_wt(all_constants)
    if ic50_ref is None:
        ic50_ref = ic50_wt(all_constants)

    v = v_m20i(
        k_oc,
        k_co,
        k_d,
        gr_uM,
        system["tmp"],
        system["d_total"],
        all_constants,
        k_on,
    )
    ic50 = ic50_m20i(k_oc, k_co, k_d, gr_uM, all_constants, k_on)

    v_norm = v / v_wt_ref if v_wt_ref > 0 else np.nan
    ic50_norm = ic50 / ic50_ref if ic50_ref > 0 else np.nan
    return v_norm, ic50_norm


def find_ic50_crossing(tmp_vals, y_norm_vals, target=0.5):
    """Log-space interpolation where normalized activity crosses target."""
    y = np.asarray(y_norm_vals)
    x = np.asarray(tmp_vals)
    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        return np.nan
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            x0, x1 = x[i], x[i + 1]
            lx0, lx1 = np.log10(x0), np.log10(x1)
            frac = (target - y0) / (y1 - y0)
            return 10 ** (lx0 + frac * (lx1 - lx0))
    return np.nan


def m20i_cpmg_k_co(all_constants):
    cpmg = all_constants["m20i_cpmg"]
    k_oc = cpmg["k_oc_m20i"]
    ratio = (1 - cpmg["p_open_m20i"]) / cpmg["p_open_m20i"]
    return k_oc, ratio * k_oc
