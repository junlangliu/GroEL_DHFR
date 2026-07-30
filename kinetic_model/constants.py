"""Constants helper module.

This file centralizes all constants used in the heatmap script so they can be
imported and reused in a single place.
"""

from __future__ import annotations

from typing import Dict, Any


def get_fundamental_constants() -> Dict[str, float]:
    """Physical constants shared by helper equations."""
    return {
        "k_b": 1.380649e-23,       # J/K (Boltzmann constant)
        "n_a": 6.02214076e23,      # mol^-1 (Avogadro's constant)
    }

def get_system_constants() -> Dict[str, float]:
    """System-level physical constants used for diffusion/k_on estimate."""
    return {
        "gr": 2.2,                 # uM, GroEL concentration
        "temperature_k": 310.0,    # K
        "viscosity_pa_s": 3e-3,    # Pa*s, E. coli viscosity
        "r_dhfr": 2e-9,            # m, DHFR radius
        "r_groel": 6.6e-9,         # m, GroEL radius
        "dhf": 10.0,               # uM, DHF concentration
        "d_total": 0.4,            # uM, total DHFR concentration
        "tmp": 88.57,              # uM, total TMP concentration
        "k_ass": 20.0,             # uM^-1 s^-1, association rate between DHFR and TMP
        "k_g": 1.0 / 8.0,          # s^-1, GroEL refolding process rate
    }

def get_m20i_constants() -> Dict[str, float]:
    """Kinetic constants for the mutant model (excluding k_d-dependent terms)."""
    return {
        "k_cat_m20i": 6.8,         # s^-1
        "k_m_m20i": 1.9,           # uM
        "k_diss_m20i": 0.058,            # s^-1, dissociation rate between DHFR and TMP
    }

def get_wt_constants() -> Dict[str, float]:
    """Constants used in `build_v_function_wt`."""
    return {
        "k_cat_wt": 5.3,              # s^-1
        "k_m_wt": 2.2,                # uM
        "k_diss_wt": 0.092,            # s^-1, dissociation rate between DHFR and TMP
    }

def get_m20i_CPMG_constants() -> Dict[str, float]:
    """Constants for the M20I CPMG experiment."""
    return {
        "k_oc_m20i": 316.0,
        "p_open_m20i": 0.962, # CPMG
    }

def get_main_heatmap_grid_constants() -> Dict[str, Any]:
    """Sampling constants for the delta_G heatmap."""
    return {
        "k_oc_min": 0.1,
        "k_oc_max": 1_000_000.0,
        "n_k_oc": 100,
        "ln_ratio_min": "ln(1/99)",
        "ln_ratio_max": "ln(1/9)",
        "n_ln_ratio": 100,
    }

def get_boundary_detection_constants() -> Dict[str, float]:
    """Constants used for derivative-based boundary detection."""
    return {
        "consecutive_n": 10,
        "threshold_fraction": 0.10,
        "smooth_window": 9,
    }

def get_plot_layout_constants() -> Dict[str, Any]:
    """Figure layout constants for heatmap and colorbar axes."""
    return {
        "figsize": (5.5, 5.5),
        "heatmap_pos": [0.12, 0.12, 0.62, 0.62],
        "cbar_pos": [0.80, 0.12, 0.03, 0.62],
        "line_width_frame": 1.0,
        "line_width_boundary": 1.8,
    }

def get_kd_scan_constants() -> Dict[str, float]:
    """Constants for the second heatmap (fixed delta_G, varying k_d)."""
    return {
        "kd_min": 0.01,
        "kd_max": 10.0,
        "n_kd": 100,
        "k_oc_min": 0.1,
        "k_oc_max": 1_000_000.0,
        "n_k_oc": 100,
    }

def get_tmp_scan_constants() -> Dict[str, float]:
    """Constants for the TMP scan experiment. unit: uM"""
    return {
        "tmp_min": 1e-6,
        "tmp_max": 100,
        "n_tmp": 1000,
    }

def get_gr_scan_constants() -> Dict[str, float]:
    """Constants for the GroEL scan experiment. unit: uM"""
    return {
        "gr_min": 0.1,
        "gr_max": 10,
        "n_gr": 100,
    }

def get_all_constants() -> Dict[str, Dict[str, Any]]:
    """Return all grouped constant dictionaries in one call."""
    return {
        "fundamental": get_fundamental_constants(),
        "system": get_system_constants(),
        "m20i": get_m20i_constants(),
        "wt": get_wt_constants(),
        "m20i_cpmg": get_m20i_CPMG_constants(),
        "main_grid": get_main_heatmap_grid_constants(),
        "boundary": get_boundary_detection_constants(),
        "layout": get_plot_layout_constants(),
        "kd_scan": get_kd_scan_constants(),
        "tmp_scan": get_tmp_scan_constants(),
        "gr_scan": get_gr_scan_constants(),
    }