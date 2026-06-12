#!/usr/bin/env python3
"""
Create a single-panel beta-parameter explanation figure.

- NP_764674 fitness vs TMP as points only.
- Solid fitted line in log10(TMP) space (TMP>0).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "aa20_aa28_fitness_all.csv"
OUT = HERE / "fitness_20L28L_beta_example_single.png"
OUT_SVG = HERE / "fitness_20L28L_beta_example_single.svg"

TARGET_ID = "NP_764674"
TMP_POINTS = [
    (0.0, "fitness_0"),
    (0.058, "fitness_0_058"),
    (0.5, "fitness_0_5"),
    (1.0, "fitness_1"),
    (10.0, "fitness_10"),
    (50.0, "fitness_50"),
    (200.0, "fitness_200"),
]


def fit_beta(row: pd.Series) -> tuple[float, float]:
    tmp_vals = np.array([v for v, _ in TMP_POINTS if v > 0], dtype=float)
    fit_cols = [c for v, c in TMP_POINTS if v > 0]
    y = row[fit_cols].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    x_log = np.log10(tmp_vals[mask])
    y_fit = y[mask]
    if y_fit.size < 2:
        raise ValueError("Need at least two TMP points to fit beta0/beta1.")
    beta1, beta0 = np.polyfit(x_log, y_fit, 1)
    return float(beta0), float(beta1)


def main() -> None:
    df = pd.read_csv(DATA)
    id_df = df[df["IDalign"] == TARGET_ID].copy()
    if id_df.empty:
        raise SystemExit(f"No rows found for {TARGET_ID}")

    # Prefer mutations == 0 row to match the mut0 context; otherwise use first row.
    if "mutations" in id_df.columns and (id_df["mutations"] == 0).any():
        row = id_df[id_df["mutations"] == 0].iloc[0]
    else:
        row = id_df.iloc[0]

    beta0, beta1 = fit_beta(row)

    tmp_pos = np.array([v for v, _ in TMP_POINTS if v > 0], dtype=float)
    fit_cols = [c for v, c in TMP_POINTS if v > 0]
    y_pos = row[fit_cols].to_numpy(dtype=float)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    fig, ax0 = plt.subplots(figsize=(5, 5))

    # Data points + dashed connecting line.
    ax0.plot(tmp_pos, y_pos, linestyle="--", color="#7a7a7a", linewidth=1.0, zorder=3)
    ax0.scatter(tmp_pos, y_pos, color="#7a7a7a", s=36, zorder=4)
    tmp_fine = np.logspace(np.log10(tmp_pos.min()), np.log10(tmp_pos.max()), 300)
    fit_line = beta1 * np.log10(tmp_fine) + beta0
    ax0.plot(tmp_fine, fit_line, linestyle="-", color="#1f77b4", linewidth=1.0, zorder=2)

    ax0.set_xscale("log")

    ax0.set_xlabel("[TMP] (\u03bcg ml$^{-1}$)")
    ax0.set_ylabel("Fitness")
    ax0.set_ylim(-6, 0)
    ax0.tick_params(width=0.5)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="--", color="#7a7a7a", linewidth=0.9, markersize=6, label=f"data ({TARGET_ID})"),
        Line2D([0], [0], linestyle="-", color="#1f77b4", linewidth=1.0, label="fitted"),
    ]
    ax0.legend(handles=legend_handles, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=350)
    fig.savefig(OUT_SVG)
    print(f"Saved {OUT}")
    print(f"Saved {OUT_SVG}")


if __name__ == "__main__":
    main()
