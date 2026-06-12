#!/usr/bin/env python3
"""
Scatter of beta0 vs beta1 per (aa20, aa28) combo with IQR error bars.

Data source: regression_per_idalign_mut0.csv (per-ID fits).
We aggregate beta0/beta1 distribution per combo, keep combos with count >= MIN_COUNT,
and plot median points with horizontal/vertical error bars spanning 25th–75th percentiles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe


def main() -> None:
    here = Path(__file__).resolve().parent
    per_id_path = here / "regression_per_idalign_mut0.csv"
    if not per_id_path.exists():
        raise SystemExit(f"missing {per_id_path}")

    df = pd.read_csv(per_id_path)
    if df.empty:
        raise SystemExit("no data in regression_per_idalign_mut0.csv")

    MIN_COUNT = 10  # only show combos with at least this many IDaligns

    rows = []
    for (aa20, aa28), g in df.groupby(["aa20", "aa28"]):
        beta0_q = g["beta0"].quantile([0.25, 0.5, 0.75])
        beta1_q = g["beta1"].quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "aa20": aa20,
                "aa28": aa28,
                "count": len(g),
                "beta0_q25": beta0_q.loc[0.25],
                "beta0_q50": beta0_q.loc[0.5],
                "beta0_q75": beta0_q.loc[0.75],
                "beta1_q25": beta1_q.loc[0.25],
                "beta1_q50": beta1_q.loc[0.5],
                "beta1_q75": beta1_q.loc[0.75],
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = here / "regression_beta_iqr_per_combo_mut0.csv"
    summary.to_csv(summary_path, index=False)

    plot_df = summary[summary["count"] >= MIN_COUNT].copy()
    if plot_df.empty:
        print(f"No combos with count >= {MIN_COUNT}")
        return

    # Monochrome palette with higher contrast (light→deep navy)
    vmin = MIN_COUNT
    vmax = plot_df["count"].max()
    norm = colors.PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
    cmap = colors.LinearSegmentedColormap.from_list(
        "mono_navy",
        ["#edf2fb", "#c0d1f0", "#7a9ad9", "#2f5ea8", "#0b3c77"],
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 6.8))

    # offsets for labels relative to data range
    x_span = plot_df["beta1_q75"].max() - plot_df["beta1_q25"].min()
    dx = 0.014 * x_span if x_span > 0 else 0.01
    y_span = plot_df["beta0_q75"].max() - plot_df["beta0_q25"].min()
    dy = 0.012 * y_span if y_span > 0 else 0.01
    for _, r in plot_df.iterrows():
        color = cmap(norm(r["count"]))
        # symmetric error distances
        xerr = np.array([[r["beta1_q50"] - r["beta1_q25"]], [r["beta1_q75"] - r["beta1_q50"]]])
        yerr = np.array([[r["beta0_q50"] - r["beta0_q25"]], [r["beta0_q75"] - r["beta0_q50"]]])
        ax.errorbar(
            r["beta1_q50"],
            r["beta0_q50"],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=7.2,
            mfc=color,
            mec="#1a1a1a",
            mew=0.45,
            ecolor=color,
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.0,
            alpha=0.95,
            zorder=3,
        )
        ax.text(
            r["beta1_q50"] + dx,
            r["beta0_q50"] + dy * 0.5 * ((-1) ** (_ % 2)),  # stagger up/down
            f"{r['aa20']}/{r['aa28']}",
            fontsize=10,
            ha="left",
            va="center",
            color="#1b1b1b",
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )

    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("beta1 (slope vs log10 TMP)")
    ax.set_ylabel("beta0 (baseline at low TMP)")
    ax.set_title("AA20/AA28 combos (mut=0): median ± IQR, color = count (>=10)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("count (IDalign)")

    fig.tight_layout()
    out_path = here / "beta_scatter_mut0_beta_iqr_count.png"
    fig.savefig(out_path, dpi=350)
    print(f"Saved {out_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
