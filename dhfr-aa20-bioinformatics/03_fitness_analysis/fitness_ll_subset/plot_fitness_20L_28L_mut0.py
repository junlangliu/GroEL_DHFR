#!/usr/bin/env python3
"""
Plot fitness vs TMP for strains with aa20=L and aa28=L (mutations=0).
- 480 IDalign entries; each plotted as a thin line.
- Median curve highlighted; beta0/beta1 from linear fit of median fitness vs log10(TMP) (TMP>0).
Output: fitness_20L28L_mut0.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "aa20_aa28_fitness_all.csv"
OUT = HERE / "fitness_20L28L_mut0.png"

TMP_COLS = [
    ("fitness_0", 0.0),
    ("fitness_0_058", 0.058),
    ("fitness_0_5", 0.5),
    ("fitness_1", 1.0),
    ("fitness_10", 10.0),
    ("fitness_50", 50.0),
    ("fitness_200", 200.0),
]

COLOR = "#1f77b4"  # blue


def melt_subset(df: pd.DataFrame) -> pd.DataFrame:
    fit_cols = [c for c, _ in TMP_COLS]
    long = df[["IDalign"] + fit_cols].melt(
        id_vars=["IDalign"], value_vars=fit_cols, var_name="tmp_col", value_name="fitness"
    )
    tmp_map = dict(TMP_COLS)
    long["TMP"] = long["tmp_col"].map(tmp_map)
    long = long[long["TMP"] > 0].copy()  # drop TMP=0 for log scale
    return long


def compute_beta(med_df: pd.DataFrame):
    x = np.log10(med_df["TMP"].to_numpy())
    y = med_df["fitness"].to_numpy()
    beta1, beta0 = np.polyfit(x, y, 1)  # slope, intercept
    return beta0, beta1


def main():
    df = pd.read_csv(DATA)
    subset = df[(df["aa20"] == "L") & (df["aa28"] == "L") & (df["mutations"] == 0)].copy()
    if subset.empty:
        raise SystemExit("No rows with aa20=L, aa28=L, mutations=0")

    n_ids = subset["IDalign"].nunique()
    long = melt_subset(subset)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # individual lines
    for ida, g in long.groupby("IDalign"):
        ax.plot(g["TMP"], g["fitness"], color=COLOR, alpha=0.06,
                linewidth=0.7, label=None)

    # median curve
    med = long.groupby("TMP", as_index=False)["fitness"].median()
    beta0, beta1 = compute_beta(med)
    ax.plot(med["TMP"], med["fitness"], color=COLOR, linewidth=2.6,
            marker="o", markersize=5.5, zorder=5, label="Median fitness")

    # linear fit in log10 space
    tmp_fine = np.logspace(np.log10(med["TMP"].min()),
                           np.log10(med["TMP"].max()), 200)
    fit_y = beta1 * np.log10(tmp_fine) + beta0
    ax.plot(tmp_fine, fit_y, color=COLOR, linestyle="--", linewidth=1.8,
            alpha=0.85, zorder=4, label="Linear fit (log₁₀ TMP)")

    ax.set_xscale("log")
    ax.set_xlabel("TMP (µg/mL)", fontsize=12, labelpad=5)
    ax.set_ylabel("Fitness (lib15, C3 baseline)", fontsize=12, labelpad=5)
    ax.set_title(f"aa20 = L,  aa28 = L  (mut = 0,  n = {n_ids})",
                 fontsize=13, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5,
                  color="#aaaaaa", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(fontsize=10, frameon=False, loc="lower left")

    # beta annotation in upper-right corner, away from data
    ax.text(0.97, 0.97,
            f"β₀ = {beta0:.2f}\nβ₁ = {beta1:.2f}",
            transform=ax.transAxes, fontsize=10,
            ha="right", va="top", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#cccccc", linewidth=0.7, alpha=0.85))

    fig.tight_layout()
    fig.savefig(OUT, dpi=350, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
