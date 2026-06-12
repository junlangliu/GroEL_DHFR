#!/usr/bin/env python3
"""
Recreate the reference I/M/L violin plot with the *exact* styling of
`fitness_by_aa20_IML_violin_all_tmp_aligned_lib15.png`, then add smooth trend
lines (PCHIP on log10 TMP) without altering the violin shapes.

Inputs
------
- aa20_aa28_fitness_lib15.csv  (lib15, C3‑baseline fitness; already aligned)

Output (new, original is untouched)
-----------------------------------
- fitness_by_aa20_IML_violin_all_tmp_aligned_lib15_trend_aligned.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
DATA = HERE / "aa20_aa28_fitness_lib15.csv"
OUT = HERE / "fitness_by_aa20_IML_violin_all_tmp_aligned_lib15_trend_aligned.png"

# Reference TMPs (same order as the original plot; drop TMP=0)
TMP_ORDER = [0.058, 0.5, 1.0, 10.0, 50.0, 200.0]
TMP_COLS = ["fitness_0_058", "fitness_0_5", "fitness_1", "fitness_10", "fitness_50", "fitness_200"]
TMP_LABELS = [str(t).rstrip("0").rstrip(".") for t in TMP_ORDER]
BASE_POS = np.arange(len(TMP_ORDER))

# Exact palette / transparency used in the published figure (Okabe‑Ito)
PALETTE = {"I": "#0072B2", "M": "#D55E00", "L": "#009E73"}
VIOLIN_ALPHA = 0.50
POINT_ALPHA = 0.40
OFFSETS = {"I": -0.24, "M": 0.0, "L": 0.24}
VIOLIN_WIDTH = 0.24


def prep_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["aa20"].isin(["I", "M", "L"])].copy()
    long = df.melt(
        id_vars=["IDalign", "aa20"],
        value_vars=TMP_COLS,
        var_name="tmp_col",
        value_name="fitness",
    )
    long["TMP"] = long["tmp_col"].str.removeprefix("fitness_").str.replace("_", ".").astype(float)
    long = long[long["TMP"].isin(TMP_ORDER)].dropna(subset=["fitness"])
    long["TMP_cat"] = pd.Categorical(long["TMP"], TMP_ORDER, ordered=True)
    return long


def add_violin_and_points(ax, data: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    handles = []

    for aa, color in PALETTE.items():
        aa_rows = []
        pos = []
        for i, tmp in enumerate(TMP_ORDER):
            vals = data.loc[(data["aa20"] == aa) & (data["TMP"] == tmp), "fitness"].to_numpy()
            aa_rows.append(vals)
            pos.append(BASE_POS[i] + OFFSETS[aa])

        violin = ax.violinplot(
            aa_rows,
            positions=pos,
            widths=VIOLIN_WIDTH,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#2f2f2f")
            body.set_alpha(VIOLIN_ALPHA)
        violin["cmedians"].set_color("#2f2f2f")
        violin["cmedians"].set_linewidth(1.4)

        # jittered points (same spread / alpha as original)
        jitter = VIOLIN_WIDTH * 0.35
        for p, vals in zip(pos, aa_rows):
            if len(vals) == 0:
                continue
            x_j = rng.normal(loc=p, scale=jitter * 0.22, size=len(vals))
            x_j = np.clip(x_j, p - jitter, p + jitter)
            ax.scatter(
                x_j,
                vals,
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.35,
                alpha=POINT_ALPHA,
                s=9,
                zorder=3,
            )
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="#2f2f2f",
                alpha=VIOLIN_ALPHA + 0.15,
                label=f"{aa} (n={int(data[data['aa20']==aa].IDalign.nunique())})",
            )
        )

    ax.set_xticks(BASE_POS)
    ax.set_xticklabels(TMP_LABELS)
    x_min = BASE_POS.min() + min(OFFSETS.values()) - VIOLIN_WIDTH * 1.1
    x_max = BASE_POS.max() + max(OFFSETS.values()) + VIOLIN_WIDTH * 1.1
    ax.set_xlim(x_min, x_max)
    ax.legend(handles=handles, title="AA20", frameon=True, loc="upper right")


def add_smooth_trends(ax, data: pd.DataFrame) -> None:
    x_dense_base = np.linspace(0, len(TMP_ORDER) - 1, 400)
    log_tmp_dense = np.interp(x_dense_base, np.arange(len(TMP_ORDER)), np.log10(TMP_ORDER))

    for aa, color in PALETTE.items():
        med = (
            data[data["aa20"] == aa]
            .groupby("TMP")["fitness"]
            .median()
            .reindex(TMP_ORDER)
            .to_numpy()
        )
        if np.any(np.isnan(med)):
            continue
        spline = PchipInterpolator(np.log10(TMP_ORDER), med)
        y_pred = spline(log_tmp_dense)
        x_dense = x_dense_base + OFFSETS[aa]
        ax.plot(x_dense, y_pred, color=color, linewidth=2.6, zorder=6)
        ax.scatter(
            BASE_POS + OFFSETS[aa],
            med,
            color="white",
            edgecolor=color,
            linewidth=1.4,
            s=60,
            zorder=7,
        )


def main():
    df = pd.read_csv(DATA)
    df["IDalign"] = df["IDalign"].astype(str)
    long = prep_long(df)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6))
    add_violin_and_points(ax, long)
    add_smooth_trends(ax, long)

    ax.set_ylabel("Fitness (lib15, C3 baseline)")
    ax.set_xlabel("TMP concentration (µg/mL)")
    ax.set_title("I/M/L fitness across TMP (lib15, aligned)\nViolin (original style) + smooth trends")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUT, dpi=400)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
