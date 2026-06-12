#!/usr/bin/env python3
"""
Faceted TMP curves (same layout as mut1_pos20_delta_curves_7_facets.png) but
plotting mutant fitness vs its paired wildtype reference.

- Mutant: solid line
- Wildtype (ref): dashed line
- Adds an extra top panel for the E.coli reference curve alone.
- TMP on log scale, one panel per sequence, vertical stacking retained.
Output: mut1_pos20_fitness_vs_ref_facets.png (does NOT overwrite existing plots).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import re


TMP_COLS = [
    ("fitness_0", 0.0),
    ("fitness_0_058", 0.058),
    ("fitness_0_5", 0.5),
    ("fitness_1", 1.0),
    ("fitness_10", 10.0),
    ("fitness_50", 50.0),
    ("fitness_200", 200.0),
]


def parse_change(mut_short: str) -> str:
    m = re.match(r"([A-Z])\d+([A-Z])", str(mut_short))
    if m:
        return f"{m.group(1)}->{m.group(2)}"
    return "unknown"


def load_ecoli_ref_curve(base: Path) -> pd.DataFrame:
    """Build the standalone E.coli reference curve from ref_fitness_* columns."""
    df = pd.read_csv(base / "mut1_pos20_fitness_delta_refmedian_7.csv")
    row = df.iloc[0]
    records = []
    for col, tmp in TMP_COLS:
        ref_col = f"ref_{col}"
        val = row.get(ref_col)
        if pd.isna(val):
            continue
        records.append({"TMP": tmp, "fitness": val})
    ref_df = pd.DataFrame(records)
    ref_df["label"] = "E.coli (WT ref)"
    return ref_df


def main():
    here = Path(__file__).resolve().parent
    df = pd.read_csv(here / "mut1_pos20_fitness_delta_refmedian_7.csv")

    # select columns and melt to long for mutant and ref
    keep_cols = ["mutID", "IDalign"]
    fit_cols = [c for c, _ in TMP_COLS]
    ref_cols = [f"ref_{c}" for c, _ in TMP_COLS]

    mut_long = df[keep_cols + fit_cols].melt(
        id_vars=keep_cols, value_vars=fit_cols, var_name="tmp_col", value_name="fitness_mut"
    )
    ref_long = df[keep_cols + ref_cols].melt(
        id_vars=keep_cols, value_vars=ref_cols, var_name="tmp_col_ref", value_name="fitness_ref"
    )
    # align TMP values
    tmp_map = dict(TMP_COLS)
    mut_long["TMP"] = mut_long["tmp_col"].map(tmp_map)
    ref_long["TMP"] = ref_long["tmp_col_ref"].str.removeprefix("ref_").map(tmp_map)

    long = mut_long.merge(ref_long[["mutID", "IDalign", "TMP", "fitness_ref"]], on=["mutID", "IDalign", "TMP"])
    long = long.dropna(subset=["TMP"])
    long = long[long["TMP"] > 0]  # drop TMP=0 for log scale

    # labels and colors
    long["mutation"] = long["mutID"].apply(lambda s: s.split("_", 1)[1] if "_" in s else s)
    long["mutation_short"] = long["mutation"].str.split("_").str[-1]
    long["change"] = long["mutation_short"].apply(parse_change)
    long["mut_display"] = long["mutation_short"].str.replace(r"(\\d+)", "20", regex=True)
    long["mut_label"] = long["mut_display"] + " (" + long["IDalign"] + ")"

    # fixed color scheme (matches delta facet panel):
    # two M->I same color; two L->M same color; others distinct.
    change_color = {
        "M->I": "#0072B2",
        "L->M": "#D55E00",
        "M->C": "#009E73",
        "M->T": "#CC79A7",
        "L->P": "#F0E442",
    }

    # order panels by TMP=10 mutant fitness median (desc)
    order_metric = (
        long[long["TMP"] == 10].groupby("mutID")["fitness_mut"].median()
        .reindex(long["mutID"].unique(), fill_value=None)
    )
    order_metric = order_metric.fillna(long.groupby("mutID")["fitness_mut"].median())
    mut_order = order_metric.sort_values(ascending=False).index.tolist()
    # keep two M->I mutants adjacent
    change_map = long.set_index("mutID")["change"].to_dict()
    m2i = [m for m in mut_order if change_map.get(m) == "M->I"]
    others = [m for m in mut_order if change_map.get(m) != "M->I"]
    order = m2i + others

    ecoli_ref = load_ecoli_ref_curve(here)

    sns.set_theme(style="whitegrid", context="talk")
    nrow = len(order) + 1  # extra panel for E.coli ref
    fig, axes = plt.subplots(nrow, 1, figsize=(8, 2.5 * nrow), sharex=True)
    if nrow == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # global y-limits across mutant, ref, and E.coli ref
    y_min = min(long["fitness_mut"].min(), long["fitness_ref"].min(), ecoli_ref["fitness"].min())
    y_max = max(long["fitness_mut"].max(), long["fitness_ref"].max(), ecoli_ref["fitness"].max())
    y_pad = 0.12 * (y_max - y_min if y_max != y_min else 1)

    # top panel: E.coli reference alone
    ax_ref = axes[0]
    g = ecoli_ref.sort_values("TMP")
    ax_ref.plot(
        g["TMP"],
        g["fitness"],
        color="#4b4b4b",
        linestyle="--",
        linewidth=2.4,
        marker="o",
        markersize=5.2,
        alpha=0.95,
        label="E.coli ref",
    )
    ax_ref.set_title("E.coli (WT reference)", fontsize=12.5)
    ax_ref.axhline(0, color="#888", linestyle="--", linewidth=0.9)
    ax_ref.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_ref.set_ylabel("Fitness")
    ax_ref.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_ref.legend(fontsize=8.5, loc="best")

    # remaining panels: mutant vs paired ref
    for ax, mutid in zip(axes[1:], order):
        g = long[long["mutID"] == mutid].sort_values("TMP")
        color = change_color.get(g["change"].iloc[0], "#555555")
        ax.plot(
            g["TMP"],
            g["fitness_ref"],
            color="#4b4b4b",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            alpha=0.9,
            label="WT ref",
        )
        ax.plot(
            g["TMP"],
            g["fitness_mut"],
            color=color,
            linestyle="-",
            linewidth=2.4,
            marker="o",
            markersize=5.2,
            alpha=0.95,
            label="Mutant",
        )
        ax.set_title(g["mut_label"].iloc[0], fontsize=12.5)
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.9)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel("Fitness")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.legend(fontsize=8.5, loc="best")

    for ax in axes:
        ax.set_xscale("log")
    axes[-1].set_xlabel("TMP (µg/mL, log scale)")

    fig.tight_layout()
    out = here / "mut1_pos20_fitness_vs_ref_facets_changecolors.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
