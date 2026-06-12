#!/usr/bin/env python3
"""
Plot fitness vs TMP (mutations=0) in an 8-panel layout (3×3 minus the empty
bottom-right) whose cell order matches the top-left block of
aa20_aa28_joint_counts_heatmap.png (heatmap row/column scan, omitting the
ninth corner cell):

  Same pivot (aa20 × aa28) and same row/col ordering by marginal totals.

Each panel shows all homolog traces for one (aa20, aa28); color is unique per
panel. No median highlight and no regression fit.

Output:
  fitness_20L28L_mut0_pairs_grid_heatmap_layout.png
  fitness_20L28L_mut0_pairs_grid_heatmap_layout.svg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "aa20_aa28_fitness_all.csv"
JOINT_COUNTS = HERE.parent.parent / "aa20_aa28_final" / "aa20_aa28_joint_counts.csv"
OUT_PNG = HERE / "fitness_20L28L_mut0_pairs_grid_heatmap_layout.png"
OUT_SVG = HERE / "fitness_20L28L_mut0_pairs_grid_heatmap_layout.svg"
OUT_PDF = HERE / "fitness_20L28L_mut0_pairs_grid_heatmap_layout.pdf"

TMP_COLS = [
    ("fitness_0", 0.0),
    ("fitness_0_058", 0.058),
    ("fitness_0_5", 0.5),
    ("fitness_1", 1.0),
    ("fitness_10", 10.0),
    ("fitness_50", 50.0),
    ("fitness_200", 200.0),
]

GRID = 3
# Skip bottom-right panel; show first 8 cells of heatmap top-left 3×3 in scan order.
CELLS = [(r, c) for r in range(GRID) for c in range(GRID) if not (r == GRID - 1 and c == GRID - 1)]
# Mathtext superscript; rm font set to Arial so digits/exponent match sans body text.
XLABEL = "[TMP] (\u03bcg ml$^{-1}$)"

FIT_COLS_TMP_GT0 = [c for c, v in TMP_COLS if v > 0]

# Line colors overridden from tab10. M/M uses a hue outside tab10 (avoids reds like tab10 #3).
PAIR_LINE_COLOR = {
    ("M", "M"): "#7c3aed",  # vivid violet — not in tab10; avoids red/orange/blue confusion with other panels
    ("I", "K"): "#0081a7",  # strong azure (tab10 blue alone can be faint at low alpha)
    ("L", "F"): "#2e7d32",  # deep forest green — more saturated than tab10 green
    ("I", "L"): "#c2185b",  # deep rose/magenta — distinct from red, orange, pink
    ("I", "M"): "#00838f",  # deep teal — not used elsewhere
}


def heatmap_order_matrix(df_counts: pd.DataFrame) -> pd.DataFrame:
    matrix = df_counts.pivot_table(
        index="aa20", columns="aa28", values="count", fill_value=0
    ).astype(int)
    row_order = matrix.sum(axis=1).sort_values(ascending=False).index
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    return matrix.loc[row_order, col_order]


def melt_mut0(df: pd.DataFrame) -> pd.DataFrame:
    fit_cols = [c for c, _ in TMP_COLS]
    long = df[["IDalign"] + fit_cols].melt(
        id_vars=["IDalign"],
        value_vars=fit_cols,
        var_name="tmp_col",
        value_name="fitness",
    )
    tmp_map = dict(TMP_COLS)
    long["TMP"] = long["tmp_col"].map(tmp_map)
    return long[long["TMP"] > 0]


def panel_colors(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def main() -> None:
    if not JOINT_COUNTS.exists():
        raise SystemExit(f"Missing joint counts: {JOINT_COUNTS}")

    counts_df = pd.read_csv(JOINT_COUNTS)
    matrix = heatmap_order_matrix(counts_df)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )

    fitness_all = pd.read_csv(DATA)
    mut0 = fitness_all[fitness_all["mutations"] == 0].copy()

    pair_specs: list[tuple[str | None, str | None, int]] = []
    for r, c in CELLS:
        if r >= matrix.shape[0] or c >= matrix.shape[1]:
            pair_specs.append((None, None, -1))
            continue
        aa20 = str(matrix.index[r])
        aa28 = str(matrix.columns[c])
        pair_specs.append((aa20, aa28, int(matrix.iloc[r, c])))

    colors = panel_colors(len(CELLS))

    # Global y-limits from all plotted fitness values (TMP > 0)
    ymins, ymaxs = [], []
    for aa20, aa28, _ in pair_specs:
        if aa20 is None:
            continue
        sub = mut0[(mut0["aa20"] == aa20) & (mut0["aa28"] == aa28)].drop_duplicates(
            subset=["IDalign"], keep="first"
        )
        if sub.empty:
            continue
        vals = sub[FIT_COLS_TMP_GT0].to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ymins.append(float(np.nanmin(vals)))
            ymaxs.append(float(np.nanmax(vals)))
    if ymins:
        ymin = min(ymins)
        ymax = max(ymaxs)
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.5
        ylims = (ymin - pad, ymax + pad)
    else:
        ylims = None

    fig, axes = plt.subplots(GRID, GRID, figsize=(12.5, 10), sharex=True, sharey=True)
    axes[GRID - 1, GRID - 1].set_axis_off()

    for k, ((r, c), (aa20, aa28, _hcount)) in enumerate(zip(CELLS, pair_specs)):
        ax = axes[r, c]
        ax.tick_params(width=0.5, axis="both", which="major", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if aa20 is None or aa28 is None:
            ax.set_axis_off()
            continue

        color = PAIR_LINE_COLOR.get((str(aa20), str(aa28)), colors[k])

        sub = mut0[(mut0["aa20"] == aa20) & (mut0["aa28"] == aa28)].drop_duplicates(
            subset=["IDalign"], keep="first"
        )
        title = f"{aa20}20{aa28}28 (n = {len(sub)} homologs)"
        ax.set_title(title, fontsize=12, fontfamily="Arial")

        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "no mut=0 rows",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontfamily="Arial",
                fontsize=12,
            )
            ax.set_xscale("log")
            if ylims is not None:
                ax.set_ylim(*ylims)
            continue

        line_alpha = (
            0.35
            if (str(aa20), str(aa28)) in PAIR_LINE_COLOR
            else 0.2
        )
        long = melt_mut0(sub)
        for _, g in long.groupby("IDalign"):
            ax.plot(
                g["TMP"],
                g["fitness"],
                color=color,
                alpha=line_alpha,
                linewidth=0.5,
            )

        ax.set_xscale("log")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.35, alpha=0.45, color="#aaaaaa")
        ax.set_axisbelow(True)
        if ylims is not None:
            ax.set_ylim(*ylims)

    # X label + tick labels on every plotted panel (sharex hides upper rows by default).
    for (r, c), (aa20, aa28, _) in zip(CELLS, pair_specs):
        if aa20 is None:
            continue
        a = axes[r, c]
        a.set_xlabel(XLABEL, fontsize=12, fontfamily="Arial")
        a.tick_params(axis="x", which="major", labelbottom=True, labelsize=11)

    for ax in axes[:, 0]:
        ax.set_ylabel("Fitness", fontsize=12, fontfamily="Arial")

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
