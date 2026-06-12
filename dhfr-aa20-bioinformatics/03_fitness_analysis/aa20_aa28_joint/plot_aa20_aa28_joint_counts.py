#!/usr/bin/env python3
"""Plot aa20 vs aa28 joint counts as heatmaps (full + >=10 masked gradient).

Color now maps linearly to raw counts (no log) for clearer interpretation.
The second panel masks counts < MIN_HIGHLIGHT to white but keeps the same
linear color scaling for the visible cells.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


def main() -> None:
    here = Path(__file__).resolve().parent
    df = pd.read_csv(here / "aa20_aa28_joint_counts.csv")

    # Threshold used in downstream filtering (e.g., only 7 pairs kept)
    MIN_HIGHLIGHT = 10

    # Build matrix of counts and order rows/cols by total count to highlight hotspots.
    matrix = df.pivot_table(
        index="aa20", columns="aa28", values="count", fill_value=0
    )
    row_order = matrix.sum(axis=1).sort_values(ascending=False).index
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix.loc[row_order, col_order]

    counts = matrix.values.astype(float)
    vmax = counts.max() if counts.size else 0
    norm = Normalize(vmin=0, vmax=vmax or 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Monochrome colormap: white for zero/missing, deeper color for higher counts.
    cmap = LinearSegmentedColormap.from_list("mono_blue", ["white", "#0b4fa8"])
    cmap.set_bad("white")

    masked = np.ma.masked_where(counts == 0, counts)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    # Annotate non-zero cells with the raw counts.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            if val == 0:
                continue
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("aa28")
    ax.set_ylabel("aa20")
    ax.set_title("aa20 vs aa28 joint counts (color ∝ count)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("count")

    plt.tight_layout()
    out_path = here / "aa20_aa28_joint_counts_heatmap.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path.name}")

    # Masked-gradient highlight: same color scale, but counts < MIN_HIGHLIGHT are white.
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    masked_hi = np.ma.masked_where(counts < MIN_HIGHLIGHT, counts)
    im2 = ax2.imshow(masked_hi, cmap=cmap, norm=norm, aspect="auto")

    vmax_hi = masked_hi.max() if masked_hi.count() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            if val == 0:
                continue
            # Colored background only when >= threshold; numbers always shown.
            if val >= MIN_HIGHLIGHT:
                color = "white" if val > vmax_hi * 0.6 else "black"
            else:
                color = "black"
            ax2.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

    ax2.set_xticks(range(matrix.shape[1]))
    ax2.set_xticklabels(matrix.columns)
    ax2.set_yticks(range(matrix.shape[0]))
    ax2.set_yticklabels(matrix.index)
    ax2.set_xlabel("aa28")
    ax2.set_ylabel("aa20")
    ax2.set_title(
        f"aa20 vs aa28 joint counts (color ∝ count, masked < {MIN_HIGHLIGHT})"
    )

    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("count")

    fig2.tight_layout()
    out_path2 = here / "aa20_aa28_joint_counts_heatmap_ge10.png"
    plt.savefig(out_path2, dpi=300)
    print(f"Saved {out_path2.name}")


if __name__ == "__main__":
    main()
