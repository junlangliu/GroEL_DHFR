#!/usr/bin/env python3
"""
Bar plot of aa20 counts (lib15, mut=0), highlighting I/M/L as the major trio.

Input: aa20_counts_paper_like_zero_mut_lib15_filtered.csv
Output: aa20_counts_highlight_IML_bar.png (does not overwrite other figures).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / "aa20_counts_paper_like_zero_mut_lib15_filtered.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"aa20_aligned": "aa20"})

    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = df["count"].sum()

    # colors: match violin palette
    highlight = {"I": "#1f77b4", "M": "#ff7f0e", "L": "#2ca02c"}
    df["color"] = df["aa20"].map(highlight).fillna("#b8bcc4")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    x = list(range(len(df)))
    ymax = df["count"].max()
    label_threshold = 20          # bars below this are "small"
    min_vis_h = ymax * 0.012      # minimum rendered bar height for visibility
    label_anchor = ymax * 0.03    # where small-bar labels sit above x-axis

    # draw bars at their true height (small bars get a minimum pixel stub)
    rendered = [max(c, min_vis_h) for c in df["count"]]
    ax.bar(x, rendered, color=df["color"], edgecolor="white", linewidth=0.5,
           width=0.62, zorder=3)

    # annotate
    for i, (cnt, aa) in enumerate(zip(df["count"], df["aa20"])):
        pct = cnt / total * 100
        label = f"{cnt}\n({pct:.1f}%)"
        if cnt >= label_threshold:
            # label sits just above the bar top
            ax.text(i, cnt + ymax * 0.02, label, ha="center", va="bottom",
                    fontsize=9, color="#1a1a1a", linespacing=1.4)
        else:
            # label sits near the x-axis; thin dotted line connects to bar stub
            ax.plot([i, i], [min_vis_h * 1.1, label_anchor * 0.85],
                    color="#888888", linewidth=0.6, linestyle=":", zorder=2)
            ax.text(i, label_anchor, label, ha="center", va="bottom",
                    fontsize=8, color="#444444", linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(df["aa20"], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", labelsize=10)

    ax.set_ylim(0, ymax * 1.22)
    ax.set_ylabel("Count", fontsize=12, labelpad=6)
    ax.set_xlabel("Residue at position 20", fontsize=12, labelpad=6)
    ax.set_title("AA20 distribution  (lib15, mut = 0)", fontsize=13,
                 fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.45,
                  alpha=0.5, color="#aaaaaa", zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=highlight[aa], edgecolor="white", label=aa)
        for aa in ["L", "M", "I"]
    ]
    ax.legend(handles=legend_handles, title="Highlighted", frameon=False,
              fontsize=10, title_fontsize=10, loc="upper right",
              handlelength=1.2, handleheight=1.0)

    fig.tight_layout()
    out = here / "aa20_counts_highlight_IML_bar.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
