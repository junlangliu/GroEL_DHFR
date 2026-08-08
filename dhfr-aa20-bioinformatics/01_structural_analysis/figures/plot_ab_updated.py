#!/usr/bin/env python3
"""
Combined A+B figure.
Panel A: large pie chart with NO callout lines; species legend on its right.
Panel B: AA bar chart.
Output: summary_panel_AB_v2.{png,pdf}  (does NOT overwrite summary_panel.png)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE.parent / "data" / "pdb_aa_list_pos20_591.csv"
OUT_BASE = HERE / "summary_panel_AB_v2"

FONT_FAMILY = "Arial"
FRAME_PT = 0.5
PANEL_LBL_SIZE = 20
PANEL_LBL_WEIGHT = "bold"
FONT_LEG = 10.5     # legend species name
FONT_COUNT = 9.5    # legend count+%


def _species(desc: str) -> str:
    part = str(desc).split(" | ")[0].strip()
    words = part.split()
    return " ".join(words[1:]) if len(words) >= 2 else ""


def _norm2(name: str) -> str:
    w = str(name).split()
    return " ".join(w[:2]) if len(w) > 2 else name


def build_pie_data(df: pd.DataFrame):
    df = df.copy()
    df["organism"] = df["Description"].apply(_species).apply(_norm2)
    df = df[df["organism"].str.strip() != ""]
    sc = df["organism"].value_counts()
    ind = sc[sc >= 10]
    other_n = int(sc[sc < 10].sum())
    labels = list(ind.index)
    values = list(ind.values)
    if other_n > 0:
        labels.append("Others")
        values.append(other_n)

    n = len(ind)
    if n <= 20:
        cols = list(plt.cm.tab20(np.linspace(0, 1, max(n, 1))))
    else:
        cols = list(np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.Set3(np.linspace(0, 1, n - 20)),
        ]))
    if other_n > 0:
        cols.append(np.array([0.5, 0.5, 0.5, 1.0]))
    return labels, values, cols


def plot_pie(ax, ax_leg, df: pd.DataFrame) -> None:
    labels, values, colors = build_pie_data(df)
    total = sum(values)

    # ── big pie, no labels ───────────────────────────────────────────────────
    wedges, _ = ax.pie(
        values, labels=None, colors=colors,
        startangle=90, radius=1.0,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_axis_off()

    # ── legend on ax_leg ─────────────────────────────────────────────────────
    ax_leg.set_axis_off()
    n = len(labels)
    y0 = 0.97
    dy = 0.97 / n          # row height in axes fraction
    sq_h = dy * 0.55
    sq_w = 0.09            # square width in axes fraction
    x_sq = 0.0
    x_sp = x_sq + sq_w + 0.06   # species text x
    x_ct = x_sq + sq_w + 0.06   # count text x (same column, shifted by va)

    for i, (sp, val, col) in enumerate(zip(labels, values, colors)):
        pct = 100.0 * val / total
        y = y0 - i * dy

        rect = mpatches.FancyBboxPatch(
            (x_sq, y - sq_h / 2), sq_w, sq_h,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="none",
            transform=ax_leg.transAxes, clip_on=False,
        )
        ax_leg.add_patch(rect)

        italic = (sp != "Others")
        ax_leg.text(
            x_sp, y + sq_h * 0.12,
            sp,
            transform=ax_leg.transAxes,
            va="bottom", ha="left",
            fontsize=FONT_LEG, fontfamily=FONT_FAMILY,
            fontstyle="italic" if italic else "normal",
        )
        ax_leg.text(
            x_ct, y - sq_h * 0.12,
            f"{val} ({pct:.1f}%)",
            transform=ax_leg.transAxes,
            va="top", ha="left",
            fontsize=FONT_COUNT, fontfamily=FONT_FAMILY,
            color="#555555",
        )


def plot_bar(ax, df: pd.DataFrame) -> None:
    sc = df["Amino_Acid"].value_counts().sort_values(ascending=False)
    aas, vals = list(sc.index), list(sc.values)
    total = sum(vals)
    x = np.arange(len(aas))

    bars = ax.bar(x, vals, color="steelblue", edgecolor="black",
                  linewidth=0.6, alpha=0.75, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(aas, fontsize=12, fontfamily=FONT_FAMILY)
    ax.set_xlabel(
        r"Amino acids aligned to $\mathit{E.\ coli}$ position 20",
        fontsize=12, fontfamily=FONT_FAMILY,
    )
    ax.set_ylabel("Count", fontsize=12, fontfamily=FONT_FAMILY)
    ymax = max(vals)
    ax.set_ylim(0, ymax * 1.22)
    ax.tick_params(axis="both", labelsize=11, width=FRAME_PT)

    for bar, cnt in zip(bars, vals):
        pct = 100.0 * cnt / total
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                f"{cnt} ({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=11, fontfamily=FONT_FAMILY)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(FRAME_PT)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "axes.linewidth": FRAME_PT,
        "xtick.major.width": FRAME_PT,
        "ytick.major.width": FRAME_PT,
    })

    df = pd.read_csv(DATA_CSV)

    # ── layout: [pie | legend | bar] ────────────────────────────────────────
    # figure 14" × 5.8"; pie is square-ish on the left
    fig = plt.figure(figsize=(14, 5.8))
    ax_pie = fig.add_axes([0.01, 0.01, 0.41, 0.97])   # ~5.7" × 5.6" ≈ square
    ax_pie.set_aspect("equal")
    ax_leg = fig.add_axes([0.44, 0.04, 0.22, 0.92])
    ax_bar = fig.add_axes([0.71, 0.13, 0.27, 0.78])

    plot_pie(ax_pie, ax_leg, df)
    plot_bar(ax_bar, df)

    # panel labels
    fig.text(0.01, 0.99, "a",
             fontsize=PANEL_LBL_SIZE, fontweight=PANEL_LBL_WEIGHT,
             fontfamily=FONT_FAMILY, va="top", ha="left")
    fig.text(0.70, 0.99, "b",
             fontsize=PANEL_LBL_SIZE, fontweight=PANEL_LBL_WEIGHT,
             fontfamily=FONT_FAMILY, va="top", ha="left")

    for ext, kw in [
        ("png", {"dpi": 300, "bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
    ]:
        out = OUT_BASE.with_suffix(f".{ext}")
        fig.savefig(out, **kw)
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
