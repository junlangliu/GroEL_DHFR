#!/usr/bin/env python3
"""Combine folA-background and dfrA-preselected AA20 counts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
FOLA_COUNTS = AUDIT_ROOT / "folA_background" / "folA_background_full_p0abq4_aa20_counts.csv"
DFRA_COUNTS = AUDIT_ROOT / "dfrA_preselected" / "dfrA_preselected_p0abq4_aa20_counts.csv"

OUT_CSV = HERE / "aa20_folA_background_full_vs_dfrA_preselected_P0ABQ4ref.csv"
OUT_PNG = HERE / "aa20_folA_background_full_vs_dfrA_preselected_P0ABQ4ref.png"

AA_STANDARD = list("ACDEFGHIKLMNPQRSTVWY")


def read_counts(path: Path, count_col: str) -> dict[str, int]:
    rows = csv.DictReader(path.open())
    return {row["aa"]: int(row[count_col]) for row in rows}


def write_combined(fol_counts: dict[str, int], dfra_counts: dict[str, int]) -> None:
    fol_total = sum(fol_counts.values())
    dfra_total = sum(dfra_counts.values())
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["aa", "folA_count", "folA_prop", "dfrA_count", "dfrA_prop"])
        for aa in AA_STANDARD:
            fcnt = fol_counts.get(aa, 0)
            dcnt = dfra_counts.get(aa, 0)
            writer.writerow(
                [
                    aa,
                    fcnt,
                    fcnt / fol_total if fol_total else 0.0,
                    dcnt,
                    dcnt / dfra_total if dfra_total else 0.0,
                ]
            )


def plot_stacked(fol_counts: dict[str, int], dfra_counts: dict[str, int]) -> None:
    fol_total = sum(fol_counts.values())
    dfra_total = sum(dfra_counts.values())
    ordered = sorted(AA_STANDARD, key=lambda aa: (fol_counts.get(aa, 0) + dfra_counts.get(aa, 0), aa), reverse=True)
    colors = {
        "L": "#D95F02",
        "M": "#7570B3",
        "I": "#B3B3B3",
        "V": "#1B9E77",
        "Q": "#E6AB02",
        "F": "#66A61E",
        "P": "#A6761D",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    handles = []
    labels = []
    for label, counts, total in [
        ("folA background", fol_counts, fol_total),
        ("dfrA", dfra_counts, dfra_total),
    ]:
        bottom = 0.0
        for aa in ordered:
            frac = counts.get(aa, 0) / total if total else 0.0
            if frac == 0:
                continue
            bar = ax.bar(label, frac, bottom=bottom, color=colors.get(aa, "#D9D9D9"), edgecolor="#333333", linewidth=0.5)
            if label == "folA background":
                handles.append(bar[0])
                labels.append(aa)
            if frac >= 0.05:
                ax.text(label, bottom + frac / 2, f"{aa} {frac:.2f}", ha="center", va="center", fontsize=10)
            bottom += frac
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion at P0ABQ4 reference AA20")
    ax.set_title("AA20 distribution: full folA/DHFR background vs preselected dfrA")
    ax.legend(handles, labels, title="AA", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    fol_counts = read_counts(FOLA_COUNTS, "folA_count")
    dfra_counts = read_counts(DFRA_COUNTS, "dfrA_count")
    write_combined(fol_counts, dfra_counts)
    plot_stacked(fol_counts, dfra_counts)
    print(f"folA total: {sum(fol_counts.values())}")
    print(f"dfrA total: {sum(dfra_counts.values())}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
