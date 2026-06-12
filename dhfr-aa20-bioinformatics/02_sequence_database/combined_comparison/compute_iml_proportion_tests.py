#!/usr/bin/env python3
"""Test I/M/L proportion differences for folA background vs dfrA."""

from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import fisher_exact


HERE = Path(__file__).resolve().parent
IN_CSV = HERE / "aa20_folA_background_full_vs_dfrA_preselected_P0ABQ4ref.csv"
OUT_CSV = HERE / "iml_proportion_tests_folA_background_vs_dfrA_preselected.csv"
OUT_MD = HERE / "iml_proportion_tests_folA_background_vs_dfrA_preselected.md"


def bh_fdr(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        value = min(running_min, p_values[idx] * n / rank)
        running_min = value
        adjusted[idx] = min(value, 1.0)
    return adjusted


def main() -> None:
    rows = list(csv.DictReader(IN_CSV.open()))
    by_aa = {row["aa"]: row for row in rows}
    fol_total = sum(int(row["folA_count"]) for row in rows)
    dfra_total = sum(int(row["dfrA_count"]) for row in rows)

    out_rows = []
    for aa in ["I", "M", "L"]:
        row = by_aa[aa]
        fol_count = int(row["folA_count"])
        dfra_count = int(row["dfrA_count"])
        odds_ratio, p_value = fisher_exact(
            [[fol_count, fol_total - fol_count], [dfra_count, dfra_total - dfra_count]],
            alternative="two-sided",
        )
        out_rows.append(
            {
                "aa": aa,
                "folA_count": fol_count,
                "folA_total": fol_total,
                "folA_prop": fol_count / fol_total,
                "dfrA_count": dfra_count,
                "dfrA_total": dfra_total,
                "dfrA_prop": dfra_count / dfra_total,
                "prop_difference_dfrA_minus_folA": (dfra_count / dfra_total) - (fol_count / fol_total),
                "fisher_odds_ratio": odds_ratio,
                "fisher_p_two_sided": p_value,
            }
        )

    for row, adj in zip(out_rows, bh_fdr([row["fisher_p_two_sided"] for row in out_rows])):
        row["bh_fdr_p"] = adj

    fields = [
        "aa",
        "folA_count",
        "folA_total",
        "folA_prop",
        "dfrA_count",
        "dfrA_total",
        "dfrA_prop",
        "prop_difference_dfrA_minus_folA",
        "fisher_odds_ratio",
        "fisher_p_two_sided",
        "bh_fdr_p",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    with OUT_MD.open("w") as handle:
        handle.write("# I/M/L Proportion Tests: folA Background vs dfrA\n\n")
        handle.write(
            "Each row tests one amino acid versus all other standard amino acids "
            "using a two-sided Fisher exact test. BH-FDR is computed across I/M/L.\n\n"
        )
        handle.write(
            "| AA | folA count/total | folA prop | dfrA count/total | dfrA prop | "
            "dfrA - folA | Fisher p | BH-FDR p |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in out_rows:
            handle.write(
                f"| {row['aa']} | {row['folA_count']}/{row['folA_total']} | "
                f"{row['folA_prop']:.6f} | {row['dfrA_count']}/{row['dfrA_total']} | "
                f"{row['dfrA_prop']:.6f} | {row['prop_difference_dfrA_minus_folA']:.6f} | "
                f"{row['fisher_p_two_sided']:.3e} | {row['bh_fdr_p']:.3e} |\n"
            )
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
