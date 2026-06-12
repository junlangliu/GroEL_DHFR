#!/usr/bin/env python3
"""
Fit log10(TMP) regression per IDalign (mut<=0, lib15) and scatter beta0/beta1.

Differs from compute_regression.py which aggregated by (aa20, aa28) medians.
Here every homolog (IDalign) gets its own slope/intercept so we can view the
dispersion of beta0/beta1 across the dataset.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TMP_POINTS: List[Tuple[float, str]] = [
    (0.0, "fitness_0"),
    (0.058, "fitness_0_058"),
    (0.5, "fitness_0_5"),
    (1.0, "fitness_1"),
    (10.0, "fitness_10"),
    (50.0, "fitness_50"),
    (200.0, "fitness_200"),
]


def fit_per_row(row: pd.Series, x_log: np.ndarray, cols: Iterable[str]) -> Optional[Tuple[float, float]]:
    y = []
    x = []
    for xi, col in zip(x_log, cols):
        val = row[col]
        if pd.isna(val):
            continue
        y.append(val)
        x.append(xi)
    if len(y) < 2:
        return None
    beta1, beta0 = np.polyfit(x, y, 1)
    return float(beta0), float(beta1)


def main() -> None:
    base_final = Path(__file__).resolve().parents[1]  # .../finalize_28
    outdir = base_final / "metrics_attempt2"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base_final / "mut0" / "aa20_aa28_fitness_mutleq0.csv", dtype={"IDalign": "string"})

    tmp_vals = np.array([v for v, _ in TMP_POINTS if v > 0])
    tmp_cols = [c for v, c in TMP_POINTS if v > 0]
    x_log = np.log10(tmp_vals)

    rows = []
    for _, r in df.iterrows():
        fit = fit_per_row(r, x_log, tmp_cols)
        if fit is None:
            continue
        beta0, beta1 = fit
        rows.append(
            {
                "IDalign": r["IDalign"],
                "aa20": r["aa20"],
                "aa28": r["aa28"],
                "lib": r.get("lib", pd.NA),
                "num_barcodes": r.get("num_barcodes", pd.NA),
                "beta0": beta0,
                "beta1": beta1,
                "n_points": len([c for c in tmp_cols if pd.notna(r[c])]),
            }
        )

    out = pd.DataFrame(rows)
    out_path = outdir / "regression_per_idalign_mut0.csv"
    out.to_csv(out_path, index=False)

    if out.empty:
        return

    # Build combo label and palette
    out["combo"] = out["aa20"] + "/" + out["aa28"]
    combos = out["combo"].unique()
    palette = sns.color_palette("husl", n_colors=len(combos))
    color_map = dict(zip(combos, palette))

    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(
        data=out,
        x="beta0",
        y="beta1",
        hue="combo",
        palette=color_map,
        s=45,
        alpha=0.75,
        edgecolor="none",
        legend=False,  # too many combos for a readable legend
        ax=ax,
    )
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("beta0 (intercept at log10 TMP = 0)")
    ax.set_ylabel("beta1 (slope vs log10 TMP)")
    ax.set_title("Per-ID regressions (mut<=0, lib15)\nColor encodes aa20/aa28 combo")
    fig.tight_layout()
    fig.savefig(outdir / "beta_scatter_per_idalign_mut0.png", dpi=300)
    plt.close(fig)

    # Save color look-up for reference
    pd.DataFrame({"combo": list(color_map.keys()), "color_rgb": [tuple(c) for c in color_map.values()]}).to_csv(
        outdir / "beta_scatter_per_idalign_colors.csv", index=False
    )


if __name__ == "__main__":
    main()
