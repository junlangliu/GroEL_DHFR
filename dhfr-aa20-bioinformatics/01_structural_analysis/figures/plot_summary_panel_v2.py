#!/usr/bin/env python3
"""
Full summary panel (a–h) combining:
  - New panel A+B design (large clean pie + legend, AA bar)
    using pdb_aa_list_pos20_591.csv
  - Original panels C–H (Ramachandran, phi/psi KDE, violin plots)
    using dihedral_angles_pos20.csv + ligand_classification.csv

Output: summary_panel_v2.{png,pdf}  (does NOT overwrite summary_panel.png)
"""

import csv
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_psi_by_ligand_category import categorize_entry  # noqa: E402

DATA_NEW   = HERE.parent / "data" / "pdb_aa_list_pos20_591.csv"
DIHEDRAL   = HERE.parent / "data" / "dihedral_angles_pos20.csv"
CLASSIF    = HERE.parent / "data" / "ligand_classification.csv"
OUT_BASE   = HERE / "summary_panel_v2"

# ── shared style ─────────────────────────────────────────────────────────────
FONT_FAMILY = "Arial"
FRAME_PT = 0.5
LW_AXIS_REF = 0.5
LW_SCATTER_EDGE = 0.3
LW_HIST_EDGE = 0.3
LW_KDE = 1.0
LW_VIOLIN = 1.0
LW_QUARTILE = 1.0
LW_STAT = 1.0
LW_THRESH = 1.0

PANEL_LBL_SIZE   = 20
PANEL_LBL_WEIGHT = "bold"

FS_CH_AXIS   = 14
FS_CH_TICK   = 13
FS_CH_LEGEND = 13
FS_CH_ANNOT  = 13
FS_B_AXIS    = 12
FS_B_TICK    = 11
FS_B_BAR     = 11
FS_LEG_SP    = 10.5
FS_LEG_CT    = 9.5

AA_COLORS = {"M": "#1f77b4", "L": "#ff7f0e", "I": "#2ca02c"}
TARGET_AAs = {"M", "L", "I", "MET", "LEU", "ILE"}
PHI_LABEL  = r"$\phi$ (°)"
PSI_LABEL  = r"$\psi$ (°)"


# ── matplotlib global style ───────────────────────────────────────────────────
def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "font.size": FS_CH_TICK,
        "axes.labelsize": FS_CH_AXIS,
        "axes.linewidth": FRAME_PT,
        "xtick.major.width": FRAME_PT,
        "ytick.major.width": FRAME_PT,
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def style_axes(ax) -> None:
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(FRAME_PT)
    ax.tick_params(axis="both", colors="black", width=FRAME_PT, labelcolor="black")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=PANEL_LBL_SIZE, fontweight=PANEL_LBL_WEIGHT,
            fontfamily=FONT_FAMILY, va="top", ha="left")


# ── panel A helpers ───────────────────────────────────────────────────────────
def _species(desc: str) -> str:
    part = str(desc).split(" | ")[0].strip()
    words = part.split()
    return " ".join(words[1:]) if len(words) >= 2 else ""

def _norm2(name: str) -> str:
    w = str(name).split()
    return " ".join(w[:2]) if len(w) > 2 else name

def _build_pie(df: pd.DataFrame):
    df = df.copy()
    df["organism"] = df["Description"].apply(_species).apply(_norm2)
    df = df[df["organism"].str.strip() != ""]
    sc = df["organism"].value_counts()
    ind = sc[sc >= 10]
    other_n = int(sc[sc < 10].sum())
    labels = list(ind.index);  values = list(ind.values)
    if other_n > 0:
        labels.append("Others"); values.append(other_n)
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


def plot_panel_a(ax_pie, ax_leg, df: pd.DataFrame) -> None:
    labels, values, colors = _build_pie(df)
    total = sum(values)

    wedges, _ = ax_pie.pie(
        values, labels=None, colors=colors, startangle=90, radius=1.0,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax_pie.set_xlim(-1.25, 1.25)
    ax_pie.set_ylim(-1.25, 1.25)
    ax_pie.set_axis_off()

    ax_leg.set_axis_off()
    n   = len(labels)
    y0  = 0.96
    dy  = 0.96 / n
    sq_h = dy * 0.55
    sq_w = 0.09
    x_sq = 0.01
    x_tx = x_sq + sq_w + 0.07

    for i, (sp, val, col) in enumerate(zip(labels, values, colors)):
        pct = 100.0 * val / total
        y   = y0 - i * dy
        rect = mpatches.FancyBboxPatch(
            (x_sq, y - sq_h / 2), sq_w, sq_h,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="none",
            transform=ax_leg.transAxes, clip_on=False,
        )
        ax_leg.add_patch(rect)
        italic = (sp != "Others")
        ax_leg.text(x_tx, y + sq_h * 0.12, sp,
                    transform=ax_leg.transAxes, va="bottom", ha="left",
                    fontsize=FS_LEG_SP, fontfamily=FONT_FAMILY,
                    fontstyle="italic" if italic else "normal")
        ax_leg.text(x_tx, y - sq_h * 0.12, f"{val} ({pct:.1f}%)",
                    transform=ax_leg.transAxes, va="top", ha="left",
                    fontsize=FS_LEG_CT, fontfamily=FONT_FAMILY, color="#555555")


# ── panel B ───────────────────────────────────────────────────────────────────
def plot_panel_b(ax, df: pd.DataFrame) -> None:
    sc   = df["Amino_Acid"].value_counts().sort_values(ascending=False)
    aas  = list(sc.index);  vals = list(sc.values);  total = sum(vals)
    x    = np.arange(len(aas))
    bars = ax.bar(x, vals, color="steelblue", edgecolor="black",
                  linewidth=0.6, alpha=0.75, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(aas, fontsize=FS_B_TICK, fontfamily=FONT_FAMILY)
    ax.set_xlabel(r"Amino acids aligned to $\mathit{E.\ coli}$ position 20",
                  fontsize=FS_B_AXIS, fontfamily=FONT_FAMILY)
    ax.set_ylabel("Count", fontsize=FS_B_AXIS, fontfamily=FONT_FAMILY)
    ymax = max(vals)
    ax.set_ylim(0, ymax * 1.22)
    ax.tick_params(axis="both", labelsize=FS_B_TICK, width=FRAME_PT)
    for bar, cnt in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                f"{cnt} ({100.*cnt/total:.1f}%)",
                ha="center", va="bottom", fontsize=FS_B_BAR, fontfamily=FONT_FAMILY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(FRAME_PT)


# ── panel C–E: Ramachandran + phi/psi KDE ────────────────────────────────────
def load_rama(csv_path: Path):
    phi_v, psi_v, aas = [], [], []
    aa_map = {"MET": "M", "LEU": "L", "ILE": "I"}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            phi_s = row.get("Phi", "").strip()
            psi_s = row.get("Psi", "").strip()
            aa    = row.get("Amino_Acid", "").strip().upper()
            faa   = row.get("Found_Amino_Acid", "").strip().upper()
            if not (phi_s and psi_s and phi_s.lower() != "none" and psi_s.lower() != "none"):
                continue
            try:
                phi, psi = float(phi_s), float(psi_s)
            except ValueError:
                continue
            code = None
            if aa in "MLI" and len(aa) == 1:
                code = aa
            elif aa in aa_map:
                code = aa_map[aa]
            elif faa in aa_map:
                code = aa_map[faa]
            if code:
                phi_v.append(phi); psi_v.append(psi); aas.append(code)
    counts = {a: aas.count(a) for a in "MLI"}
    return phi_v, psi_v, aas, counts


def plot_panel_c(ax, phi_v, psi_v, aas, counts) -> None:
    for aa in "MLI":
        xs = [phi_v[i] for i, a in enumerate(aas) if a == aa]
        ys = [psi_v[i] for i, a in enumerate(aas) if a == aa]
        if xs:
            ax.scatter(xs, ys, alpha=0.6, s=28,
                       label=f"{aa} (n={counts[aa]})",
                       c=AA_COLORS[aa], edgecolors="black", linewidths=LW_SCATTER_EDGE)
    ax.set_xlabel(PHI_LABEL, fontsize=FS_CH_AXIS)
    ax.set_ylabel(PSI_LABEL, fontsize=FS_CH_AXIS)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=LW_AXIS_REF, alpha=0.5)
    ax.axvline(0, color="k", lw=LW_AXIS_REF, alpha=0.5)
    ax.legend(loc="upper right", fontsize=FS_CH_LEGEND, frameon=False,
              prop={"family": FONT_FAMILY})
    ax.tick_params(labelsize=FS_CH_TICK)
    style_axes(ax)


def plot_panel_d(ax, phi_v, aas, counts) -> None:
    bins = np.linspace(-180, 0, 37)
    xs   = np.linspace(-180, 0, 200)
    for aa in "MLI":
        vals = np.array([phi_v[i] for i, a in enumerate(aas) if a == aa])
        if len(vals) > 1:
            ax.hist(vals, bins=bins, alpha=0.3, density=True,
                    label=f"{aa} (n={counts[aa]})", color=AA_COLORS[aa],
                    edgecolor="black", linewidth=LW_HIST_EDGE)
            try:
                ax.plot(xs, gaussian_kde(vals)(xs), color=AA_COLORS[aa], lw=LW_KDE)
            except Exception:
                pass
    ax.set_xlabel(PHI_LABEL, fontsize=FS_CH_AXIS)
    ax.set_ylabel("Probability Density", fontsize=FS_CH_AXIS)
    ax.set_xlim(-180, 0)
    ax.axvline(0, color="k", lw=LW_AXIS_REF, alpha=0.5)
    ax.legend(loc="upper left", fontsize=FS_CH_LEGEND, frameon=False,
              prop={"family": FONT_FAMILY})
    ax.tick_params(labelsize=FS_CH_TICK)
    style_axes(ax)


def plot_panel_e(ax, psi_v, aas, counts) -> None:
    bins = np.linspace(0, 180, 37)
    xs   = np.linspace(0, 180, 200)
    for aa in "MLI":
        vals = np.array([psi_v[i] for i, a in enumerate(aas) if a == aa])
        if len(vals) > 1:
            ax.hist(vals, bins=bins, alpha=0.3, density=True,
                    label=f"{aa} (n={counts[aa]})", color=AA_COLORS[aa],
                    edgecolor="black", linewidth=LW_HIST_EDGE)
            try:
                ax.plot(xs, gaussian_kde(vals)(xs), color=AA_COLORS[aa], lw=LW_KDE)
            except Exception:
                pass
    ax.set_xlabel(PSI_LABEL, fontsize=FS_CH_AXIS)
    ax.set_ylabel("Probability Density", fontsize=FS_CH_AXIS)
    ax.set_xlim(0, 180)
    ax.axvline(0, color="k", lw=LW_AXIS_REF, alpha=0.5)
    ax.legend(loc="upper left", fontsize=FS_CH_LEGEND, frameon=False,
              prop={"family": FONT_FAMILY})
    ax.tick_params(labelsize=FS_CH_TICK)
    style_axes(ax)


# ── panels F–H: violin plots ──────────────────────────────────────────────────
def load_violin(classif_csv: Path, dihedral_csv: Path) -> dict:
    psi_map = {}
    with open(dihedral_csv) as f:
        for row in csv.DictReader(f):
            pid = row.get("PDB_ID", "").strip().upper()
            aa  = row.get("Amino_Acid", "").strip().upper()
            ps  = row.get("Psi", "").strip()
            if row.get("Status", "").strip() == "Success" and aa in "MLI" \
               and ps and ps.lower() != "none":
                try:
                    psi_map[pid] = (float(ps), aa)
                except ValueError:
                    pass

    cats = ["NAP_NDP_FOL", "NAP_NDP_MTX_TOP", "other"]
    data = {aa: {c: [] for c in cats} for aa in "MLI"}
    with open(classif_csv) as f:
        for row in csv.DictReader(f):
            pid    = row.get("PDB_ID", "").strip().upper()
            pos_aa = row.get("Pos20_Amino_Acid", "").strip().upper()
            cof    = row.get("Cofactor", "").strip()
            sub    = row.get("Substrate_Antibiotic", "").strip()
            if pos_aa in "MLI" and pid in psi_map:
                psi_val, _ = psi_map[pid]
                if psi_val >= 50:
                    data[pos_aa][categorize_entry(cof, sub)].append(psi_val)
    return data


def plot_panel_fgh(ax, aa: str, data: dict, threshold: float = 128.39) -> None:
    cats   = ["NAP_NDP_FOL", "NAP_NDP_MTX_TOP", "other"]
    clrs   = {"NAP_NDP_FOL": "steelblue", "NAP_NDP_MTX_TOP": "orange", "other": "gray"}
    lbl_map = {"NAP_NDP_FOL":    "NAP/NDP\n−FOL",
               "NAP_NDP_MTX_TOP":"NAP/NDP\n−MTX/TOP",
               "other":          "Other"}

    vdata, vpos, vclrs = [], [], []
    for i, cat in enumerate(cats):
        d = data[aa][cat]
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(
            [f"{lbl_map[c]}\n(n={len(data[aa][c])})" for c in cats],
            fontsize=10)
        if d:
            vdata.append(d); vpos.append(i); vclrs.append(clrs[cat])

    ax.set_xlim(-0.5, 2.5)
    if vdata:
        parts = ax.violinplot(vdata, positions=vpos, widths=0.6,
                              showmeans=False, showmedians=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(vclrs[j]); pc.set_alpha(0.4)
            pc.set_edgecolor("black"); pc.set_linewidth(LW_VIOLIN)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_edgecolor("black"); parts[key].set_linewidth(LW_VIOLIN)

    for i, cat in enumerate(cats):
        d = data[aa][cat]
        if not d:
            continue
        q25, med, q75 = np.percentile(d, [25, 50, 75])
        ax.hlines([q25, med, q75], i - 0.18, i + 0.18, colors="black", lw=LW_QUARTILE)

    # stat brackets
    valid = {cat: data[aa][cat] for cat in cats if len(data[aa][cat]) > 1}
    comps = []
    clist = list(valid.keys())
    for i in range(len(clist)):
        for j in range(i + 1, len(clist)):
            c1, c2 = clist[i], clist[j]
            try:
                _, p = mannwhitneyu(valid[c1], valid[c2], alternative="two-sided")
                p_str = "n.s." if p >= 0.05 else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
                comps.append({"p1": cats.index(c1), "p2": cats.index(c2),
                               "pstr": p_str, "span": abs(cats.index(c2) - cats.index(c1))})
            except Exception:
                pass
    comps.sort(key=lambda x: (-x["span"], x["p1"]))
    y_base, y_off = 35, [0, 10, 20]
    for idx, comp in enumerate(comps):
        p1, p2, by = comp["p1"], comp["p2"], y_base + y_off[idx % 3]
        ax.plot([p1, p1], [by, by + 3], "k-", lw=LW_STAT)
        ax.plot([p2, p2], [by, by + 3], "k-", lw=LW_STAT)
        ax.plot([p1, p2], [by, by], "k-", lw=LW_STAT)
        ax.text((p1 + p2) / 2, by - 2, comp["pstr"],
                ha="center", va="top", fontsize=FS_CH_ANNOT, fontfamily=FONT_FAMILY)

    ax.axhline(threshold, color="red", linestyle="--", lw=LW_THRESH, alpha=0.7,
               label=f"Threshold ({threshold}°)")
    ax.set_ylabel(PSI_LABEL, fontsize=FS_CH_AXIS)
    ax.legend(loc="upper left", fontsize=FS_CH_LEGEND, frameon=False,
              prop={"family": FONT_FAMILY})
    ax.set_ylim(0, 180)
    ax.tick_params(labelsize=FS_CH_TICK)
    style_axes(ax)


# ── assemble figure ───────────────────────────────────────────────────────────
def main() -> None:
    setup_style()

    df_new = pd.read_csv(DATA_NEW)
    phi_v, psi_v, aas, counts = load_rama(DIHEDRAL)
    violin_data = load_violin(CLASSIF, DIHEDRAL)

    # ── figure dimensions ────────────────────────────────────────────────────
    # AB section matches summary_panel_AB_v2.png exactly (14" × 5.8").
    # The combined canvas is 14" wide × 16.5" tall; the AB block sits at the
    # top 5.8", C–H fill the remaining ~10.7" below.
    FW, FH = 14.0, 16.5        # figure width / height in inches
    AB_H  = 5.8                # height of AB section (same as standalone)
    # Scale factor: fraction of FH occupied by the AB block
    s = AB_H / FH              # ≈ 0.352
    # y0 of the AB block bottom in figure coords
    y_ab = (FH - AB_H) / FH   # ≈ 0.648

    # Standalone axes coords (from plot_ab_updated.py, figsize=(14, 5.8)):
    #   ax_pie : [0.01, 0.01, 0.41, 0.97]
    #   ax_leg : [0.44, 0.04, 0.22, 0.92]
    #   ax_bar : [0.71, 0.13, 0.27, 0.78]
    # Mapped to combined figure (same x fracs; y scaled into the AB block):
    def ab_axes(x0, y0_sa, w, h_sa):
        return [x0, y_ab + y0_sa * s, w, h_sa * s]

    fig = plt.figure(figsize=(FW, FH))

    # ── top row: A (pie + legend) and B (bar) ───────────────────────────────
    ax_pie = fig.add_axes(ab_axes(0.01, 0.01, 0.41, 0.97))
    ax_pie.set_aspect("equal")
    ax_leg = fig.add_axes(ab_axes(0.44, 0.04, 0.22, 0.92))
    ax_bar = fig.add_axes(ab_axes(0.71, 0.13, 0.27, 0.78))

    # ── bottom 2 rows: C–H via GridSpec ─────────────────────────────────────
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=y_ab - 0.02,   # small gap below AB block
                           bottom=0.03,
                           hspace=0.30, wspace=0.34)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[1, 2])

    # ── plot ─────────────────────────────────────────────────────────────────
    plot_panel_a(ax_pie, ax_leg, df_new)
    plot_panel_b(ax_bar, df_new)

    if phi_v:
        plot_panel_c(ax_c, phi_v, psi_v, aas, counts)
        plot_panel_d(ax_d, phi_v, aas, counts)
        plot_panel_e(ax_e, psi_v, aas, counts)
    else:
        for ax in (ax_c, ax_d, ax_e):
            ax.text(0.5, 0.5, "No M/L/I dihedral data",
                    ha="center", va="center", transform=ax.transAxes)

    for ax, aa in [(ax_f, "M"), (ax_g, "L"), (ax_h, "I")]:
        plot_panel_fgh(ax, aa, violin_data)

    # ── panel labels ─────────────────────────────────────────────────────────
    y_lbl = y_ab + 0.99 * s    # top of AB block
    fig.text(0.01, y_lbl, "a", fontsize=PANEL_LBL_SIZE, fontweight=PANEL_LBL_WEIGHT,
             fontfamily=FONT_FAMILY, va="top", ha="left")
    fig.text(0.70, y_lbl, "b", fontsize=PANEL_LBL_SIZE, fontweight=PANEL_LBL_WEIGHT,
             fontfamily=FONT_FAMILY, va="top", ha="left")
    for ax, lbl in [(ax_c, "c"), (ax_d, "d"), (ax_e, "e"),
                    (ax_f, "f"), (ax_g, "g"), (ax_h, "h")]:
        panel_label(ax, lbl)

    # ── save ─────────────────────────────────────────────────────────────────
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
