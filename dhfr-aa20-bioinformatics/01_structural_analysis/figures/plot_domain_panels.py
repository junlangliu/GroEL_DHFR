#!/usr/bin/env python3
"""
Plot panels A–H separately for prokaryotes and eukaryotes.
- Panel A: species pie (domain-filtered, legend on right, no callout lines)
- Panel B: AA bar chart (domain-filtered)
- Panels C–H: same tests/style as plot_summary_panel_v2.py
- Panel D legend moved to upper-left

Output (does NOT overwrite any existing figures):
  summary_panel_prokaryotes_v2.{png,pdf}
  summary_panel_eukaryotes_v2.{png,pdf}
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

DATA_NEW = HERE.parent / "data" / "pdb_aa_list_pos20_591.csv"
DIHEDRAL = HERE.parent / "data" / "dihedral_angles_pos20.csv"
CLASSIF  = HERE.parent / "data" / "ligand_classification.csv"

# ── domain classification ─────────────────────────────────────────────────────
SPECIES_DOMAIN: dict[str, str] = {
    "Escherichia coli":              "prokaryote",
    "Staphylococcus aureus":         "prokaryote",
    "Mycobacterium tuberculosis":    "prokaryote",
    "Bacillus anthracis":            "prokaryote",
    "Mycobacterium ulcerans":        "prokaryote",
    "Lacticaseibacillus casei":      "prokaryote",
    "Klebsiella pneumoniae":         "prokaryote",
    "Yersinia pestis":               "prokaryote",
    "Coxiella burnetii":             "prokaryote",
    "Haloferax volcanii":            "prokaryote",
    "Mycobacterium avium":           "prokaryote",
    "Moritella profunda":            "prokaryote",
    "Enterococcus faecalis":         "prokaryote",
    "Mycolicibacterium smegmatis":   "prokaryote",
    "Stenotrophomonas maltophilia":  "prokaryote",
    "Geobacillus stearothermophilus":"prokaryote",
    "Streptococcus pneumoniae":      "prokaryote",
    "Mycobacterium kansasii":        "prokaryote",
    "Mycobacteroides abscessus":     "prokaryote",
    "Bacillus subtilis":             "prokaryote",
    "uncultured bacterium":          "prokaryote",
    "Homo sapiens":                  "eukaryote",
    "Pneumocystis carinii":          "eukaryote",
    "Candida albicans":              "eukaryote",
    "Plasmodium falciparum":         "eukaryote",
    "Toxoplasma gondii":             "eukaryote",
    "Nakaseomyces glabratus":        "eukaryote",
    "Trypanosoma cruzi":             "eukaryote",
    "Gallus gallus":                 "eukaryote",
    "Cryptosporidium hominis":       "eukaryote",
    "Mus musculus":                  "eukaryote",
    "Wuchereria bancrofti":          "eukaryote",
    "Plasmodium vivax":              "eukaryote",
    "Candidozyma auris":             "eukaryote",
    "Trypanosoma brucei":            "eukaryote",
    "Aspergillus flavus":            "eukaryote",
    "Schistosoma mansoni":           "eukaryote",
    "Pneumocystis jirovecii":        "eukaryote",
    "Arabidopsis thaliana":          "eukaryote",
    "Tequatrovirus T4":              "skip",
}

DOMAIN_TITLES   = {"prokaryote": "Prokaryotes", "eukaryote": "Eukaryotes"}
DOMAIN_OUTNAMES = {
    "prokaryote": "summary_panel_prokaryotes_v2",
    "eukaryote":  "summary_panel_eukaryotes_v2",
}


def _sp(desc: str) -> str:
    part = str(desc).split(" | ")[0].strip()
    w = part.split()
    return " ".join(w[1:3]) if len(w) >= 2 else ""


def build_domain_sets(csv_path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(csv_path)
    sets: dict[str, set[str]] = {"prokaryote": set(), "eukaryote": set()}
    for _, row in df.iterrows():
        pid = str(row["PDB_ID"]).strip().upper()
        sp  = _sp(str(row.get("Description", "")))
        dom = SPECIES_DOMAIN.get(sp)
        if dom in sets:
            sets[dom].add(pid)
    for dom, s in sets.items():
        print(f"  {dom}: {len(s)} PDB IDs")
    return sets


# ── style ──────────────────────────────────────────────────────────────────────
FONT  = "Arial"
FPT   = 0.5
FSA   = 14   # axis label
FST   = 13   # tick / legend
FSLP  = 20   # panel letter
FWP   = "bold"
FSLEG = 10.5
FSCT  = 9.5
AA_C  = {"M": "#1f77b4", "L": "#ff7f0e", "I": "#2ca02c"}
PHIL  = r"$\phi$ (°)"
PSIL  = r"$\psi$ (°)"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT],
        "font.size": FST, "axes.labelsize": FSA,
        "axes.linewidth": FPT, "xtick.major.width": FPT,
        "ytick.major.width": FPT, "axes.unicode_minus": False,
        "axes.grid": False, "axes.edgecolor": "black",
    })


def sax(ax) -> None:
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(FPT)
    ax.tick_params(axis="both", colors="black", width=FPT)
    leg = ax.get_legend()
    if leg:
        leg.set_frame_on(False)


def plbl(ax, lbl, x=-0.18, y=1.04):
    ax.text(x, y, lbl, transform=ax.transAxes,
            fontsize=FSLP, fontweight=FWP, fontfamily=FONT, va="top", ha="left")


# ── professional palette (no jarring red) ─────────────────────────────────────
# Tableau-derived muted palette; reds replaced with rose/teal variants.
PROF_PALETTE = [
    "#4e79a7",  # steel blue
    "#f28e2b",  # orange
    "#59a14f",  # green
    "#76b7b2",  # teal
    "#edc948",  # amber
    "#b07aa1",  # purple
    "#9c755f",  # brown
    "#ff9da7",  # soft rose (not jarring)
    "#a0cbe8",  # sky blue
    "#ffbe7d",  # peach
    "#8cd17d",  # light green
    "#86bcb6",  # light teal
    "#d4a6c8",  # lavender
    "#499894",  # dark teal
    "#f1ce63",  # light amber
    "#bab0ac",  # warm gray
    "#d7b5a6",  # sand
    "#6b9ac4",  # medium blue
    "#c2a5cf",  # light purple
    "#9d7660",  # saddle brown
]


# ── panel A: pie + legend ─────────────────────────────────────────────────────
def _norm2(name: str) -> str:
    w = str(name).split(); return " ".join(w[:2]) if len(w) > 2 else name


def build_pie(df: pd.DataFrame):
    df = df.copy()
    df["organism"] = df["Description"].apply(_sp).apply(_norm2)
    df = df[df["organism"].str.strip() != ""]
    sc  = df["organism"].value_counts()
    ind = sc[sc >= 5]          # lower threshold for domain subsets
    other_n = int(sc[sc < 5].sum())
    labels = list(ind.index);  values = list(ind.values)
    if other_n > 0:
        labels.append("Others"); values.append(other_n)
    n = len(ind)
    # Use professional palette instead of tab20 (avoids jarring reds)
    cols = [np.array((*[int(PROF_PALETTE[i % len(PROF_PALETTE)].lstrip("#")[j*2:j*2+2], 16)/255
                        for j in range(3)], 1.0))
            for i in range(n)]
    if other_n > 0:
        cols.append(np.array([0.6, 0.6, 0.6, 1.0]))
    return labels, values, cols


def plot_a(ax_pie, ax_leg, df: pd.DataFrame) -> None:
    labels, values, colors = build_pie(df)
    total = sum(values)

    wedges, _ = ax_pie.pie(
        values, labels=None, colors=colors, startangle=90, radius=1.0,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax_pie.set_xlim(-1.25, 1.25); ax_pie.set_ylim(-1.25, 1.25)
    ax_pie.set_axis_off()

    # legend on ax_leg — no callout lines, no overlap
    ax_leg.set_axis_off()
    n  = len(labels)
    y0 = 0.97
    dy = 0.97 / max(n, 1)
    sh = dy * 0.55
    sw = 0.09
    xsq, xtx = 0.01, 0.01 + sw + 0.07

    for i, (sp, val, col) in enumerate(zip(labels, values, colors)):
        pct = 100.0 * val / total
        y   = y0 - i * dy
        rect = mpatches.FancyBboxPatch(
            (xsq, y - sh / 2), sw, sh, boxstyle="square,pad=0",
            facecolor=col, edgecolor="none",
            transform=ax_leg.transAxes, clip_on=False)
        ax_leg.add_patch(rect)
        italic = (sp != "Others")
        ax_leg.text(xtx, y + sh * 0.12, sp,
                    transform=ax_leg.transAxes, va="bottom", ha="left",
                    fontsize=FSLEG, fontfamily=FONT,
                    fontstyle="italic" if italic else "normal")
        ax_leg.text(xtx, y - sh * 0.12, f"{val} ({pct:.1f}%)",
                    transform=ax_leg.transAxes, va="top", ha="left",
                    fontsize=FSCT, fontfamily=FONT, color="#555555")


# ── panel B ───────────────────────────────────────────────────────────────────
def plot_b(ax, df: pd.DataFrame) -> None:
    sc   = df["Amino_Acid"].value_counts().sort_values(ascending=False)
    aas  = list(sc.index);  vals = list(sc.values);  total = sum(vals)
    x    = np.arange(len(aas))
    bars = ax.bar(x, vals, color="steelblue", edgecolor="black",
                  linewidth=0.6, alpha=0.75, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(aas, fontsize=FST, fontfamily=FONT)
    ax.set_xlabel(r"Amino acids aligned to $\mathit{E.\ coli}$ position 20",
                  fontsize=12, fontfamily=FONT)
    ax.set_ylabel("Count", fontsize=12, fontfamily=FONT)
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, ymax * 1.25)
    ax.tick_params(axis="both", labelsize=FST, width=FPT)
    for bar, cnt in zip(bars, vals):
        pct = 100.0 * cnt / total
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                f"{cnt} ({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontfamily=FONT)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"): ax.spines[sp].set_linewidth(FPT)


# ── panels C–E ────────────────────────────────────────────────────────────────
def load_rama(csv_path: Path, pdb_ids: set[str]):
    phi_v, psi_v, aas = [], [], []
    aa_map = {"MET": "M", "LEU": "L", "ILE": "I"}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("PDB_ID", "").strip().upper() not in pdb_ids:
                continue
            ps = row.get("Phi", "").strip(); qs = row.get("Psi", "").strip()
            aa = row.get("Amino_Acid", "").strip().upper()
            faa = row.get("Found_Amino_Acid", "").strip().upper()
            if not (ps and qs and ps.lower() != "none"): continue
            try: phi, psi = float(ps), float(qs)
            except ValueError: continue
            code = (aa if aa in "MLI" and len(aa)==1
                    else aa_map.get(aa) or aa_map.get(faa))
            if code:
                phi_v.append(phi); psi_v.append(psi); aas.append(code)
    return phi_v, psi_v, aas, {a: aas.count(a) for a in "MLI"}


def plot_c(ax, phi_v, psi_v, aas, counts) -> None:
    for aa in "MLI":
        xs = [phi_v[i] for i, a in enumerate(aas) if a == aa]
        ys = [psi_v[i] for i, a in enumerate(aas) if a == aa]
        if xs:
            ax.scatter(xs, ys, alpha=0.6, s=28, label=f"{aa} (n={counts[aa]})",
                       c=AA_C[aa], edgecolors="black", linewidths=0.3)
    ax.set_xlabel(PHIL, fontsize=FSA); ax.set_ylabel(PSIL, fontsize=FSA)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=FST, frameon=False, prop={"family": FONT})
    ax.tick_params(labelsize=FST); sax(ax)


def plot_d(ax, phi_v, aas, counts) -> None:
    bins = np.linspace(-180, 0, 37); xs = np.linspace(-180, 0, 200)
    for aa in "MLI":
        vals = np.array([phi_v[i] for i, a in enumerate(aas) if a == aa])
        if len(vals) > 1:
            ax.hist(vals, bins=bins, alpha=0.3, density=True,
                    label=f"{aa} (n={counts[aa]})", color=AA_C[aa],
                    edgecolor="black", linewidth=0.3)
            try: ax.plot(xs, gaussian_kde(vals)(xs), color=AA_C[aa], lw=1.0)
            except Exception: pass
    ax.set_xlabel(PHIL, fontsize=FSA)
    ax.set_ylabel("Probability Density", fontsize=FSA)
    ax.set_xlim(-180, 0); ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=FST, frameon=False, prop={"family": FONT})
    ax.tick_params(labelsize=FST); sax(ax)


def plot_e(ax, psi_v, aas, counts) -> None:
    bins = np.linspace(0, 180, 37); xs = np.linspace(0, 180, 200)
    for aa in "MLI":
        vals = np.array([psi_v[i] for i, a in enumerate(aas) if a == aa])
        if len(vals) > 1:
            ax.hist(vals, bins=bins, alpha=0.3, density=True,
                    label=f"{aa} (n={counts[aa]})", color=AA_C[aa],
                    edgecolor="black", linewidth=0.3)
            try: ax.plot(xs, gaussian_kde(vals)(xs), color=AA_C[aa], lw=1.0)
            except Exception: pass
    ax.set_xlabel(PSIL, fontsize=FSA)
    ax.set_ylabel("Probability Density", fontsize=FSA)
    ax.set_xlim(0, 180); ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=FST, frameon=False, prop={"family": FONT})
    ax.tick_params(labelsize=FST); sax(ax)


# ── panels F–H ────────────────────────────────────────────────────────────────
def load_violin(classif_csv: Path, dihedral_csv: Path, pdb_ids: set[str]) -> dict:
    psi_map: dict[str, tuple] = {}
    with open(dihedral_csv) as f:
        for row in csv.DictReader(f):
            pid = row.get("PDB_ID", "").strip().upper()
            if pid not in pdb_ids: continue
            aa = row.get("Amino_Acid", "").strip().upper()
            ps = row.get("Psi", "").strip()
            if row.get("Status", "").strip() == "Success" and aa in "MLI" \
               and ps and ps.lower() != "none":
                try: psi_map[pid] = (float(ps), aa)
                except ValueError: pass
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


def plot_fgh(ax, aa: str, data: dict, threshold: float = 128.39) -> None:
    cats    = ["NAP_NDP_FOL", "NAP_NDP_MTX_TOP", "other"]
    clrs    = {"NAP_NDP_FOL": "steelblue", "NAP_NDP_MTX_TOP": "orange", "other": "gray"}
    # Break long labels onto two lines so they don't overlap
    lbl_map = {"NAP_NDP_FOL":    "NAP/NDP\n−FOL",
               "NAP_NDP_MTX_TOP":"NAP/NDP\n−MTX/TOP",
               "other":          "Other"}
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [f"{lbl_map[c]}\n(n={len(data[aa][c])})" for c in cats], fontsize=10)
    ax.set_xlim(-0.5, 2.5)

    vdata, vpos, vclrs = [], [], []
    for i, cat in enumerate(cats):
        d = data[aa][cat]
        if d: vdata.append(d); vpos.append(i); vclrs.append(clrs[cat])
    if vdata:
        parts = ax.violinplot(vdata, positions=vpos, widths=0.6,
                              showmeans=False, showmedians=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(vclrs[j]); pc.set_alpha(0.4)
            pc.set_edgecolor("black"); pc.set_linewidth(1.0)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_edgecolor("black"); parts[key].set_linewidth(1.0)

    for i, cat in enumerate(cats):
        d = data[aa][cat]
        if not d: continue
        q25, med, q75 = np.percentile(d, [25, 50, 75])
        ax.hlines([q25, med, q75], i-0.18, i+0.18, colors="black", lw=1.0)

    valid = {c: data[aa][c] for c in cats if len(data[aa][c]) > 1}
    clist = list(valid.keys())
    comps = []
    for i in range(len(clist)):
        for j in range(i+1, len(clist)):
            c1, c2 = clist[i], clist[j]
            try:
                _, p = mannwhitneyu(valid[c1], valid[c2], alternative="two-sided")
                ps = ("n.s." if p >= 0.05 else
                      "p<0.001" if p < 0.001 else f"p={p:.3f}")
                comps.append({"p1": cats.index(c1), "p2": cats.index(c2),
                               "pstr": ps, "span": abs(cats.index(c2)-cats.index(c1))})
            except Exception: pass
    comps.sort(key=lambda x: (-x["span"], x["p1"]))
    y_base, y_off = 35, [0, 10, 20]
    for idx, comp in enumerate(comps):
        p1, p2 = comp["p1"], comp["p2"]
        by = y_base + y_off[idx % 3]
        ax.plot([p1,p1],[by,by+3],"k-",lw=1.0)
        ax.plot([p2,p2],[by,by+3],"k-",lw=1.0)
        ax.plot([p1,p2],[by,by],"k-",lw=1.0)
        ax.text((p1+p2)/2, by-2, comp["pstr"],
                ha="center", va="top", fontsize=FST, fontfamily=FONT)

    ax.axhline(threshold, color="red", linestyle="--", lw=1.0, alpha=0.7,
               label=f"Threshold ({threshold}°)")
    ax.set_ylabel(PSIL, fontsize=FSA)
    ax.legend(loc="upper left", fontsize=FST, frameon=False, prop={"family": FONT})
    ax.set_ylim(0, 180); ax.tick_params(labelsize=FST); sax(ax)


# ── figure assembly ───────────────────────────────────────────────────────────
# AB section is identical to summary_panel_AB_v2.png (14" × 5.8").
FW, FH, AB_H = 15.5, 16.5, 5.8
s    = AB_H / FH
y_ab = (FH - AB_H) / FH

def ab_ax(x0, y0_sa, w, h_sa):
    return [x0, y_ab + y0_sa * s, w, h_sa * s]


def make_figure(domain: str, domain_df: pd.DataFrame,
                phi_v, psi_v, aas, counts, violin_data) -> plt.Figure:
    fig = plt.figure(figsize=(FW, FH))

    # ── A: pie + legend ──────────────────────────────────────────────────────
    ax_pie = fig.add_axes(ab_ax(0.01, 0.01, 0.41, 0.97))
    ax_pie.set_aspect("equal")
    ax_leg = fig.add_axes(ab_ax(0.44, 0.04, 0.22, 0.92))
    ax_bar = fig.add_axes(ab_ax(0.71, 0.13, 0.27, 0.78))

    # ── C–H: GridSpec ────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=y_ab - 0.02,
                           bottom=0.03,
                           hspace=0.30, wspace=0.34)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[1, 2])

    # ── plot ─────────────────────────────────────────────────────────────────
    plot_a(ax_pie, ax_leg, domain_df)
    plot_b(ax_bar, domain_df)

    if phi_v:
        plot_c(ax_c, phi_v, psi_v, aas, counts)
        plot_d(ax_d, phi_v, aas, counts)
        plot_e(ax_e, psi_v, aas, counts)
    else:
        for ax in (ax_c, ax_d, ax_e):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
    for ax, aa in [(ax_f,"M"),(ax_g,"L"),(ax_h,"I")]:
        plot_fgh(ax, aa, violin_data)

    # ── panel labels ─────────────────────────────────────────────────────────
    y_lbl = y_ab + 0.99 * s
    fig.text(0.01, y_lbl, "a", fontsize=FSLP, fontweight=FWP,
             fontfamily=FONT, va="top", ha="left")
    fig.text(0.70, y_lbl, "b", fontsize=FSLP, fontweight=FWP,
             fontfamily=FONT, va="top", ha="left")
    for ax, lbl in [(ax_c,"c"),(ax_d,"d"),(ax_e,"e"),
                    (ax_f,"f"),(ax_g,"g"),(ax_h,"h")]:
        plbl(ax, lbl)

    fig.suptitle(DOMAIN_TITLES[domain], fontsize=16, fontfamily=FONT,
                 fontweight="bold", y=0.998)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    setup()
    domain_sets = build_domain_sets(DATA_NEW)
    df_all = pd.read_csv(DATA_NEW)

    for domain, pdb_ids in domain_sets.items():
        print(f"\n── {DOMAIN_TITLES[domain]} (n={len(pdb_ids)}) ──")
        domain_df = df_all[df_all["PDB_ID"].str.upper().isin(pdb_ids)].copy()
        phi_v, psi_v, aas, counts = load_rama(DIHEDRAL, pdb_ids)
        print(f"   M={counts['M']}, L={counts['L']}, I={counts['I']}")
        violin_data = load_violin(CLASSIF, DIHEDRAL, pdb_ids)

        fig = make_figure(domain, domain_df, phi_v, psi_v, aas, counts, violin_data)
        out_base = HERE / DOMAIN_OUTNAMES[domain]
        for ext, kw in [("png", {"dpi": 300, "bbox_inches": "tight"}),
                        ("pdf", {"bbox_inches": "tight"})]:
            out = out_base.with_suffix(f".{ext}")
            fig.savefig(out, **kw)
            print(f"   Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
