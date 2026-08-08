#!/usr/bin/env python3
"""
Combine species pie chart, AA distribution, Ramachandran (M/L/I), and
psi-by-ligand violin plots into a single 3x3 panel figure.

Layout:
  (1,1)-(1,2) Species pie chart (spans two columns)
  (1,3) AA distribution at pos 20
  (2,1-3) Ramachandran scatter, phi KDE, psi KDE (M/L/I)
  (3,1-3) Psi violin plots by ligand category (M, L, I)
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import ConnectionPatch
from scipy.stats import gaussian_kde, mannwhitneyu

ROOT = Path(__file__).resolve().parent
MSA_DIR = ROOT / "msa"
ANALYSIS_DIR = ROOT / "analysis"

sys.path.insert(0, str(MSA_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
from plot_psi_by_ligand_category import categorize_entry  # noqa: E402
from plot_species_pie import extract_species_from_description  # noqa: E402

AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
TARGET_RESIDUES = {"M", "L", "I", "MET", "LEU", "ILE"}
AA_COLORS = {"M": "#1f77b4", "L": "#ff7f0e", "I": "#2ca02c"}
AA_LABELS = {"M": "M", "L": "L", "I": "I"}
FONT_FAMILY = "Arial"
PHI_LABEL = r"$\phi$ (°)"
PSI_LABEL = r"$\psi$ (°)"

# Line widths (points)
FRAME_WIDTH_PT = 0.5          # axis spines and tick marks
LINE_WIDTH_AXIS_REF = 0.5     # Ramachandran center lines; phi/psi x = 0
LINE_WIDTH_SCATTER_EDGE = 0.3
LINE_WIDTH_HIST_EDGE = 0.3
LINE_WIDTH_KDE = 1.0
LINE_WIDTH_VIOLIN_EDGE = 1.0
LINE_WIDTH_QUARTILE = 1.0
LINE_WIDTH_STAT_BRACKET = 1.0
LINE_WIDTH_THRESHOLD = 1.0

# Full-page export (Word / print): ~17 cm wide (A4 text width)
FIG_WIDTH_IN = 6.7
FIG_HEIGHT_IN = 8.0
EXPORT_DPI = 300

# Font sizes (full-page layout)
PANEL_LABEL_FONTSIZE = 20
PANEL_LABEL_FONTWEIGHT = "bold"
# Panel a (pie chart)
FONT_SIZE_PIE_LABEL = 10
FONT_SIZE_PIE_PCT = 8
# Panel b
FONT_SIZE_PANEL_B_AXIS = 12
FONT_SIZE_PANEL_B_TICK = 11
FONT_SIZE_PANEL_B_BAR_LABEL = 11
# Panels c–h
FONT_SIZE_PANEL_CH_AXIS = 14
FONT_SIZE_PANEL_CH_TICK = 13
FONT_SIZE_PANEL_CH_LEGEND = 13
FONT_SIZE_PANEL_CH_ANNOT = 13


def setup_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "font.size": FONT_SIZE_PANEL_CH_TICK,
        "font.weight": "normal",
        "axes.labelsize": FONT_SIZE_PANEL_CH_AXIS,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT_SIZE_PANEL_CH_TICK,
        "ytick.labelsize": FONT_SIZE_PANEL_CH_TICK,
        "legend.fontsize": FONT_SIZE_PANEL_CH_LEGEND,
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.edgecolor": "black",
        "axes.linewidth": FRAME_WIDTH_PT,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": FRAME_WIDTH_PT,
        "ytick.major.width": FRAME_WIDTH_PT,
        "xtick.minor.width": FRAME_WIDTH_PT,
        "ytick.minor.width": FRAME_WIDTH_PT,
    })


def _set_arial(obj) -> None:
    if hasattr(obj, "set_fontfamily"):
        obj.set_fontfamily(FONT_FAMILY)
    if hasattr(obj, "set_fontweight"):
        obj.set_fontweight("normal")


def add_panel_label(ax, label: str, x: float = -0.18, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight=PANEL_LABEL_FONTWEIGHT,
        fontfamily=FONT_FAMILY,
        va="top",
        ha="left",
    )


def style_axes(ax, twin_axes=None) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(FRAME_WIDTH_PT)
    ax.tick_params(
        axis="both",
        colors="black",
        width=FRAME_WIDTH_PT,
        labelcolor="black",
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        _set_arial(label)
    _set_arial(ax.xaxis.label)
    _set_arial(ax.yaxis.label)
    _set_arial(ax.title)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        for text in leg.get_texts():
            _set_arial(text)

    if twin_axes:
        for tax in twin_axes:
            tax.grid(False)
            tax.tick_params(
                axis="y",
                colors="black",
                width=FRAME_WIDTH_PT,
                labelcolor=tax.yaxis.label.get_color(),
            )
            for label in tax.get_yticklabels():
                _set_arial(label)
            _set_arial(tax.yaxis.label)


def find_aligned_position(reference_id: str, reference_pos: int, aligned_fasta: Path) -> int:
    aligned_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(aligned_fasta, "fasta")}
    ref_seq = None
    for seq_id, seq in aligned_seqs.items():
        if seq_id.startswith(reference_id):
            ref_seq = seq
            break
    if ref_seq is None:
        raise ValueError(f"Reference sequence {reference_id} not found in alignment")

    non_gap_count = 0
    for i, aa in enumerate(ref_seq):
        if aa != "-":
            non_gap_count += 1
            if non_gap_count == reference_pos:
                return i
    raise ValueError(f"Could not map reference position {reference_pos} in {reference_id}")


def get_aa_counts_at_position(aligned_fasta: Path, aligned_pos: int, reference_id: str) -> tuple[dict, str]:
    amino_acids = []
    ref_aa = None
    for rec in SeqIO.parse(aligned_fasta, "fasta"):
        seq = str(rec.seq)
        aa = seq[aligned_pos] if aligned_pos < len(seq) else "-"
        amino_acids.append(aa)
        if rec.id.startswith(reference_id):
            ref_aa = aa
    return dict(Counter(amino_acids)), ref_aa


def _stagger_pie_label_radii(
    angles_deg: list[float],
    base_radius: float = 1.55,
    min_sep_deg: float = 12.0,
    step: float = 0.14,
) -> list[float]:
    """Increase label radius where neighboring wedge angles are too close."""
    n = len(angles_deg)
    if n == 0:
        return []
    radii = [base_radius] * n
    order = np.argsort(angles_deg)
    for k in range(1, n):
        i_curr = order[k]
        i_prev = order[k - 1]
        if angles_deg[i_curr] - angles_deg[i_prev] < min_sep_deg:
            radii[i_curr] = max(radii[i_curr], radii[i_prev] + step)
    i_first = order[0]
    i_last = order[-1]
    wrap_sep = (360.0 - angles_deg[i_last]) + angles_deg[i_first]
    if wrap_sep < min_sep_deg:
        radii[i_first] = max(radii[i_first], radii[i_last] + step)
    return radii


def _add_pie_species_callout(
    ax,
    wedge,
    val: int,
    total: int,
    species: str,
    text_radius: float,
    pie_radius: float,
) -> None:
    ang = np.deg2rad((wedge.theta2 + wedge.theta1) / 2.0)
    x, y = np.cos(ang), np.sin(ang)
    pct = 100.0 * val / total if total else 0.0
    xy = (0.9 * pie_radius * x, 0.9 * pie_radius * y)
    tx, ty = text_radius * x, text_radius * y

    count_area = TextArea(
        f"{val} ({pct:.1f}%) ",
        textprops={
            "fontsize": FONT_SIZE_PIE_PCT,
            "fontfamily": FONT_FAMILY,
            "fontweight": "normal",
        },
    )
    species_props = {
        "fontsize": FONT_SIZE_PIE_LABEL,
        "fontfamily": FONT_FAMILY,
        "fontweight": "normal",
    }
    if species != "Others":
        species_props["style"] = "italic"
    species_area = TextArea(species, textprops=species_props)
    box = HPacker(children=[count_area, species_area], align="center", pad=0, sep=0)

    ax.add_artist(
        AnnotationBbox(
            box,
            (tx, ty),
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
    )
    ax.add_artist(
        ConnectionPatch(
            xyA=xy,
            xyB=(tx, ty),
            coordsA="data",
            coordsB="data",
            arrowstyle="-",
            linewidth=0.5,
            color="black",
            shrinkA=2,
            shrinkB=4,
        )
    )


def plot_species_pie_on_ax(ax, input_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    df["organism"] = df["Description"].apply(extract_species_from_description)
    df = df[df["organism"].str.strip() != ""]

    def normalize_to_two_words(name):
        words = str(name).split()
        return " ".join(words[:2]) if len(words) > 2 else name

    df["organism_normalized"] = df["organism"].apply(normalize_to_two_words)
    species_counts = df["organism_normalized"].value_counts()

    species_individual = species_counts[species_counts >= 10]
    species_others = species_counts[species_counts < 10]
    other_count = species_others.sum() if len(species_others) > 0 else 0

    pie_data = list(species_individual.values)
    pie_labels = list(species_individual.index)
    if other_count > 0:
        pie_data.append(other_count)
        pie_labels.append("Others")

    n_individual = len(species_individual)
    if n_individual <= 20:
        individual_colors = plt.cm.tab20(np.linspace(0, 1, n_individual))
    elif n_individual <= 32:
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, n_individual - 20))
        individual_colors = np.vstack([colors_tab20, colors_set3])
    elif n_individual <= 41:
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors_pastel1 = plt.cm.Pastel1(np.linspace(0, 1, n_individual - 32))
        individual_colors = np.vstack([colors_tab20, colors_set3, colors_pastel1])
    else:
        individual_colors = plt.cm.turbo(np.linspace(0, 1, n_individual))

    colors = list(individual_colors) + (["#808080"] if other_count > 0 else [])

    pie_radius = 1.2
    total = sum(pie_data)
    wedges, _ = ax.pie(
        pie_data,
        labels=None,
        colors=colors,
        startangle=90,
        radius=pie_radius,
    )

    angles_deg = [(w.theta2 + w.theta1) / 2.0 for w in wedges]
    label_radii = _stagger_pie_label_radii(
        angles_deg,
        base_radius=1.5,
        min_sep_deg=13.0,
        step=0.16,
    )
    for wedge, val, species, text_r in zip(wedges, pie_data, pie_labels, label_radii):
        _add_pie_species_callout(ax, wedge, val, total, species, text_r, pie_radius)

    label_margin = 2.35
    ax.set_xlim(-label_margin, label_margin)
    ax.set_ylim(-label_margin, label_margin)
    ax.set_aspect("equal")

    style_axes(ax)


def plot_aa_distribution(ax, aa_counts: dict) -> None:
    items = sorted(aa_counts.items(), key=lambda x: (-x[1], x[0]))
    sorted_aas = [aa for aa, _ in items]
    counts = [c for _, c in items]
    total = sum(counts)
    percentages = [(c / total * 100) if total > 0 else 0 for c in counts]
    x = np.arange(len(sorted_aas))

    bars = ax.bar(x, counts, color="steelblue", edgecolor="black", alpha=0.7, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_aas, fontsize=FONT_SIZE_PANEL_B_TICK)
    ax.set_xlabel(
        r"Amino acids aligned to $\mathit{E.\ coli}$ position 20",
        fontsize=FONT_SIZE_PANEL_B_AXIS,
    )
    ax.set_ylabel("Count", fontsize=FONT_SIZE_PANEL_B_AXIS)
    ax.set_ylim(0, 350)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_PANEL_B_TICK)

    ymax = max(counts) if counts else 1
    for bar, count, pct in zip(bars, counts, percentages):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.02,
                f"{count} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_PANEL_B_BAR_LABEL,
                fontfamily=FONT_FAMILY,
            )

    style_axes(ax)


def load_ramachandran_mli_data(csv_path: Path) -> tuple[list, list, list, dict[str, int]]:
    phi_values, psi_values, amino_acids = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phi_str = row.get("Phi", "").strip()
            psi_str = row.get("Psi", "").strip()
            aa = row.get("Amino_Acid", "").strip().upper()
            found_aa = row.get("Found_Amino_Acid", "").strip().upper()

            is_target = (
                aa in TARGET_RESIDUES
                or found_aa in TARGET_RESIDUES
                or (len(aa) == 1 and AA_1_TO_3.get(aa) in TARGET_RESIDUES)
                or found_aa in {"MET", "LEU", "ILE"}
            )

            if not (phi_str and psi_str and phi_str.lower() != "none" and psi_str.lower() != "none"):
                continue

            try:
                phi = float(phi_str)
                psi = float(psi_str)
            except ValueError:
                continue

            if is_target:
                if len(aa) == 1 and aa in "MLI":
                    code = aa
                elif aa in ["MET", "LEU", "ILE"]:
                    code = {"MET": "M", "LEU": "L", "ILE": "I"}[aa]
                elif found_aa in ["MET", "LEU", "ILE"]:
                    code = {"MET": "M", "LEU": "L", "ILE": "I"}[found_aa]
                else:
                    code = aa[0] if aa else "?"
                phi_values.append(phi)
                psi_values.append(psi)
                amino_acids.append(code)

    counts = {aa: sum(1 for a in amino_acids if a == aa) for aa in "MLI"}
    return phi_values, psi_values, amino_acids, counts


def plot_ramachandran_scatter(ax, phi_values, psi_values, amino_acids, counts) -> None:
    for aa in ["M", "L", "I"]:
        aa_phi = [phi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa]
        aa_psi = [psi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa]
        if aa_phi:
            ax.scatter(
                aa_phi,
                aa_psi,
                alpha=0.6,
                s=28,
                label=f"{AA_LABELS[aa]} (n={counts[aa]})",
                c=AA_COLORS[aa],
                edgecolors="black",
                linewidths=LINE_WIDTH_SCATTER_EDGE,
            )
    ax.set_xlabel(PHI_LABEL, fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_ylabel(PSI_LABEL, fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=LINE_WIDTH_AXIS_REF, alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=LINE_WIDTH_AXIS_REF, alpha=0.5)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_PANEL_CH_LEGEND, frameon=False, prop={"family": FONT_FAMILY, "weight": "normal"})
    ax.tick_params(axis="both", labelsize=FONT_SIZE_PANEL_CH_TICK)
    style_axes(ax)


def plot_phi_distribution(ax, phi_values, amino_acids, counts) -> None:
    x_phi = np.linspace(-180, 0, 200)
    phi_bins = np.linspace(-180, 0, 37)
    for aa in ["M", "L", "I"]:
        aa_phi = np.array([phi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa])
        if len(aa_phi) > 1:
            ax.hist(
                aa_phi,
                bins=phi_bins,
                alpha=0.3,
                density=True,
                label=f"{AA_LABELS[aa]} (n={counts[aa]})",
                color=AA_COLORS[aa],
                edgecolor="black",
                linewidth=LINE_WIDTH_HIST_EDGE,
            )
            try:
                kde = gaussian_kde(aa_phi)
                ax.plot(x_phi, kde(x_phi), color=AA_COLORS[aa], linewidth=LINE_WIDTH_KDE, alpha=1.0)
            except Exception:
                pass

    ax.set_xlabel(PHI_LABEL, fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_ylabel("Probability Density", fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_xlim(-180, 0)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_PANEL_CH_LEGEND, frameon=False, prop={"family": FONT_FAMILY, "weight": "normal"})
    ax.axvline(x=0, color="k", linestyle="-", linewidth=LINE_WIDTH_AXIS_REF, alpha=0.5)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_PANEL_CH_TICK)
    style_axes(ax)


def plot_psi_distribution(ax, psi_values, amino_acids, counts) -> None:
    x_psi = np.linspace(0, 180, 200)
    psi_bins = np.linspace(0, 180, 37)
    for aa in ["M", "L", "I"]:
        aa_psi = np.array([psi_values[i] for i in range(len(amino_acids)) if amino_acids[i] == aa])
        if len(aa_psi) > 1:
            ax.hist(
                aa_psi,
                bins=psi_bins,
                alpha=0.3,
                density=True,
                label=f"{AA_LABELS[aa]} (n={counts[aa]})",
                color=AA_COLORS[aa],
                edgecolor="black",
                linewidth=LINE_WIDTH_HIST_EDGE,
            )
            try:
                kde = gaussian_kde(aa_psi)
                ax.plot(x_psi, kde(x_psi), color=AA_COLORS[aa], linewidth=LINE_WIDTH_KDE, alpha=1.0)
            except Exception:
                pass

    ax.set_xlabel(PSI_LABEL, fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_ylabel("Probability Density", fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.set_xlim(0, 180)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_PANEL_CH_LEGEND, frameon=False, prop={"family": FONT_FAMILY, "weight": "normal"})
    ax.axvline(x=0, color="k", linestyle="-", linewidth=LINE_WIDTH_AXIS_REF, alpha=0.5)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_PANEL_CH_TICK)
    style_axes(ax)


def load_psi_ligand_data(classification_csv: Path, dihedral_csv: Path) -> dict:
    psi_values = {}
    with open(dihedral_csv, "r") as f:
        for row in csv.DictReader(f):
            pdb_id = row.get("PDB_ID", "").strip().upper()
            aa = row.get("Amino_Acid", "").strip().upper()
            psi_str = row.get("Psi", "").strip()
            status = row.get("Status", "").strip()
            if status == "Success" and aa in {"M", "L", "I"} and psi_str and psi_str.lower() != "none":
                try:
                    psi_values[pdb_id] = float(psi_str)
                except (ValueError, TypeError):
                    pass

    categories = ["NAP_NDP_FOL", "NAP_NDP_MTX_TOP", "other"]
    data = {aa: {cat: [] for cat in categories} for aa in ["M", "L", "I"]}

    with open(classification_csv, "r") as f:
        for row in csv.DictReader(f):
            pdb_id = row.get("PDB_ID", "").strip().upper()
            pos20_aa = row.get("Pos20_Amino_Acid", "").strip().upper()
            cofactor_str = row.get("Cofactor", "").strip()
            substrate_str = row.get("Substrate_Antibiotic", "").strip()

            aa_group = None
            if pos20_aa == "M" and pdb_id in psi_values:
                aa_group = "M"
            elif pos20_aa == "L":
                aa_group = "L"
            elif pos20_aa == "I":
                aa_group = "I"

            if aa_group and pdb_id in psi_values:
                psi_val = psi_values[pdb_id]
                if psi_val >= 50:
                    category = categorize_entry(cofactor_str, substrate_str)
                    data[aa_group][category].append(psi_val)

    return data


def plot_violin_on_ax(ax, aa: str, data: dict, psi_threshold: float = 128.39) -> None:
    categories = ["NAP_NDP_FOL", "NAP_NDP_MTX_TOP", "other"]
    category_colors = {"NAP_NDP_FOL": "steelblue", "NAP_NDP_MTX_TOP": "orange", "other": "gray"}
    category_labels = {
        "NAP_NDP_FOL": "NAP/NDP-FOL",
        "NAP_NDP_MTX_TOP": "NAP/NDP-MTX/TOP",
        "other": "Other",
    }

    plot_data = []
    plot_labels = []
    plot_colors = []
    for cat in categories:
        psi_vals = data[aa][cat]
        plot_data.append(psi_vals if psi_vals else [])
        plot_labels.append(f"{category_labels[cat]}\n(n={len(psi_vals)})")
        plot_colors.append(category_colors[cat])

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(plot_labels, fontsize=FONT_SIZE_PANEL_CH_TICK)
    ax.set_xlim(-0.5, 2.5)

    plot_data_for_violin = []
    positions_for_violin = []
    for i, d in enumerate(plot_data):
        if len(d) > 0:
            plot_data_for_violin.append(d)
            positions_for_violin.append(i)

    if plot_data_for_violin:
        parts = ax.violinplot(
            plot_data_for_violin,
            positions=positions_for_violin,
            widths=0.6,
            showmeans=False,
            showmedians=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            orig_pos = positions_for_violin[i]
            pc.set_facecolor(plot_colors[orig_pos])
            pc.set_alpha(0.4)
            pc.set_edgecolor("black")
            pc.set_linewidth(LINE_WIDTH_VIOLIN_EDGE)
        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if partname in parts:
                parts[partname].set_edgecolor("black")
                parts[partname].set_linewidth(LINE_WIDTH_VIOLIN_EDGE)

    for i, cat in enumerate(categories):
        psi_vals = data[aa][cat]
        if not psi_vals:
            continue
        q25, median, q75 = np.percentile(psi_vals, [25, 50, 75])
        ax.hlines([q25, median, q75], i - 0.18, i + 0.18, colors="black", linewidth=LINE_WIDTH_QUARTILE)

    psi_data = {cat: data[aa][cat] for cat in categories if len(data[aa][cat]) > 0}
    cat_list = list(psi_data.keys())
    comparisons = []
    for i in range(len(cat_list)):
        for j in range(i + 1, len(cat_list)):
            cat1, cat2 = cat_list[i], cat_list[j]
            vals1, vals2 = psi_data[cat1], psi_data[cat2]
            if len(vals1) > 1 and len(vals2) > 1:
                try:
                    _, p_value = mannwhitneyu(vals1, vals2, alternative="two-sided")
                    if p_value >= 0.05:
                        p_str = "n.s."
                    elif p_value < 0.001:
                        p_str = "p<0.001"
                    else:
                        p_str = f"p={p_value:.3f}"
                    comparisons.append({
                        "pos1": categories.index(cat1),
                        "pos2": categories.index(cat2),
                        "p_str": p_str,
                        "span": abs(categories.index(cat2) - categories.index(cat1)),
                    })
                except Exception:
                    pass

    comparisons.sort(key=lambda x: (-x["span"], x["pos1"]))
    y_label = 35
    y_offsets = [0, 10, 20]
    for idx, comp in enumerate(comparisons):
        pos1, pos2 = comp["pos1"], comp["pos2"]
        bracket_y = y_label + y_offsets[idx % len(y_offsets)]
        ax.plot([pos1, pos1], [bracket_y, bracket_y + 3], "k-", linewidth=LINE_WIDTH_STAT_BRACKET)
        ax.plot([pos2, pos2], [bracket_y, bracket_y + 3], "k-", linewidth=LINE_WIDTH_STAT_BRACKET)
        ax.plot([pos1, pos2], [bracket_y, bracket_y], "k-", linewidth=LINE_WIDTH_STAT_BRACKET)
        ax.text(
            (pos1 + pos2) / 2,
            bracket_y - 2,
            comp["p_str"],
            ha="center",
            va="top",
            fontsize=FONT_SIZE_PANEL_CH_ANNOT,
            fontweight="normal",
            fontfamily=FONT_FAMILY,
        )

    ax.axhline(
        y=psi_threshold,
        color="red",
        linestyle="--",
        linewidth=LINE_WIDTH_THRESHOLD,
        label=f"Threshold ({psi_threshold}°)",
        alpha=0.7,
    )
    ax.set_ylabel(PSI_LABEL, fontsize=FONT_SIZE_PANEL_CH_AXIS)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_PANEL_CH_LEGEND, frameon=False, prop={"family": FONT_FAMILY, "weight": "normal"})
    ax.set_ylim(0, 180)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_PANEL_CH_TICK)
    style_axes(ax)


def plot_summary_panel(
    species_csv: Path,
    aligned_fasta: Path,
    dihedral_csv: Path,
    classification_csv: Path,
    output_path: Path,
    reference_id: str = "1DDR",
    reference_pos: int = 20,
    psi_threshold: float = 128.39,
) -> None:
    aligned_pos = find_aligned_position(reference_id, reference_pos, aligned_fasta)
    aa_counts, _ = get_aa_counts_at_position(aligned_fasta, aligned_pos, reference_id)
    phi_values, psi_values, amino_acids, counts = load_ramachandran_mli_data(dihedral_csv)
    ligand_data = load_psi_ligand_data(classification_csv, dihedral_csv)

    setup_matplotlib_style()
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.15, wspace=0.32)

    ax_pie = fig.add_subplot(gs[0, 0:2])
    ax_aa = fig.add_subplot(gs[0, 2])

    ax_rama = fig.add_subplot(gs[1, 0])
    ax_phi = fig.add_subplot(gs[1, 1])
    ax_psi_hist = fig.add_subplot(gs[1, 2])

    ax_vm = fig.add_subplot(gs[2, 0])
    ax_vl = fig.add_subplot(gs[2, 1])
    ax_vi = fig.add_subplot(gs[2, 2])

    plot_species_pie_on_ax(ax_pie, species_csv)
    plot_aa_distribution(ax_aa, aa_counts)

    if phi_values:
        plot_ramachandran_scatter(ax_rama, phi_values, psi_values, amino_acids, counts)
        plot_phi_distribution(ax_phi, phi_values, amino_acids, counts)
        plot_psi_distribution(ax_psi_hist, psi_values, amino_acids, counts)
    else:
        for ax in (ax_rama, ax_phi, ax_psi_hist):
            ax.text(0.5, 0.5, "No M/L/I dihedral data", ha="center", va="center", transform=ax.transAxes)

    plot_violin_on_ax(ax_vm, "M", ligand_data, psi_threshold)
    plot_violin_on_ax(ax_vl, "L", ligand_data, psi_threshold)
    plot_violin_on_ax(ax_vi, "I", ligand_data, psi_threshold)

    for ax, label in zip(
        [ax_pie, ax_aa, ax_rama, ax_phi, ax_psi_hist, ax_vm, ax_vl, ax_vi],
        list("abcdefgh"),
    ):
        add_panel_label(ax, label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".pdf":
        plt.savefig(output_path, bbox_inches="tight", format="pdf")
        print(f"Saved summary panel to {output_path}")
    elif output_path.suffix.lower() == ".png":
        plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
        px_w = int(FIG_WIDTH_IN * EXPORT_DPI)
        px_h = int(FIG_HEIGHT_IN * EXPORT_DPI)
        print(f"Saved summary panel to {output_path} ({px_w}×{px_h} px at {EXPORT_DPI} DPI)")
        print(f"Saved summary panel to {pdf_path}")
    else:
        plt.savefig(f"{output_path}.pdf", bbox_inches="tight", format="pdf")
        print(f"Saved summary panel to {output_path}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate 3x3 summary panel figure")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (.pdf or .png; default: summary_panel.pdf)",
    )
    parser.add_argument("--reference", type=str, default="1DDR")
    parser.add_argument("--position", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=128.39)
    args = parser.parse_args()

    data_dir = ROOT / "data"
    species_csv = data_dir / "pdb_aa_list_pos20_591.csv"
    aligned_fasta = data_dir / "dhfr_sequences_hmmer_aligned.fasta"
    dihedral_csv = data_dir / "dihedral_angles_pos20.csv"
    classification_csv = data_dir / "ligand_classification.csv"
    output_path = Path(args.output) if args.output else ROOT / "summary_panel.pdf"

    for path, label in [
        (species_csv, "species CSV"),
        (aligned_fasta, "aligned FASTA"),
        (dihedral_csv, "dihedral CSV"),
        (classification_csv, "ligand classification CSV"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found: {path}")
            raise SystemExit(1)

    plot_summary_panel(
        species_csv=species_csv,
        aligned_fasta=aligned_fasta,
        dihedral_csv=dihedral_csv,
        classification_csv=classification_csv,
        output_path=output_path,
        reference_id=args.reference,
        reference_pos=args.position,
        psi_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
