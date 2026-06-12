#!/usr/bin/env python3
"""
Compare AA20 distributions between folA homologs and dfrA-family DHFRs.

Inputs (each MSA must contain the E. coli folA reference P0ABQ4 so we can map
the alignment column corresponding to reference position 20):
- folA MSA (representatives from mmseqs clustering with ref injected):
  data_processed/phylo/phylogenetics/mmseqs2_mafft/mmseqs_0.5/dhfr_mmseqs_thr0.5_mafft_with_p0abq4.fasta
- dfrA MSA (Pfam-trimmed dfrA family with reference appended):
  data_processed/dfr_family/dfr_hmmer_filtered_mafft.fasta

Outputs:
- aa20_folA_dfrA.csv  (counts + proportions)
- aa20_folA_dfrA.png  (stacked bars, folA vs dfrA)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
FOLA_MSA = (
    BASE_DIR
    / ".."
    / "phylo"
    / "phylogenetics"
    / "mmseqs2_mafft"
    / "folA_full"
    / "dhfr_pfam_full_with_ref_mafft.fasta"
)
# dfrA MSA re-aligned together with the folA reference (P0ABQ4)
DFRA_MSA = BASE_DIR / "dfrA_plus_folA_ref_mafft.fasta"
# Use E. coli folA (P0ABQ4) as the single anchor for both folA and dfrA mappings
DFRA_REF_FASTA = BASE_DIR / "folA_ref_P0ABQ4.fasta"
# Keep original outputs untouched; write new outputs with suffix _v2
OUT_CSV = BASE_DIR / "aa20_folA_dfrA_v2.csv"
OUT_PNG = BASE_DIR / "aa20_folA_dfrA_v2.png"

AA_STANDARD = sorted("ACDEFGHIKLMNPQRSTVWY")
FOLA_REF_PAT = re.compile(r"P0ABQ4|DYR_ECOLI", re.IGNORECASE)
DFRA_REF_PAT = FOLA_REF_PAT  # both sides anchored to folA reference
DFRA_PAT = re.compile(r"\bdfrA", re.IGNORECASE)


def load_reference_sequence(path: Path) -> str:
    """Load a reference FASTA (single record) and return the ungapped sequence."""
    seq_parts: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line)
    if not seq_parts:
        raise RuntimeError(f"No sequence found in reference FASTA: {path}")
    return "".join(seq_parts)


def read_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def find_ref(records: List[Tuple[str, str]], ref_pat, ref_seq: str | None = None) -> Tuple[str, str]:
    """Locate the reference row in an MSA.

    Priority:
    1) Exact sequence match to the provided ungapped reference sequence (if given).
    2) Header regex match (ref_pat).
    """
    if ref_seq is not None:
        for header, seq in records:
            if seq.replace("-", "") == ref_seq:
                return header, seq

    for header, seq in records:
        if ref_pat.search(header):
            return header, seq

    missing = "sequence" if ref_seq is not None else "pattern"
    raise RuntimeError(
        f"Reference not found in MSA by {missing}: {ref_pat.pattern if ref_pat else 'n/a'}"
    )


def ref_col_for_pos(seq: str, pos: int = 20) -> int:
    count = 0
    for idx, aa in enumerate(seq):
        if aa != "-":
            count += 1
        if count == pos:
            return idx
    raise RuntimeError(f"Reference shorter than {pos} non-gap residues.")


def distribution(
    msa_path: Path,
    keep_fn,
    ref_pat,
    exclude_ref: bool = True,
    ref_seq: str | None = None,
) -> Tuple[Dict[str, int], int, int]:
    records = list(read_fasta(msa_path))
    ref_header, ref_seq = find_ref(records, ref_pat, ref_seq)
    col_idx = ref_col_for_pos(ref_seq, pos=20)

    counts = {aa: 0 for aa in AA_STANDARD}
    total = 0
    used = 0
    for header, seq in records:
        if exclude_ref and header == ref_header:
            continue
        if not keep_fn(header):
            continue
        if col_idx >= len(seq):
            continue
        aa = seq[col_idx]
        if aa in counts:
            counts[aa] += 1
            total += 1
            used += 1
    return counts, total, used


def write_csv(
    fol_counts: Dict[str, int],
    fol_total: int,
    dfra_counts: Dict[str, int],
    dfra_total: int,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as handle:
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


def plot_stacked(
    fol_counts: Dict[str, int],
    fol_total: int,
    dfra_counts: Dict[str, int],
    dfra_total: int,
    out_path: Path,
) -> None:
    # Order AAs by overall abundance (folA + dfrA), descending; tie-break alphabetical
    ordered_aas = sorted(
        AA_STANDARD,
        key=lambda aa: -(fol_counts.get(aa, 0) + dfra_counts.get(aa, 0))
    )

    fol_fracs = [fol_counts.get(a, 0) / fol_total if fol_total else 0.0 for a in ordered_aas]
    dfra_fracs = [dfra_counts.get(a, 0) / dfra_total if dfra_total else 0.0 for a in ordered_aas]

    # Palette: ColorBrewer Paired (12) cycled to cover 20 AA; high contrast on light backgrounds
    base_palette = [
        "#1f78b4", "#a6cee3", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
        "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
    ]
    palette = (base_palette * ((len(ordered_aas) // len(base_palette)) + 1))[: len(ordered_aas)]
    color_by_aa = {aa: palette[i % len(palette)] for i, aa in enumerate(ordered_aas)}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    # Stacked bars with explicit legend for all AAs that appear in either group
    handles = []
    labels = []

    bottom = 0.0
    for aa, frac in zip(ordered_aas, fol_fracs):
        if frac == 0 and dfra_fracs[ordered_aas.index(aa)] == 0:
            continue
        h = ax.bar("folA", frac, bottom=bottom, color=color_by_aa[aa], edgecolor="#2f2f2f", linewidth=0.5)
        bottom += frac
        handles.append(h[0]); labels.append(aa)
        if frac >= 0.05:  # annotate major components
            ax.text("folA", bottom - frac / 2, f"{aa} {frac:.2f}", ha="center", va="center",
                    fontsize=8, color="#1f1f1f")

    bottom = 0.0
    for aa, frac in zip(ordered_aas, dfra_fracs):
        if frac == 0 and fol_fracs[ordered_aas.index(aa)] == 0:
            continue
        ax.bar("dfrA", frac, bottom=bottom, color=color_by_aa[aa], edgecolor="#2f2f2f", linewidth=0.5)
        bottom += frac
        if frac >= 0.05:
            ax.text("dfrA", bottom - frac / 2, f"{aa} {frac:.2f}", ha="center", va="center",
                    fontsize=8, color="#1f1f1f")

    ax.set_ylabel("Proportion at reference position 20")
    ax.set_title("AA20 distribution: folA homologs vs dfrA family")
    ax.legend(handles, labels, title="AA", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ref_seq = load_reference_sequence(DFRA_REF_FASTA)

    fol_counts, fol_total, fol_used = distribution(
        FOLA_MSA,
        keep_fn=lambda h: "dfrA" not in h.lower(),
        ref_pat=FOLA_REF_PAT,
        ref_seq=ref_seq,
    )
    dfra_counts, dfra_total, dfra_used = distribution(
        DFRA_MSA,
        keep_fn=lambda h: bool(DFRA_PAT.search(h)),
        ref_pat=DFRA_REF_PAT,
        ref_seq=ref_seq,
    )

    write_csv(fol_counts, fol_total, dfra_counts, dfra_total, OUT_CSV)
    plot_stacked(fol_counts, fol_total, dfra_counts, dfra_total, OUT_PNG)

    print(f"folA sequences used: {fol_used}, non-gap at pos20: {fol_total}")
    print(f"dfrA sequences used: {dfra_used}, non-gap at pos20: {dfra_total}")
    print(f"CSV: {OUT_CSV}")
    print(f"Plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
