#!/usr/bin/env python3
"""Count P0ABQ4 AA20 residues in the full folA/DHFR background set."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
IC50_BASE = AUDIT_ROOT.parents[1]

FOLA_MSA = (
    IC50_BASE
    / "data_processed"
    / "phylo"
    / "phylogenetics"
    / "mmseqs2_mafft"
    / "folA_full"
    / "dhfr_pfam_full_with_ref_mafft.fasta"
)
P0ABQ4_FASTA = IC50_BASE / "data_processed" / "dfr_family" / "folA_ref_P0ABQ4.fasta"

OUT_COUNTS = HERE / "folA_background_full_p0abq4_aa20_counts.csv"
OUT_SUMMARY = HERE / "folA_background_full_p0abq4_aa20_summary.tsv"

AA_STANDARD = list("ACDEFGHIKLMNPQRSTVWY")
P0ABQ4_PAT = re.compile(r"P0ABQ4|DYR_ECOLI", re.IGNORECASE)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    parts: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line.strip())
        if header is not None:
            records.append((header, "".join(parts)))
    return records


def load_reference(path: Path) -> str:
    return "".join(
        line.strip()
        for line in path.open()
        if line.strip() and not line.startswith(">")
    )


def find_reference(records: list[tuple[str, str]], ref_seq: str) -> tuple[str, str]:
    for header, seq in records:
        if seq.replace("-", "") == ref_seq:
            return header, seq
    for header, seq in records:
        if P0ABQ4_PAT.search(header):
            return header, seq
    raise RuntimeError("P0ABQ4 reference not found in folA background MSA")


def ref_col_for_pos(aligned_ref: str, pos: int = 20) -> int:
    seen = 0
    for idx, aa in enumerate(aligned_ref):
        if aa != "-":
            seen += 1
        if seen == pos:
            return idx
    raise RuntimeError(f"P0ABQ4 reference has fewer than {pos} non-gap residues")


def main() -> None:
    records = read_fasta(FOLA_MSA)
    ref_seq = load_reference(P0ABQ4_FASTA)
    ref_header, ref_aln = find_reference(records, ref_seq)
    col_idx = ref_col_for_pos(ref_aln, pos=20)

    counts = Counter({aa: 0 for aa in AA_STANDARD})
    stats = Counter()
    examples = {"gap": [], "nonstandard": []}

    for header, seq in records:
        stats["total_records"] += 1
        if header == ref_header:
            stats["p0abq4_reference"] += 1
            continue
        if "dfra" in header.lower():
            stats["excluded_header_contains_dfra"] += 1
            continue
        stats["included_non_reference"] += 1
        aa = seq[col_idx] if col_idx < len(seq) else ""
        if aa in counts:
            counts[aa] += 1
            stats["standard_aa20"] += 1
        elif aa == "-":
            stats["gap_aa20"] += 1
            if len(examples["gap"]) < 3:
                examples["gap"].append(header)
        else:
            stats["nonstandard_aa20"] += 1
            if len(examples["nonstandard"]) < 3:
                examples["nonstandard"].append(f"{header}\t{aa}")

    total = stats["standard_aa20"]
    with OUT_COUNTS.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["aa", "folA_count", "folA_prop"])
        for aa in AA_STANDARD:
            count = counts[aa]
            writer.writerow([aa, count, count / total if total else 0.0])

    with OUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerow(["input_msa", FOLA_MSA])
        writer.writerow(["p0abq4_reference_header", ref_header])
        writer.writerow(["p0abq4_aa20_alignment_column_1based", col_idx + 1])
        for key in [
            "total_records",
            "p0abq4_reference",
            "included_non_reference",
            "excluded_header_contains_dfra",
            "standard_aa20",
            "gap_aa20",
            "nonstandard_aa20",
        ]:
            writer.writerow([key, stats[key]])
        for key, values in examples.items():
            writer.writerow([f"example_{key}", " || ".join(values)])

    print(f"folA standard AA20: {total}")
    print(f"Wrote {OUT_COUNTS}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
