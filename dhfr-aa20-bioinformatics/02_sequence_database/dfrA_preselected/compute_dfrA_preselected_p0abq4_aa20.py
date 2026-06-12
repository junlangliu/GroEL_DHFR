#!/usr/bin/env python3
"""Count P0ABQ4 AA20 residues in the preselected dfrA set.

dfrA membership is fixed before MAFFT in `dfrA_plus_folA_ref.fasta`.
This script counts every non-P0ABQ4 record in the corresponding MSA and does
not re-filter dfrA membership from potentially truncated MAFFT headers.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
IC50_BASE = AUDIT_ROOT.parents[1]
DFR_DIR = IC50_BASE / "data_processed" / "dfr_family"

DFRA_PRESELECTED_FASTA = DFR_DIR / "dfrA_plus_folA_ref.fasta"
DFRA_MSA = DFR_DIR / "dfrA_plus_folA_ref_mafft.fasta"
P0ABQ4_FASTA = DFR_DIR / "folA_ref_P0ABQ4.fasta"

OUT_COUNTS = HERE / "dfrA_preselected_p0abq4_aa20_counts.csv"
OUT_SUMMARY = HERE / "dfrA_preselected_p0abq4_aa20_summary.tsv"
OUT_HEADER_LOSS = HERE / "dfrA_post_msa_regex_loss.tsv"

AA_STANDARD = list("ACDEFGHIKLMNPQRSTVWY")
P0ABQ4_PAT = re.compile(r"P0ABQ4|DYR_ECOLI", re.IGNORECASE)
DFRA_PAT = re.compile(r"\bdfrA", re.IGNORECASE)


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
    raise RuntimeError("P0ABQ4 reference not found in dfrA MSA")


def ref_col_for_pos(aligned_ref: str, pos: int = 20) -> int:
    seen = 0
    for idx, aa in enumerate(aligned_ref):
        if aa != "-":
            seen += 1
        if seen == pos:
            return idx
    raise RuntimeError(f"P0ABQ4 reference has fewer than {pos} non-gap residues")


def main() -> None:
    records = read_fasta(DFRA_MSA)
    ref_seq = load_reference(P0ABQ4_FASTA)
    ref_header, ref_aln = find_reference(records, ref_seq)
    col_idx = ref_col_for_pos(ref_aln, pos=20)

    preselected_records = read_fasta(DFRA_PRESELECTED_FASTA)
    preselected_non_p0abq4 = [
        header
        for header, seq in preselected_records
        if seq.replace("-", "") != ref_seq and not P0ABQ4_PAT.search(header)
    ]
    preselected_dfra_headers = [h for h in preselected_non_p0abq4 if DFRA_PAT.search(h)]

    counts = Counter({aa: 0 for aa in AA_STANDARD})
    stats = Counter()
    examples = {"gap": [], "nonstandard": []}
    post_msa_regex_losses: list[str] = []

    for header, seq in records:
        stats["total_records"] += 1
        if header == ref_header:
            stats["p0abq4_reference"] += 1
            continue
        stats["included_non_p0abq4"] += 1
        if not DFRA_PAT.search(header):
            post_msa_regex_losses.append(header)
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
        writer.writerow(["aa", "dfrA_count", "dfrA_prop"])
        for aa in AA_STANDARD:
            count = counts[aa]
            writer.writerow([aa, count, count / total if total else 0.0])

    with OUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerow(["preselected_input_fasta", DFRA_PRESELECTED_FASTA])
        writer.writerow(["input_msa", DFRA_MSA])
        writer.writerow(["p0abq4_reference_header", ref_header])
        writer.writerow(["p0abq4_aa20_alignment_column_1based", col_idx + 1])
        writer.writerow(["preselected_non_p0abq4_records", len(preselected_non_p0abq4)])
        writer.writerow(["preselected_headers_matching_dfra_regex", len(preselected_dfra_headers)])
        for key in [
            "total_records",
            "p0abq4_reference",
            "included_non_p0abq4",
            "standard_aa20",
            "gap_aa20",
            "nonstandard_aa20",
        ]:
            writer.writerow([key, stats[key]])
        writer.writerow(["post_msa_records_not_matching_dfra_regex", len(post_msa_regex_losses)])
        for key, values in examples.items():
            writer.writerow([f"example_{key}", " || ".join(values)])

    with OUT_HEADER_LOSS.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["reason", "msa_header"])
        for header in post_msa_regex_losses:
            writer.writerow(["non_P0ABQ4_record_not_matching_post_msa_dfra_regex", header])

    print(f"dfrA standard AA20: {total}")
    print(f"Wrote {OUT_COUNTS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_HEADER_LOSS}")


if __name__ == "__main__":
    main()
