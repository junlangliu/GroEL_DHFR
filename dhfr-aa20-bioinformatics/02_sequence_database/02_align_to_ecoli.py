#!/usr/bin/env python3
"""
STEP 2 - Align both families to E. coli DHFR (P0ABQ4) numbering.

Strategy: project every sequence onto a fixed folA profile MSA with
MAFFT --keeplength --add, so all sequences share one coordinate system
anchored to the E. coli DHFR reference (P0ABQ4). Met20 = profile column 77.

Inputs (data/):
    03_folA_profile_msa_134x296.faa       (134-row folA profile; contains P0ABQ4)
    01_folA_chromosomal_c50_817.faa       (chromosomal folA, from step 1)
    02_dfr_representatives_60.faa         (dfr representatives, from step 1)

Outputs (data/):
    04_folA804_aligned_to_profile.faa     (profile 134 + 804 added folA = 951 rows)
    04_dfr60_aligned_to_profile.faa       (profile 134 + 60 added dfr  = 194 rows)

The 804 count = 817 input folA minus the 13 that are already present in the
profile (added rows exclude anything whose id is in the profile).

Tools required on PATH: mafft (>=7.5)
"""
import subprocess
from Bio import SeqIO

PROF = "data/03_folA_profile_msa_134x296.faa"
jobs = [("data/01_folA_chromosomal_c50_817.faa", "data/04_folA804_aligned_to_profile.faa"),
        ("data/02_dfr_representatives_60.faa",   "data/04_dfr60_aligned_to_profile.faa")]

prof_ids = {r.id for r in SeqIO.parse(PROF, "fasta")}
anchor = next(r for r in SeqIO.parse(PROF, "fasta") if "P0ABQ4" in r.id)
print(f"profile: {len(prof_ids)} rows, width {len(anchor.seq)}; anchor {anchor.id}")

for src, out in jobs:
    subprocess.run(f"mafft --thread 8 --keeplength --add {src} {PROF} > {out}",
                   shell=True, check=True)
    added = [r for r in SeqIO.parse(out, "fasta") if r.id not in prof_ids]
    print(f"{out}: {len(added)} sequences added onto profile")
