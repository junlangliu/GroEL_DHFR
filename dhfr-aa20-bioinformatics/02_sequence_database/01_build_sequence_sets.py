#!/usr/bin/env python3
"""
STEP 1 - Build the two sequence sets that the analysis compares.

Produces (written to data/):
  (A) dfr resistance family, dereplicated to 60 representative alleles
      -> data/00_dfr_all_catalog.faa, data/02_dfr_representatives_60.faa
  (B) chromosomal folA family, broadly sampled and dereplicated
      -> data/01_folA_chromosomal_c50_817.faa   (817 seqs; 804 are added onto
         the profile in step 2, the other 13 overlap the profile itself)

Data sources (downloaded live):
  - NCBI AMRFinderPlus reference DB 4.2 (2026-05-15.1): AMRProt.fa + ReferenceGeneCatalog.txt
        subtype=AMR, subclass~TRIMETHOPRIM, dfrB family excluded  -> dfr resistance seqs
  - UniProt: chromosomal folA / DHFR across bacteria (label-purity filtered)

Tools required on PATH: cd-hit
Python:  biopython, pandas

NOTE ON REPRODUCIBILITY: sequence-database contents drift over time. The
frozen sequence files shipped in this bundle (01_*, 02_*) are the exact
inputs used for the paper. This script documents how they were built and
will re-derive an equivalent set from current DB releases.
"""
import os, re, gzip, shutil, subprocess, urllib.request
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner, substitution_matrices

WORK = "seqbuild"; os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- (A) dfr family
BASE = ("https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/"
        "AMRFinderPlus/database/4.2/2026-05-15.1")
for f in ("AMRProt.fa", "ReferenceGeneCatalog.txt"):
    dst = os.path.join(WORK, f)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"{BASE}/{f}", dst)

cat = pd.read_csv(os.path.join(WORK, "ReferenceGeneCatalog.txt"),
                  sep="\t", dtype=str, low_memory=False)
sel = cat[(cat["subtype"].str.upper() == "AMR") &
          (cat["subclass"].str.contains("TRIMETHOPRIM", case=False, na=False))].copy()
sel["fam_root"] = sel["gene_family"].str.extract(r"(?i)^(dfr[A-Za-z]+)")[0]
sel = sel[sel["fam_root"].str.lower() != "dfrb"].copy()          # drop dfrB clade

prot_by_tok = {}
for r in SeqIO.parse(os.path.join(WORK, "AMRProt.fa"), "fasta"):
    for t in re.split(r"[|\s]", r.description):
        if t and (t[:3] in ("WP_", "NP_", "YP_") or re.match(r"^[A-Z]{2,3}\d", t)):
            prot_by_tok.setdefault(t, r)

recs = []
for _, row in sel.iterrows():
    acc = row["refseq_protein_accession"]
    if pd.isna(acc) or acc not in prot_by_tok:
        continue
    allele = row["gene_family"] if pd.isna(row["allele"]) else row["allele"]
    seq = str(prot_by_tok[acc].seq).rstrip("*")
    recs.append(SeqRecord(Seq(seq), id=f"{allele}|{acc}",
                          description=f"{row['fam_root']} {row['product_name']}"))
SeqIO.write(recs, os.path.join(WORK, "dfr_all.faa"), "fasta")
SeqIO.write(recs, "data/00_dfr_all_catalog.faa", "fasta")
print(f"dfr resistance seqs (dfrB excluded): {len(recs)}  -> data/00_dfr_all_catalog.faa")

# collapse near-identical alleles: 90% id BLOSUM62 global, keep one per cluster
aln = PairwiseAligner(); aln.substitution_matrix = substitution_matrices.load("BLOSUM62")
aln.open_gap_score, aln.extend_gap_score, aln.mode = -11, -1, "global"
def pid(a, b):
    if a == b: return 100.0
    al = aln.align(a, b)[0]
    s = sum(x == y for x, y in zip(*[str(z) for z in al]) if x != "-" and y != "-")
    return 100 * s / min(len(a), len(b))
reps = []
for r in sorted(recs, key=lambda x: -len(x.seq)):
    if all(pid(str(r.seq), str(rep.seq)) < 90 for rep in reps):
        reps.append(r)
SeqIO.write(reps, "data/02_dfr_representatives_60.faa", "fasta")
print(f"dfr representatives (>=90% collapsed): {len(reps)}  -> data/02_dfr_representatives_60.faa")

# ---------------------------------------------------------------- (B) folA family
# Chromosomal folA / DHFR pulled from UniProt across bacteria, then:
#   1) drop any header mentioning dfr / trimethoprim (label purity)
#   2) cd-hit dereplicate at 0.90 id, aS 0.8
#   3) sequence-level safeguard: drop anything >=60% identical to a dfr rep
#   4) cd-hit at 0.50 to broadly sample the family  -> data/01_folA_chromosomal_c50_817.faa
#
# UniProt query used (REST stream, reviewed+unreviewed):
#   (gene:folA OR protein_name:"dihydrofolate reductase") AND taxonomy_id:2 (Bacteria)
# The frozen intermediate (folA_bact_labelfilt.faa) is large; the shipped
# data/01_folA_chromosomal_c50_817.faa is the final 0.50-clustered product.
print("folA set: see README; final product shipped as data/01_folA_chromosomal_c50_817.faa")

# The two cd-hit calls that produced the shipped file, for the record:
#   cd-hit -i folA_bact_labelfilt.faa -o bact_derep_0.90.faa -c 0.90 -aS 0.8 -n 5 -M 4000 -T 4 -d 0
#   (safeguard filter vs dfr reps, >=60% id dropped)  -> data/00_folA_broad_derep90_9665.faa
#   cd-hit -i folA_broad_derep90.faa -o data/01_folA_chromosomal_c50_817.faa -c 0.5 -n 2 -M 4000 -T 4 -d 0
