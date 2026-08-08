#!/usr/bin/env python3
"""
STEP 3 - Fisher exact test + BH-FDR, then both figures.

Compares per-residue amino-acid composition between:
  - chromosomal folA family   (n = 804, from folA_broad_c50 projected on profile)
  - dfr resistance family     (n = 60,  representative alleles)
both aligned to E. coli DHFR (UniProt P0ABQ4) numbering via a folA profile MSA.

Inputs  (produced by 02_align_to_ecoli.py, in data/):
    04_folA804_aligned_to_profile.faa   (profile 134 rows + 804 added folA)
    04_dfr60_aligned_to_profile.faa     (profile 134 rows + 60 added dfr)
    03_folA_profile_msa_134x296.faa     (profile; supplies the P0ABQ4 anchor)

Outputs:
    data/05_fisher_results_perresidue.csv    (every residue x position tested)
    figures/folA_vs_dfr_volcano.png / .svg
    figures/folA_vs_dfr_volcano_all_enriched.png / .svg
    figures/recap_pairs_perposition.png / .svg

Run:  python 03_fisher_and_figures.py
"""
import os
import numpy as np, pandas as pd
from Bio import SeqIO
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

PROF   = "data/03_folA_profile_msa_134x296.faa"
FOLA   = "data/04_folA804_aligned_to_profile.faa"
DFR    = "data/04_dfr60_aligned_to_profile.faa"
AA     = list("ACDEFGHIKLMNPQRSTVWY")
os.makedirs("figures", exist_ok=True)

# ---- anchor: E. coli DHFR P0ABQ4 defines residue<->column map, positions 1..159 ----
prof = list(SeqIO.parse(PROF, "fasta"))
prof_ids = {r.id for r in prof}
anchor = str(next(r for r in prof if "P0ABQ4" in r.id).seq)
col_of, rp = {}, 0
for c, ch in enumerate(anchor):
    if ch != "-":
        rp += 1; col_of[rp] = c
positions = list(range(1, 160))
cols = [col_of[p] for p in positions]
assert col_of[20] == 77 and anchor[77] == "M", "Met20 must map to profile column 77"

def added(path):
    """rows added by mafft --add, i.e. everything not in the profile."""
    return [r for r in SeqIO.parse(path, "fasta") if r.id not in prof_ids]

def matrix(recs):
    return np.array([[str(r.seq)[c] for c in cols] for r in recs])

A_folA = matrix(added(FOLA)); nF = len(A_folA)
A_dfr  = matrix(added(DFR));  nD = len(A_dfr)
print(f"folA n={nF}, dfr n={nD}")

# ---- Fisher exact per (position, residue), 2x2 = [[dfr+, dfr-],[folA+, folA-]] ----
rows, pv, keyidx = [], [], []
for pi, p in enumerate(positions):
    cF, cD = A_folA[:, pi], A_dfr[:, pi]
    for aa in AA:
        fF = int((cF == aa).sum()); fD = int((cD == aa).sum())
        if fF == 0 and fD == 0:
            continue
        odd, pval = fisher_exact([[fD, nD - fD], [fF, nF - fF]])
        rows.append(dict(position=p, aa=aa, ecoli_wt=anchor[col_of[p]],
                         n_dfr=fD, frac_dfr=fD / nD, n_folA=fF, frac_folA=fF / nF,
                         delta_pp=100 * (fD / nD - fF / nF), odds_ratio=odd, p_value=pval))
        pv.append(pval); keyidx.append(len(rows) - 1)

rej, q, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
for k, qq, rj in zip(keyidx, q, rej):
    rows[k]["q_value_BH"] = qq; rows[k]["sig_FDR05"] = bool(rj)

res = pd.DataFrame(rows)
res["neglog10p"] = -np.log10(res["p_value"].clip(lower=1e-300))
res = res.sort_values(["p_value", "position"]).reset_index(drop=True)
res.to_csv("data/05_fisher_results_perresidue.csv", index=False)
print("wrote data/05_fisher_results_perresidue.csv  rows =", len(res))
r20 = res[(res.position == 20) & (res.aa == "I")].iloc[0]
print(f"  M20I check: delta={r20.delta_pp:+.1f} pp  p={r20.p_value:.2e}  q={r20.q_value_BH:.2e}")

# ================= FIGURE 1: volcano =================
plt.rcParams.update({"font.family": "Arial", "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "lines.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(3.5, 2.5))
res["neglog10q"] = -np.log10(res["q_value_BH"].clip(lower=1e-300))
thr = -np.log10(0.05)
ns = res[res.neglog10q <  thr]                          # not significant (q > 0.05)
up = res[(res.neglog10q >= thr) & (res.delta_pp > 0)]   # significant enrichment
dn = res[(res.neglog10q >= thr) & (res.delta_pp < 0)]   # significant depletion
ax.scatter(ns.delta_pp, ns.neglog10q, c=ns.delta_pp.abs(), cmap="Greys",
           vmin=0, vmax=50, s=6, edgecolor="#999", linewidth=0.5, alpha=0.6, zorder=2)
ax.scatter(up.delta_pp, up.neglog10q, c=up.delta_pp.abs(), cmap="Reds",
           vmin=0, vmax=50, s=8, edgecolor="#b0453a", linewidth=0.5, alpha=0.9, zorder=3)
ax.scatter(dn.delta_pp, dn.neglog10q, c=dn.delta_pp.abs(), cmap="Blues",
           vmin=0, vmax=50, s=8, edgecolor="#3a5aa0", linewidth=0.5, alpha=0.9, zorder=3)
ax.axhline(thr, ls=":", c="#bbb", lw=0.5, zorder=1)
ax.text(59, thr + 0.4, "q = 0.05", fontsize=7, color="#999", ha="right")
ax.text(-44, 21.5, "depleted", fontsize=7, color="#3a5aa0", ha="left", va="top")
ax.text(2, 21.5, "enriched", fontsize=7, color="#d62728", ha="left", va="top")
# among the top 1% most-enriched / most-depleted, label those overlapping our experimental hits
exp_hits = {5, 20, 21, 24, 26, 27, 28, 30, 94, 153}
n_lab = max(1, round(0.01 * len(res)))
top = pd.concat([res.nlargest(n_lab, "delta_pp"), res.nsmallest(n_lab, "delta_pp")])
lab = top[top.position.isin(exp_hits)]
# also label any significantly-depleted experimental site (most-depleted residue per position)
dn_hits = (dn[dn.position.isin(exp_hits)].sort_values("delta_pp")
             .groupby("position", as_index=False).first())
lab = pd.concat([lab, dn_hits]).drop_duplicates(subset=["position", "aa"])
from adjustText import adjust_text
texts, pts = [], []
for _, row in lab.iterrows():
    t = ax.text(row.delta_pp, row.neglog10q, f"{row.aa}{row.position}", fontsize=7,
                color="#222", ha="center", va="center", zorder=6)
    texts.append(t); pts.append((row.delta_pp, row.neglog10q))
adjust_text(texts, ax=ax, expand=(1.4, 1.6))
nudge = {"F153": (-9, 0), "L24": (-2, -4.3), "E26": (0, -1.2)}  # manual tweaks (data units)
for t in texts:
    dx, dy = nudge.get(t.get_text(), (0, 0))
    if dx or dy:
        x, y = t.get_position(); t.set_position((x + dx, y + dy))
# save final label + point positions to reuse the identical left (depleted) side in FIGURE 1B
fig1_label = {t.get_text(): (t.get_position(), (px, py))
              for t, (px, py) in zip(texts, pts)}
for t, (px, py) in zip(texts, pts):
    tx, ty = t.get_position()
    ax.annotate("", xy=(px, py), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5), zorder=5)
ax.axvline(0, c="#ccc", lw=0.5, zorder=1)
ax.set_xlabel(r"Δ residue frequency, $\it{dfr}$ − $\it{folA}$ (percentage)")
ax.set_ylabel(r"$\mathregular{-log_{10}}$ q (BH-FDR)")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
sm = ScalarMappable(norm=Normalize(0, 50), cmap="Greys"); sm.set_array([])
cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
cb = fig.colorbar(sm, cax=cax)
cb.outline.set_linewidth(0.5); cax.tick_params(width=0.5)
cb.set_label(r"|Δ residue frequency|, $\it{dfr}$ − $\it{folA}$ (percentage)")
ax.set_xlim(-45, 62)
fig.tight_layout()
fig.savefig("figures/folA_vs_dfr_volcano.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/folA_vs_dfr_volcano.svg", bbox_inches="tight")
print("wrote figures/folA_vs_dfr_volcano.png")
print("wrote figures/folA_vs_dfr_volcano.svg")

# ===== FIGURE 1B: volcano, label ALL enriched experimental-hit positions =====
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.scatter(ns.delta_pp, ns.neglog10q, c=ns.delta_pp.abs(), cmap="Greys",
           vmin=0, vmax=50, s=6, edgecolor="#999", linewidth=0.5, alpha=0.6, zorder=2)
ax.scatter(up.delta_pp, up.neglog10q, c=up.delta_pp.abs(), cmap="Reds",
           vmin=0, vmax=50, s=8, edgecolor="#b0453a", linewidth=0.5, alpha=0.9, zorder=3)
ax.scatter(dn.delta_pp, dn.neglog10q, c=dn.delta_pp.abs(), cmap="Blues",
           vmin=0, vmax=50, s=8, edgecolor="#3a5aa0", linewidth=0.5, alpha=0.9, zorder=3)
ax.axhline(thr, ls=":", c="#bbb", lw=0.5, zorder=1)
ax.text(59, thr + 0.4, "q = 0.05", fontsize=7, color="#999", ha="right")
ax.text(-44, 21.5, "depleted", fontsize=7, color="#3a5aa0", ha="left", va="top")
ax.text(2, 21.5, "enriched", fontsize=7, color="#d62728", ha="left", va="top")
# LEFT (depleted): reuse the exact label positions from FIGURE 1 -> identical left side
for name, (labxy, pxy) in fig1_label.items():
    if pxy[0] >= 0:            # keep only depleted (left, Δ<0) labels
        continue
    ax.text(labxy[0], labxy[1], name, fontsize=7, color="#222",
            ha="center", va="center", zorder=6)
    ax.annotate("", xy=pxy, xytext=labxy,
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5), zorder=5)
# RIGHT (enriched): label the most-enriched significant residue at every experimental position
up_hits = (up[up.position.isin(exp_hits)].sort_values("delta_pp", ascending=False)
             .groupby("position", as_index=False).first())
texts, pts = [], []
for _, row in up_hits.iterrows():
    t = ax.text(row.delta_pp, row.neglog10q, f"{row.aa}{row.position}", fontsize=7,
                color="#222", ha="center", va="center", zorder=6)
    texts.append(t); pts.append((row.delta_pp, row.neglog10q))
adjust_text(texts, ax=ax, expand=(1.6, 1.8))
for t, (px, py) in zip(texts, pts):
    tx, ty = t.get_position()
    ax.annotate("", xy=(px, py), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5), zorder=5)
ax.axvline(0, c="#ccc", lw=0.5, zorder=1)
ax.set_xlabel(r"Δ residue frequency, $\it{dfr}$ − $\it{folA}$ (percentage)")
ax.set_ylabel(r"$\mathregular{-log_{10}}$ q (BH-FDR)")
sm = ScalarMappable(norm=Normalize(0, 50), cmap="Greys"); sm.set_array([])
cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
cb = fig.colorbar(sm, cax=cax)
cb.outline.set_linewidth(0.5); cax.tick_params(width=0.5)
cb.set_label(r"|Δ residue frequency|, $\it{dfr}$ − $\it{folA}$ (percentage)")
ax.set_xlim(-45, 62)
fig.tight_layout()
fig.savefig("figures/folA_vs_dfr_volcano_all_enriched.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/folA_vs_dfr_volcano_all_enriched.svg", bbox_inches="tight")
print("wrote figures/folA_vs_dfr_volcano_all_enriched.png")
print("wrote figures/folA_vs_dfr_volcano_all_enriched.svg")

# ================= FIGURE 2: dominant swap per position =================
dep = res[(res.delta_pp < 0) & (res.q_value_BH < 0.05)]
enr = res[(res.delta_pp > 0) & (res.q_value_BH < 0.05)]
rep_pairs = []
for p in positions:
    d = dep[dep.position == p]; e = enr[enr.position == p]
    if len(d) == 0 or len(e) == 0:
        continue
    dr = d.loc[d.delta_pp.idxmin()]; er = e.loc[e.delta_pp.idxmax()]
    rep_pairs.append(dict(position=p, lost=dr.aa, lost_delta=dr.delta_pp,
                          gained=er.aa, gained_delta=er.delta_pp, pair=f"{dr.aa}{p}{er.aa}"))
R = pd.DataFrame(rep_pairs)
print(f"positions with a significant loss AND gain (q<0.05): {len(R)}")

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.scatter(R.lost_delta, R.gained_delta, c="#666",
           s=8, edgecolor="#444", linewidth=0.5, alpha=0.85, zorder=2)
lim = 50
xx = np.linspace(-lim, 0, 2)
ax.plot(xx, -xx, ls=":", c="#bbb", lw=0.5, zorder=1)
ax.text(-11, 11, "1:1 swap", fontsize=5, color="#999", ha="center", va="bottom",
        rotation=-45, rotation_mode="anchor")
ax.axhline(0, c="#ccc", lw=0.5, zorder=1); ax.axvline(0, c="#ccc", lw=0.5, zorder=1)
# label experimental-hit positions
labR = R[R.position.isin(exp_hits)]
red_pairs = {"M20I", "D27E"}       # highlighted in red
leader_pairs = {"I94S", "L24A"}    # keep leader lines only for these
texts, pts = [], []
for _, row in labR.iterrows():
    is_red = row.pair in red_pairs
    col = "#d62728" if is_red else "#666"
    fs = 7 if is_red else 6
    t = ax.text(row.lost_delta, row.gained_delta, row.pair, fontsize=fs,
                color=col, ha="center", va="center", zorder=6)
    texts.append(t); pts.append((row.lost_delta, row.gained_delta, row.pair))
adjust_text(texts, ax=ax, expand=(1.4, 1.6))
nudgeR = {"D27E": (1.2, 0), "L28Q": (0, -0.5), "E26G": (0, -0.5),
          "L24A": (2.5, -2), "I94S": (0, -2)}  # manual tweaks (data units)
for t in texts:
    dx, dy = nudgeR.get(t.get_text(), (0, 0))
    if dx or dy:
        x, y = t.get_position(); t.set_position((x + dx, y + dy))
for t, (px, py, pr) in zip(texts, pts):
    if pr not in leader_pairs:
        continue
    tx, ty = t.get_position()
    ax.annotate("", xy=(px, py), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5), zorder=5)
ax.set_xlim(-lim, 0); ax.set_ylim(0, lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Δ residue frequency, most-depleted (percentage)")
ax.set_ylabel("Δ residue frequency, most-enriched (percentage)")
fig.tight_layout()
fig.savefig("figures/recap_pairs_perposition.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/recap_pairs_perposition.svg", bbox_inches="tight")
print("wrote figures/recap_pairs_perposition.png")
print("wrote figures/recap_pairs_perposition.svg")
