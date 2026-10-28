#!/usr/bin/env python
"""Overlap of NLTSA measures within and across anaesthesia / psychedelic datasets.

    python scripts/consciousness_bench/measure_overlap.py

For each (dataset, contrast): Spearman overlap on within-subject deltas, the
same after partialling out the spectral parameters, the pooled Pearson
(Maschke 2024 convention), clusters, effective dimensionality. Then pairwise
Mantel / RV comparison of the delta-overlap matrices across contrasts, and the
anaesthesia-vs-psychedelic difference matrix. Writes CSVs and a figure to
$CB_OUT/overlap/.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from neurojax.analysis.nltsa import MEASURES
from neurojax.bench.measure_overlap import (cluster_measures, compare_overlap, delta_table,
                                            effective_dimensionality, overlap_matrix)

CB = Path(os.environ.get("CB_OUT", "/data/datasets/consciousness-bench"))
OUT = CB / "overlap"
OUT.mkdir(exist_ok=True)
NLTSA = list(MEASURES)
CONTRASTS = [  # (label, dataset, a, b, class)
    ("propofol_mild", "propofol-chennu2016", "baseline", "mild", "anaesthesia"),
    ("propofol_moderate", "propofol-chennu2016", "baseline", "moderate", "anaesthesia"),
    ("propofol_recovery", "propofol-chennu2016", "baseline", "recovery", "anaesthesia"),
    ("dmt_vs_ec", "dmt-pallavicini2021", "EC", "DMT", "psychedelic"),
    ("dmt_vs_eo", "dmt-pallavicini2021", "EO", "DMT", "psychedelic"),
    ("eo_vs_ec", "dmt-pallavicini2021", "EC", "EO", "control"),
    ("psilocybin", "psilocybin-microdose-cavanna2022", "placebo", "psilocybin", "psychedelic"),
]


def load(ds):
    sc = pd.read_csv(CB / ds / "scores.csv")
    sp_path = CB / ds / "scores_spectral.csv"
    sp = pd.read_csv(sp_path) if sp_path.exists() else sc[~sc.measure.isin(NLTSA)]
    return sc[sc.measure.isin(NLTSA)], sp


mats, summary = {}, []
for label, ds, a, b, cls in CONTRASTS:
    if not (CB / ds / "scores.csv").exists():
        print(f"[skip] {label}: no scores")
        continue
    sc, sp = load(ds)
    if not {a, b} <= set(sc.condition):
        continue
    R = overlap_matrix(sc, a, b, unit="delta")
    Rp = overlap_matrix(sc, a, b, unit="delta", spectral_scores=sp)
    Rpool = overlap_matrix(sc, a, b, unit="pooled", method="pearson")
    for name, M in (("delta", R), ("delta_partial_spectral", Rp), ("pooled_pearson", Rpool)):
        M.to_csv(OUT / f"overlap_{label}_{name}.csv")
    mats[label] = dict(R=R, Rp=Rp, Rpool=Rpool, cls=cls)
    cl = cluster_measures(R, 0.5)
    ed = effective_dimensionality(delta_table(sc, a, b))
    edp = effective_dimensionality(delta_table(sc, a, b).pipe(lambda d: d))
    iu = np.triu_indices(len(R), 1)
    summary.append(dict(contrast=label, cls=cls, n=R.attrs["n"],
                        mean_abs_r_delta=float(np.nanmean(np.abs(R.to_numpy()[iu]))),
                        mean_abs_r_partial=float(np.nanmean(np.abs(Rp.to_numpy()[iu]))),
                        mean_abs_r_pooled=float(np.nanmean(np.abs(Rpool.to_numpy()[iu]))),
                        n_clusters=int(cl.max()), n_pc90=ed["n_components"], participation_ratio=ed["participation_ratio"],
                        first_pc_frac=ed["first_pc_frac"],
                        clusters="; ".join(",".join(cl.index[cl == k]) for k in sorted(cl.unique()))))
S = pd.DataFrame(summary)
S.to_csv(OUT / "summary.csv", index=False)
print(S.drop(columns="clusters").to_string(index=False, float_format="%.2f"))
for _, r in S.iterrows():
    print(f"\n{r.contrast} clusters (|r|>=0.5 average linkage):")
    for c in r.clusters.split("; "):
        print("   ", c)

# cross-contrast comparison of the delta overlap structure
labels = list(mats)
cmpm = pd.DataFrame(np.nan, index=labels, columns=labels)
rows = []
for i, l1 in enumerate(labels):
    for l2 in labels[i + 1:]:
        c = compare_overlap(mats[l1]["R"], mats[l2]["R"], n_perm=2000)
        cmpm.loc[l1, l2] = cmpm.loc[l2, l1] = c["mantel_r"]
        rows.append(dict(a=l1, b=l2, cls_a=mats[l1]["cls"], cls_b=mats[l2]["cls"], mantel_r=c["mantel_r"],
                         mantel_p=c["mantel_p"], rv=c["rv"]))
C = pd.DataFrame(rows)
C.to_csv(OUT / "cross_contrast_mantel.csv", index=False)
print("\nMantel r between delta-overlap matrices:")
print(C.to_string(index=False, float_format="%.3f"))
same = C[(C.cls_a == C.cls_b)].mantel_r
diff = C[(C.cls_a != C.cls_b) & (C.cls_a != "control") & (C.cls_b != "control")].mantel_r
print(f"\nwithin-class mean Mantel r = {same.mean():.3f}, anaesthesia-vs-psychedelic mean = {diff.mean():.3f}")

# anaesthesia vs psychedelic difference on the average matrices
def avg(cls):
    Ms = [m["R"] for m in mats.values() if m["cls"] == cls]
    cols = [c for c in Ms[0].columns if all(c in M.columns for M in Ms)]
    return sum(M.loc[cols, cols] for M in Ms) / len(Ms)


if any(m["cls"] == "anaesthesia" for m in mats.values()) and any(m["cls"] == "psychedelic" for m in mats.values()):
    An, Ps = avg("anaesthesia"), avg("psychedelic")
    c = compare_overlap(An, Ps, n_perm=5000)
    c["diff"].to_csv(OUT / "diff_psychedelic_minus_anaesthesia.csv")
    c["pairs"].to_csv(OUT / "pairs_psychedelic_minus_anaesthesia.csv", index=False)
    print(f"\nanaesthesia vs psychedelic mean matrices: Mantel r={c['mantel_r']:.3f} (p={c['mantel_p']:.3f}), RV={c['rv']:.3f}")
    print("pairs whose coupling changes most (r psychedelic - r anaesthesia):")
    print(c["pairs"].head(15).to_string(index=False, float_format="%.2f"))

    common = [c for c in An.columns if c in Ps.columns]
    order = [m for m in cluster_measures(An.loc[common, common], 0.5).attrs["order"]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, M, t in zip(axes, (An, Ps, c["diff"]), ("anaesthesia (propofol, mean of 3 contrasts)",
                                                      "psychedelic (DMT, psilocybin; mean of 3)", "psychedelic − anaesthesia")):
        im = ax.imshow(M.loc[order, order].to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        ax.set_title(t, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman r of within-subject change")
    fig.suptitle("Overlap of NLTSA measures: within-subject change correlations", fontsize=12)
    fig.savefig(OUT / "overlap_anaesthesia_vs_psychedelic.png", dpi=130, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(mats), figsize=(4 * len(mats), 4))
    for ax, (l, m) in zip(np.atleast_1d(axes), mats.items()):
        cl = cluster_measures(m["R"], 0.5)
        dendrogram(cl.attrs["linkage"], labels=list(m["R"].index), ax=ax, leaf_font_size=6, leaf_rotation=90,
                   color_threshold=0.5)
        ax.set_title(f"{l} ({m['cls']})", fontsize=9); ax.set_ylabel("1 − |r|")
    fig.tight_layout()
    fig.savefig(OUT / "dendrograms.png", dpi=130)
print(f"[done] -> {OUT}")
