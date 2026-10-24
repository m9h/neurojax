"""Tests for measure-overlap analysis on synthetic score tables."""

import numpy as np
import pandas as pd
import pytest

from neurojax.bench.measure_overlap import (
    cluster_measures,
    compare_overlap,
    delta_table,
    effective_dimensionality,
    overlap_matrix,
)


def _scores(seed, n=30, flip=False):
    """Three measures: m1, m2 share a latent state change; m3 independent; s1
    is a 'spectral' parameter driving m1 and m2. Between-subject baseline
    noise is large so pooled correlations differ from delta correlations."""
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n):
        base = rng.randn() * 3
        state = rng.randn()
        spec = state + 0.3 * rng.randn()
        m1 = spec + 0.2 * rng.randn()
        m2 = (-1 if flip else 1) * (spec + 0.2 * rng.randn())
        m3 = rng.randn()
        for cond, k in (("a", 0.0), ("b", 1.0)):
            rows += [dict(dataset="d", subject=f"s{s}", condition=cond, measure="m1", value=k * m1 + base),
                     dict(dataset="d", subject=f"s{s}", condition=cond, measure="m2", value=k * m2 + base),
                     dict(dataset="d", subject=f"s{s}", condition=cond, measure="m3", value=k * m3 + base)]
    spec_rows = [dict(dataset="d", subject=r["subject"], condition=r["condition"], measure="specparam_exponent",
                      value=(r["value"] if r["measure"] == "m1" else np.nan)) for r in rows]
    return pd.DataFrame(rows), pd.DataFrame(spec_rows).dropna()


def test_delta_vs_pooled_units():
    df, sp = _scores(0)
    d = delta_table(df, "a", "b")
    assert d.shape == (30, 3)
    R_delta = overlap_matrix(df, "a", "b", unit="delta")
    R_pooled = overlap_matrix(df, "a", "b", unit="pooled")
    assert R_delta.loc["m1", "m2"] > 0.8
    assert abs(R_delta.loc["m1", "m3"]) < 0.5
    # pooled: shared baseline + condition shift inflate m1-m3
    assert R_pooled.loc["m1", "m3"] > R_delta.loc["m1", "m3"]
    assert R_delta.attrs["n"] == 30 and "p" in R_delta.attrs


def test_partial_out_spectral_removes_shared_variance():
    df, sp = _scores(1)
    R = overlap_matrix(df, "a", "b", unit="delta")
    Rp = overlap_matrix(df, "a", "b", unit="delta", spectral_scores=sp, partial=("specparam_exponent",))
    assert Rp.loc["m1", "m2"] < R.loc["m1", "m2"] - 0.3


def test_clusters_and_comparison():
    df1, _ = _scores(2)
    df2, _ = _scores(3, flip=True)
    R1 = overlap_matrix(df1, "a", "b")
    R2 = overlap_matrix(df2, "a", "b")
    cl = cluster_measures(R1, threshold=0.5)
    assert cl["m1"] == cl["m2"] != cl["m3"]
    assert set(cl.attrs["order"]) == {"m1", "m2", "m3"}
    cmp = compare_overlap(R1, R2, n_perm=200)
    assert set(cmp) >= {"mantel_r", "mantel_p", "rv", "diff", "pairs"}
    top = cmp["pairs"].iloc[0]
    assert {top["a"], top["b"]} == {"m1", "m2"} and top["diff"] < -1.0  # the flipped pair
    same = compare_overlap(R1, overlap_matrix(_scores(4)[0], "a", "b"), n_perm=200)
    assert same["mantel_r"] > cmp["mantel_r"]


def test_effective_dimensionality():
    df, _ = _scores(5)
    ed = effective_dimensionality(delta_table(df, "a", "b"))
    assert ed["n_measures"] == 3 and 1 <= ed["n_components"] <= 3
    assert 1.0 <= ed["participation_ratio"] <= 3.0
