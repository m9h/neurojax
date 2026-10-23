# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Overlap between complexity / NLTSA measures, and how it differs between
anaesthesia and psychedelic states.

Maschke et al. 2024 (Fig. 6b) report a Pearson matrix over pooled
participant × condition values. Pooling mixes two variance sources: stable
between-subject differences and the within-subject drug shift, and when the
shift is large every responsive measure correlates with every other
regardless of what it measures. Here the overlap is assessed on

* ``unit="delta"``: within-subject changes ``cond_b − cond_a`` (one row per
  subject) — which measures carry the *same state information*;
* ``unit="baseline"``: ``cond_a`` values — which measures covary across people;
* ``unit="pooled"``: the Maschke convention, for comparison;

with Spearman by default, optional partial correlation that removes a set of
spectral parameters (specparam exponent/offset/alpha power, band powers), so
the remaining overlap is shared *non-spectral* information. Measures are then
grouped by average-linkage clustering on ``1 − |r|``, and two datasets'
matrices are compared with a Mantel correlation of their upper triangles,
the RV coefficient, and a permutation p-value, plus the element-wise
difference to find pairs whose coupling changes between states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

__all__ = ["wide_table", "delta_table", "overlap_matrix", "partial_out", "cluster_measures",
           "compare_overlap", "effective_dimensionality"]


def wide_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Long scores (dataset, subject, condition, measure, value) -> rows
    (subject, condition) × columns measure."""
    return scores.pivot_table(index=["subject", "condition"], columns="measure", values="value")


def delta_table(scores: pd.DataFrame, cond_a: str, cond_b: str) -> pd.DataFrame:
    """Subjects × measures of ``cond_b − cond_a`` (subjects with both)."""
    w = wide_table(scores)
    a, b = w.xs(cond_a, level="condition"), w.xs(cond_b, level="condition")
    common = a.index.intersection(b.index)
    return (b.loc[common] - a.loc[common])


def partial_out(X: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """Residualise every column of ``X`` on the columns of ``Z`` (with
    intercept), aligned on index; rows with any NaN in Z are dropped."""
    Z = Z.loc[X.index].dropna()
    X = X.loc[Z.index]
    A = np.column_stack([np.ones(len(Z)), (Z - Z.mean()).to_numpy()])
    out = X.copy()
    for c in X.columns:
        y = X[c].to_numpy(float)
        ok = np.isfinite(y)
        beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        out[c] = np.where(ok, y - A @ beta, np.nan)
    return out


def overlap_matrix(scores: pd.DataFrame, cond_a: str, cond_b: str, unit: str = "delta",
                   method: str = "spearman", measures: list[str] | None = None,
                   spectral_scores: pd.DataFrame | None = None,
                   partial: tuple[str, ...] = ("specparam_exponent", "specparam_offset", "specparam_alpha_power",
                                               "relpow_delta", "relpow_theta", "relpow_beta")) -> pd.DataFrame:
    """Measure × measure correlation matrix (``attrs['p']`` holds p-values,
    ``attrs['n']`` the rows used). With ``spectral_scores`` the same unit of
    the spectral parameters is partialled out of every measure first."""
    if unit == "delta":
        X = delta_table(scores, cond_a, cond_b)
        Z = delta_table(spectral_scores, cond_a, cond_b) if spectral_scores is not None else None
    elif unit == "baseline":
        X = wide_table(scores).xs(cond_a, level="condition")
        Z = wide_table(spectral_scores).xs(cond_a, level="condition") if spectral_scores is not None else None
    elif unit == "pooled":
        w = wide_table(scores)
        X = w[w.index.get_level_values("condition").isin([cond_a, cond_b])]
        Z = wide_table(spectral_scores).loc[X.index] if spectral_scores is not None else None
    else:
        raise ValueError("unit must be delta, baseline or pooled")
    if measures is not None:
        X = X[[m for m in measures if m in X.columns]]
    if Z is not None:
        cols = [c for c in partial if c in Z.columns]
        X = partial_out(X, Z[cols])
    X = X.dropna(axis=1, how="all")
    if method == "spearman":
        r, p = spearmanr(X.to_numpy(float), nan_policy="omit")
        if np.ndim(r) == 0:  # scipy returns scalars for exactly two columns
            r, p = np.array([[1.0, r], [r, 1.0]]), np.array([[0.0, p], [p, 0.0]])
    elif method == "pearson":
        r = X.corr(method="pearson").to_numpy()
        n = len(X)
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(np.abs(t), n - 2)
    else:
        raise ValueError(method)
    R = pd.DataFrame(r, index=X.columns, columns=X.columns)
    R.attrs["p"] = pd.DataFrame(p, index=X.columns, columns=X.columns)
    R.attrs["n"] = len(X)
    R.attrs["unit"] = unit
    return R


def cluster_measures(R: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Average-linkage clusters on distance ``1 − |r|``; measures within a
    cluster have |r| ≳ 1 − threshold with each other on average. Returns
    cluster id per measure, ordered by dendrogram leaf order (``attrs['order']``)."""
    D = 1 - np.abs(R.to_numpy())
    np.fill_diagonal(D, 0)
    D = np.nan_to_num(D, nan=1.0)
    L = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(L, t=threshold, criterion="distance")
    from scipy.cluster.hierarchy import leaves_list
    s = pd.Series(labels, index=R.index, name="cluster")
    s.attrs["order"] = list(R.index[leaves_list(L)])
    s.attrs["linkage"] = L
    return s


def _upper(R: pd.DataFrame, cols) -> np.ndarray:
    M = R.loc[cols, cols].to_numpy()
    iu = np.triu_indices(len(cols), 1)
    return M[iu]


def compare_overlap(R1: pd.DataFrame, R2: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """Similarity of two measure-overlap structures on their common measures:
    Mantel ``r`` (Spearman between upper triangles) with a label-permutation
    p-value, the RV coefficient (multivariate R² between the two matrices),
    and ``diff`` = R2 − R1 with the pairs that change most."""
    cols = [c for c in R1.columns if c in R2.columns]
    u1, u2 = _upper(R1, cols), _upper(R2, cols)
    ok = np.isfinite(u1) & np.isfinite(u2)
    r_obs = spearmanr(u1[ok], u2[ok]).statistic
    rng = np.random.RandomState(seed)
    M2 = R2.loc[cols, cols].to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(cols))
        Mp = M2[np.ix_(perm, perm)]
        up = Mp[np.triu_indices(len(cols), 1)]
        null[i] = spearmanr(u1[ok], up[ok]).statistic
    p = float((np.sum(null >= r_obs) + 1) / (n_perm + 1))
    A, B = np.nan_to_num(R1.loc[cols, cols].to_numpy()), np.nan_to_num(M2)
    rv = float(np.trace(A @ B) / np.sqrt(np.trace(A @ A) * np.trace(B @ B)))
    diff = R2.loc[cols, cols] - R1.loc[cols, cols]
    iu = np.triu_indices(len(cols), 1)
    pairs = pd.DataFrame(dict(a=np.array(cols)[iu[0]], b=np.array(cols)[iu[1]],
                              r1=u1, r2=u2, diff=u2 - u1)).dropna()
    pairs = pairs.reindex(pairs["diff"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return dict(mantel_r=float(r_obs), mantel_p=p, rv=rv, n_measures=len(cols), diff=diff, pairs=pairs)


def effective_dimensionality(X: pd.DataFrame, var_frac: float = 0.9) -> dict:
    """PCA of z-scored columns: components for ``var_frac`` of the variance and
    the participation ratio (Σλ)² / Σλ²; both say how many independent
    quantities the measure set actually reports."""
    Z = X.dropna(axis=1, how="any")
    Z = (Z - Z.mean()) / Z.std(ddof=1)
    Z = Z.dropna(axis=1)
    lam = np.linalg.eigvalsh(np.cov(Z.to_numpy().T))[::-1]
    lam = np.clip(lam, 0, None)
    cum = np.cumsum(lam) / lam.sum()
    return dict(n_components=int(np.searchsorted(cum, var_frac) + 1), participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()),
                n_measures=Z.shape[1], n_rows=Z.shape[0], first_pc_frac=float(lam[0] / lam.sum()))
