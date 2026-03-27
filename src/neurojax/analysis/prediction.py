"""Multi-modal TMS response prediction framework.

Extracts features from structural connectivity, brain dynamics (HMM/DyNeMo),
and metabolic data, then predicts TMS outcomes (TEP amplitude, complexity,
local/network transition time) using cross-validated regression.

Designed for the WAND dataset: 170 subjects × multiple modalities.

Pipeline::

    Per subject:
      Connectome → graph features (strength, clustering, density)
      HMM states → dynamics features (occupancy, lifetime, switching)
      MRS → metabolic features (GABA, glutamate)
      ─── merge ───→ feature vector (p features)

    Across subjects:
      Feature matrix (N, p) + target (N,)
      → Cross-validated ridge regression
      → R², feature importance ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Feature containers
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """Named feature matrix for one modality.

    Parameters
    ----------
    features : (n_subjects, n_features) array.
    names : list of str — one name per feature column.
    modality : str — label for the modality (e.g., "structural", "functional").
    """

    features: jnp.ndarray
    names: list[str]
    modality: str = ""

    @property
    def n_subjects(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


@dataclass
class PredictionResult:
    """Result of a prediction model.

    Attributes
    ----------
    y_pred : predicted values.
    r_squared : coefficient of determination.
    mse : mean squared error.
    coefficients : model coefficients (for linear models).
    feature_names : names of features used.
    """

    y_pred: jnp.ndarray
    r_squared: float
    mse: float
    coefficients: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    feature_names: list[str] = field(default_factory=list)


def merge_feature_sets(feature_sets: list[FeatureSet]) -> FeatureSet:
    """Concatenate multiple FeatureSets horizontally.

    Parameters
    ----------
    feature_sets : list of FeatureSet, all with same n_subjects.

    Returns
    -------
    Merged FeatureSet with concatenated features and names.
    """
    features = jnp.concatenate([fs.features for fs in feature_sets], axis=1)
    names = []
    for fs in feature_sets:
        names.extend(fs.names)
    modalities = "+".join(fs.modality for fs in feature_sets if fs.modality)
    return FeatureSet(features=features, names=names, modality=modalities)


# ---------------------------------------------------------------------------
# Feature extraction: connectome
# ---------------------------------------------------------------------------


def extract_connectome_features(
    weights: jnp.ndarray,
    prefix: str = "",
) -> FeatureSet:
    """Extract graph-theoretic features from a structural connectome.

    Produces a single-subject feature vector. Call per subject and stack.

    Parameters
    ----------
    weights : (N, N) connectivity matrix.
    prefix : str — prefix for feature names (e.g., "sub01_").

    Returns
    -------
    FeatureSet with n_subjects=1 and features derived from the graph.
    """
    W = np.array(weights, copy=True)
    np.fill_diagonal(W, 0)
    N = W.shape[0]

    # Node strengths
    strengths = np.sum(W, axis=1)

    features = []
    names = []

    # Global metrics
    features.append(np.mean(strengths))
    names.append(f"{prefix}mean_strength")

    features.append(np.std(strengths))
    names.append(f"{prefix}strength_std")

    # Density
    n_possible = N * (N - 1)
    density = np.sum(W > 0) / n_possible
    features.append(density)
    names.append(f"{prefix}density")

    # Clustering coefficient (simplified: using weighted)
    from neurojax.analysis.funcnet import clustering_coefficient
    cc = np.asarray(clustering_coefficient(jnp.array(W > np.median(W[W > 0])).astype(float)))
    features.append(np.mean(cc))
    names.append(f"{prefix}clustering_coeff")

    features.append(np.std(cc))
    names.append(f"{prefix}clustering_std")

    # Asymmetry
    asym = np.mean(np.abs(W - W.T))
    features.append(asym)
    names.append(f"{prefix}asymmetry")

    # Max strength (hub strength)
    features.append(np.max(strengths))
    names.append(f"{prefix}max_strength")

    # Strength entropy
    s_norm = strengths / (strengths.sum() + 1e-10)
    s_norm = s_norm[s_norm > 0]
    entropy = -np.sum(s_norm * np.log(s_norm))
    features.append(entropy)
    names.append(f"{prefix}strength_entropy")

    return FeatureSet(
        features=jnp.array(features)[None, :],  # (1, n_features)
        names=names,
        modality="structural",
    )


# ---------------------------------------------------------------------------
# Feature extraction: dynamics (HMM/DyNeMo summary stats)
# ---------------------------------------------------------------------------


def extract_dynamics_features(
    fractional_occupancy: jnp.ndarray,
    mean_lifetime: jnp.ndarray,
    switching_rate: jnp.ndarray,
    prefix: str = "",
) -> FeatureSet:
    """Extract dynamics features from HMM/DyNeMo summary statistics.

    Parameters
    ----------
    fractional_occupancy : (n_subjects, K) — per-state occupancy.
    mean_lifetime : (n_subjects, K) — per-state mean lifetime.
    switching_rate : (n_subjects,) — transitions per second.

    Returns
    -------
    FeatureSet with dynamics features.
    """
    n_subjects = fractional_occupancy.shape[0]
    K = fractional_occupancy.shape[1]

    features = []
    names = []

    # Fractional occupancy per state
    for k in range(K):
        features.append(fractional_occupancy[:, k])
        names.append(f"{prefix}fo_state{k}")

    # Mean lifetime per state
    for k in range(K):
        features.append(mean_lifetime[:, k])
        names.append(f"{prefix}lifetime_state{k}")

    # Switching rate
    features.append(switching_rate)
    names.append(f"{prefix}switching_rate")

    X = jnp.stack(features, axis=1)  # (n_subjects, n_features)
    return FeatureSet(features=X, names=names, modality="dynamics")


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


def ridge_predict(
    X: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1.0,
) -> PredictionResult:
    """Fit ridge regression and return predictions + metrics.

    Parameters
    ----------
    X : (N, p) feature matrix.
    y : (N,) target vector.
    alpha : float — regularization strength.

    Returns
    -------
    PredictionResult with in-sample predictions.
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    N, p = X_np.shape

    # Standardize
    X_mean = X_np.mean(axis=0)
    X_std = X_np.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_s = (X_np - X_mean) / X_std
    y_mean = y_np.mean()
    y_s = y_np - y_mean

    # Ridge: beta = (X^T X + alpha I)^{-1} X^T y
    XtX = X_s.T @ X_s
    beta = np.linalg.solve(XtX + alpha * np.eye(p), X_s.T @ y_s)

    y_pred = X_s @ beta + y_mean
    residuals = y_np - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_np - y_mean) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    mse = float(np.mean(residuals ** 2))

    return PredictionResult(
        y_pred=jnp.array(y_pred),
        r_squared=float(r2),
        mse=mse,
        coefficients=jnp.array(beta),
    )


# ---------------------------------------------------------------------------
# Cross-validated prediction
# ---------------------------------------------------------------------------


def cross_validated_predict(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> PredictionResult:
    """K-fold cross-validated ridge regression.

    Parameters
    ----------
    X : (N, p) features.
    y : (N,) targets.
    n_folds : int — number of folds. N for LOO.
    alpha : float — ridge regularization.

    Returns
    -------
    PredictionResult with out-of-fold predictions and cross-validated R².
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    N, p = X_np.shape

    # Create fold indices
    indices = np.arange(N)
    fold_size = N // n_folds
    y_pred = np.zeros(N)

    for fold in range(n_folds):
        if fold == n_folds - 1:
            test_idx = indices[fold * fold_size:]
        else:
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_test = X_np[test_idx]

        # Standardize using training stats
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train_s = (X_train - X_mean) / X_std
        X_test_s = (X_test - X_mean) / X_std
        y_mean = y_train.mean()
        y_train_s = y_train - y_mean

        # Ridge
        XtX = X_train_s.T @ X_train_s
        beta = np.linalg.solve(XtX + alpha * np.eye(p), X_train_s.T @ y_train_s)

        y_pred[test_idx] = X_test_s @ beta + y_mean

    ss_res = np.sum((y_np - y_pred) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    mse = float(np.mean((y_np - y_pred) ** 2))

    # Fit full model for coefficients
    full_result = ridge_predict(X, y, alpha)

    return PredictionResult(
        y_pred=jnp.array(y_pred),
        r_squared=float(r2),
        mse=mse,
        coefficients=full_result.coefficients,
    )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance(
    X: jnp.ndarray,
    y: jnp.ndarray,
    names: list[str],
    alpha: float = 1.0,
    n_folds: int = 5,
) -> list[dict]:
    """Rank features by importance using permutation-based drop in R².

    For each feature, shuffle it and measure how much R² drops.
    Larger drop = more important feature.

    Parameters
    ----------
    X : (N, p) features.
    y : (N,) targets.
    names : list of str — feature names.
    alpha : float — ridge regularization.
    n_folds : int — CV folds for baseline R².

    Returns
    -------
    List of dicts sorted by importance (descending), each with:
        ``"name"``, ``"importance"``, ``"baseline_r2"``, ``"permuted_r2"``
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    rng = np.random.default_rng(0)

    # Baseline R²
    baseline = cross_validated_predict(X, y, n_folds, alpha)
    baseline_r2 = baseline.r_squared

    results = []
    for j, name in enumerate(names):
        # Shuffle feature j
        X_perm = X_np.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_result = cross_validated_predict(jnp.array(X_perm), y, n_folds, alpha)
        drop = baseline_r2 - perm_result.r_squared
        results.append({
            "name": name,
            "importance": max(drop, 0.0),  # clamp negative drops
            "baseline_r2": baseline_r2,
            "permuted_r2": perm_result.r_squared,
        })

    # Sort by importance descending
    results.sort(key=lambda d: d["importance"], reverse=True)
    return results
