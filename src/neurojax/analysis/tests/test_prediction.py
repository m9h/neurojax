"""Tests for multi-modal TMS response prediction — TDD RED phase.

Predicts TMS outcomes (TEP amplitude, complexity, transition time)
from pre-TMS features (structural, functional, metabolic).
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.prediction import (
    FeatureSet,
    PredictionResult,
    cross_validated_predict,
    extract_connectome_features,
    extract_dynamics_features,
    feature_importance,
    merge_feature_sets,
    ridge_predict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_connectome():
    """10-region symmetric connectome."""
    rng = np.random.default_rng(42)
    W = rng.random((10, 10))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    return jnp.array(W)


@pytest.fixture
def synthetic_dynamics_features():
    """Fake HMM summary stats for 20 subjects."""
    rng = np.random.default_rng(0)
    return {
        "fractional_occupancy": jnp.array(rng.dirichlet(np.ones(4), size=20)),
        "mean_lifetime": jnp.array(rng.exponential(50, size=(20, 4))),
        "switching_rate": jnp.array(rng.uniform(1, 10, size=20)),
    }


@pytest.fixture
def synthetic_targets():
    """20 subjects, 3 target variables."""
    rng = np.random.default_rng(1)
    return jnp.array(rng.normal(size=(20, 3)))


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------

class TestFeatureSet:
    def test_construction(self):
        X = jnp.ones((10, 5))
        names = ["f1", "f2", "f3", "f4", "f5"]
        fs = FeatureSet(features=X, names=names, modality="structural")
        assert fs.features.shape == (10, 5)
        assert len(fs.names) == 5

    def test_n_subjects(self):
        fs = FeatureSet(features=jnp.ones((20, 3)), names=["a", "b", "c"])
        assert fs.n_subjects == 20

    def test_n_features(self):
        fs = FeatureSet(features=jnp.ones((20, 7)), names=[f"f{i}" for i in range(7)])
        assert fs.n_features == 7


class TestMergeFeatureSets:
    def test_concatenates_features(self):
        fs1 = FeatureSet(jnp.ones((10, 3)), ["a", "b", "c"], "mod1")
        fs2 = FeatureSet(jnp.ones((10, 2)), ["d", "e"], "mod2")
        merged = merge_feature_sets([fs1, fs2])
        assert merged.features.shape == (10, 5)
        assert len(merged.names) == 5

    def test_preserves_names(self):
        fs1 = FeatureSet(jnp.ones((5, 2)), ["a", "b"], "x")
        fs2 = FeatureSet(jnp.ones((5, 1)), ["c"], "y")
        merged = merge_feature_sets([fs1, fs2])
        assert merged.names == ["a", "b", "c"]

    def test_single_set(self):
        fs = FeatureSet(jnp.ones((5, 3)), ["a", "b", "c"])
        merged = merge_feature_sets([fs])
        assert merged.features.shape == (5, 3)


# ---------------------------------------------------------------------------
# Feature extraction: connectome
# ---------------------------------------------------------------------------

class TestExtractConnectomeFeatures:
    def test_returns_feature_set(self, synthetic_connectome):
        fs = extract_connectome_features(synthetic_connectome)
        assert isinstance(fs, FeatureSet)

    def test_includes_standard_metrics(self, synthetic_connectome):
        fs = extract_connectome_features(synthetic_connectome)
        names = set(fs.names)
        assert "mean_strength" in names
        assert "clustering_coeff" in names
        assert "density" in names

    def test_correct_n_features(self, synthetic_connectome):
        fs = extract_connectome_features(synthetic_connectome)
        assert fs.n_features > 3  # at least strength, clustering, density


# ---------------------------------------------------------------------------
# Feature extraction: dynamics
# ---------------------------------------------------------------------------

class TestExtractDynamicsFeatures:
    def test_returns_feature_set(self, synthetic_dynamics_features):
        fs = extract_dynamics_features(
            fractional_occupancy=synthetic_dynamics_features["fractional_occupancy"],
            mean_lifetime=synthetic_dynamics_features["mean_lifetime"],
            switching_rate=synthetic_dynamics_features["switching_rate"],
        )
        assert isinstance(fs, FeatureSet)
        assert fs.n_subjects == 20

    def test_feature_count(self, synthetic_dynamics_features):
        fs = extract_dynamics_features(
            fractional_occupancy=synthetic_dynamics_features["fractional_occupancy"],
            mean_lifetime=synthetic_dynamics_features["mean_lifetime"],
            switching_rate=synthetic_dynamics_features["switching_rate"],
        )
        # 4 FO + 4 lifetime + 1 switching = 9
        assert fs.n_features == 9


# ---------------------------------------------------------------------------
# Ridge regression prediction
# ---------------------------------------------------------------------------

class TestRidgePredict:
    def test_output_shape(self, synthetic_targets):
        X = jr.normal(jr.PRNGKey(0), (20, 5))
        y = synthetic_targets[:, 0]
        result = ridge_predict(X, y, alpha=1.0)
        assert isinstance(result, PredictionResult)
        assert result.y_pred.shape == y.shape

    def test_perfect_fit_low_error(self):
        """Perfectly predictable target → low error."""
        X = jnp.eye(10)
        y = jnp.arange(10, dtype=float)
        result = ridge_predict(X, y, alpha=0.001)
        assert result.r_squared > 0.9

    def test_coefficients_shape(self):
        X = jr.normal(jr.PRNGKey(0), (20, 5))
        y = jr.normal(jr.PRNGKey(1), (20,))
        result = ridge_predict(X, y, alpha=1.0)
        assert result.coefficients.shape == (5,)


# ---------------------------------------------------------------------------
# Cross-validated prediction
# ---------------------------------------------------------------------------

class TestCrossValidatedPredict:
    def test_returns_result(self, synthetic_targets):
        X = jr.normal(jr.PRNGKey(0), (20, 5))
        y = synthetic_targets[:, 0]
        result = cross_validated_predict(X, y, n_folds=5)
        assert isinstance(result, PredictionResult)

    def test_r_squared_bounded(self, synthetic_targets):
        X = jr.normal(jr.PRNGKey(0), (20, 5))
        y = synthetic_targets[:, 0]
        result = cross_validated_predict(X, y, n_folds=5)
        assert result.r_squared <= 1.0

    def test_predictable_target(self):
        """Feature that perfectly predicts target → high R²."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.normal(size=50) * 0.1
        result = cross_validated_predict(jnp.array(X), jnp.array(y), n_folds=5)
        assert result.r_squared > 0.5

    def test_loo_cv(self, synthetic_targets):
        """Leave-one-out cross-validation."""
        X = jr.normal(jr.PRNGKey(0), (15, 3))
        y = synthetic_targets[:15, 0]
        result = cross_validated_predict(X, y, n_folds=15)  # LOO
        assert result.y_pred.shape == (15,)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_returns_per_feature(self):
        X = jr.normal(jr.PRNGKey(0), (30, 4))
        y = X[:, 0] * 3.0 + jr.normal(jr.PRNGKey(1), (30,)) * 0.1
        imp = feature_importance(X, y, names=["f0", "f1", "f2", "f3"])
        assert len(imp) == 4

    def test_informative_feature_ranks_high(self):
        """Feature that drives the target should have high importance."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        y = X[:, 2] * 5.0 + rng.normal(size=50) * 0.1
        imp = feature_importance(
            jnp.array(X), jnp.array(y),
            names=["f0", "f1", "f2_important", "f3"]
        )
        # f2 should have highest importance
        best = max(imp, key=lambda x: x["importance"])
        assert best["name"] == "f2_important"

    def test_returns_sorted(self):
        X = jr.normal(jr.PRNGKey(0), (30, 3))
        y = jr.normal(jr.PRNGKey(1), (30,))
        imp = feature_importance(X, y, names=["a", "b", "c"])
        importances = [d["importance"] for d in imp]
        assert importances == sorted(importances, reverse=True)
