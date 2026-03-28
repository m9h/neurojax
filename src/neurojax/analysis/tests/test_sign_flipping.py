"""Tests for MEG sign flipping — resolves beamformer/MNE sign ambiguity.

Sign flipping aligns the arbitrary ±1 sign per parcel across subjects
so that group-level HMM/DyNeMo analyses are consistent.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.sign_flipping import (
    apply_sign_flips,
    compute_template_covariance,
    find_sign_flips,
    sign_flip_metrics,
)


def _make_consistent_data(key, n_subjects=4, n_parcels=10, n_times=500):
    """Create synthetic data where all subjects have the same sign pattern."""
    keys = jr.split(key, n_subjects)
    # Shared spatial pattern — all subjects have the same covariance structure
    template_key, *subj_keys = jr.split(key, n_subjects + 1)
    # A shared mixing matrix to ensure consistent covariance structure
    A = jr.normal(template_key, (n_parcels, n_parcels)) * 0.5
    A = A + jnp.eye(n_parcels)
    data_list = []
    for k in subj_keys:
        noise = jr.normal(k, (n_parcels, n_times))
        data_list.append(A @ noise)
    return data_list


def _make_flipped_data(key, flips_per_subject, n_parcels=10, n_times=500):
    """Create consistent data then apply known sign flips per subject.

    Parameters
    ----------
    flips_per_subject : list of (n_parcels,) arrays with ±1 entries.
    """
    n_subjects = len(flips_per_subject)
    data_list = _make_consistent_data(
        key, n_subjects=n_subjects, n_parcels=n_parcels, n_times=n_times
    )
    flipped = []
    for data, signs in zip(data_list, flips_per_subject):
        flipped.append(data * signs[:, None])
    return flipped


class TestSignFlipIdentity:
    """Already-consistent data should require no (or trivial) flips."""

    def test_sign_flip_identity(self):
        data = _make_consistent_data(jr.PRNGKey(0), n_subjects=4, n_parcels=8)
        signs_list = find_sign_flips(data)

        assert len(signs_list) == 4
        for signs in signs_list:
            assert signs.shape == (8,)
            # Every entry should be ±1
            np.testing.assert_array_equal(jnp.abs(signs), 1.0)

        # For already-consistent data, the product of each subject's signs
        # with the first subject's signs should be all +1 (i.e., relative
        # alignment is identity).  We compare relative to subject 0.
        for signs in signs_list:
            relative = signs * signs_list[0]
            # All relative signs should agree
            np.testing.assert_array_equal(relative, relative[0] * jnp.ones(8))


class TestSignFlipKnownFlips:
    """Data with known sign flips should be correctly detected and corrected."""

    def test_sign_flip_known_flips(self):
        n_parcels = 10
        # Subject 0: no flips; Subject 1: parcels 2, 5, 7 flipped
        flips = [
            jnp.ones(n_parcels),
            jnp.ones(n_parcels).at[jnp.array([2, 5, 7])].set(-1.0),
        ]
        data = _make_flipped_data(jr.PRNGKey(42), flips, n_parcels=n_parcels)
        signs_list = find_sign_flips(data)

        # After applying recovered signs, the two subjects should be aligned
        corrected = [apply_sign_flips(d, s) for d, s in zip(data, signs_list)]

        # Covariances of corrected subjects should be close
        cov0 = corrected[0] @ corrected[0].T / corrected[0].shape[1]
        cov1 = corrected[1] @ corrected[1].T / corrected[1].shape[1]
        # Off-diagonal sign structure should match
        sign_cov0 = jnp.sign(cov0)
        sign_cov1 = jnp.sign(cov1)
        agreement = jnp.mean(sign_cov0 == sign_cov1)
        assert float(agreement) > 0.85


class TestSignFlipCorrelationImproves:
    """After flipping, inter-subject covariance correlation should increase."""

    def test_sign_flip_correlation_improves(self):
        n_parcels = 12
        # Introduce random flips to 3 subjects
        key = jr.PRNGKey(7)
        flips = []
        for i in range(3):
            k = jr.PRNGKey(i + 100)
            f = jnp.where(jr.bernoulli(k, 0.5, (n_parcels,)), 1.0, -1.0)
            flips.append(f)

        data = _make_flipped_data(key, flips, n_parcels=n_parcels, n_times=800)

        # Measure correlation before
        metrics_before = sign_flip_metrics(data, data)

        # Find and apply flips
        signs_list = find_sign_flips(data)
        corrected = [apply_sign_flips(d, s) for d, s in zip(data, signs_list)]

        metrics_after = sign_flip_metrics(data, corrected)

        assert metrics_after["mean_correlation_after"] >= metrics_before["mean_correlation_after"]


class TestSignFlipMultipleSubjects:
    """Sign flipping works with N > 2 subjects."""

    def test_sign_flip_multiple_subjects(self):
        n_subjects = 6
        n_parcels = 8
        key = jr.PRNGKey(99)
        flips = []
        for i in range(n_subjects):
            k = jr.PRNGKey(i + 200)
            f = jnp.where(jr.bernoulli(k, 0.5, (n_parcels,)), 1.0, -1.0)
            flips.append(f)

        data = _make_flipped_data(key, flips, n_parcels=n_parcels, n_times=600)
        signs_list = find_sign_flips(data)

        assert len(signs_list) == n_subjects
        for signs in signs_list:
            assert signs.shape == (n_parcels,)
            np.testing.assert_array_equal(jnp.abs(signs), 1.0)


class TestSignFlipParcellated:
    """Works on parcellated timeseries with shape (n_parcels, n_timepoints)."""

    def test_sign_flip_parcellated(self):
        n_parcels = 38  # typical DK atlas
        n_times = 1000
        key = jr.PRNGKey(11)
        data = _make_consistent_data(
            key, n_subjects=3, n_parcels=n_parcels, n_times=n_times
        )
        # Verify input shapes are (n_parcels, n_times)
        for d in data:
            assert d.shape == (n_parcels, n_times)

        signs_list = find_sign_flips(data)
        for signs in signs_list:
            assert signs.shape == (n_parcels,)


class TestSignFlipDeterministic:
    """Same input gives same output — no uncontrolled randomness."""

    def test_sign_flip_deterministic(self):
        n_parcels = 10
        flips = [
            jnp.ones(n_parcels),
            jnp.ones(n_parcels).at[jnp.array([1, 4, 9])].set(-1.0),
            jnp.ones(n_parcels).at[jnp.array([3, 6])].set(-1.0),
        ]
        data = _make_flipped_data(jr.PRNGKey(55), flips, n_parcels=n_parcels)

        signs_a = find_sign_flips(data)
        signs_b = find_sign_flips(data)

        for sa, sb in zip(signs_a, signs_b):
            np.testing.assert_array_equal(sa, sb)


class TestSignFlipMetrics:
    """sign_flip_metrics returns useful diagnostic info."""

    def test_sign_flip_metrics(self):
        n_parcels = 10
        flips = [
            jnp.ones(n_parcels),
            jnp.ones(n_parcels).at[jnp.array([0, 3, 7])].set(-1.0),
        ]
        data = _make_flipped_data(jr.PRNGKey(77), flips, n_parcels=n_parcels)
        signs_list = find_sign_flips(data)
        corrected = [apply_sign_flips(d, s) for d, s in zip(data, signs_list)]

        metrics = sign_flip_metrics(data, corrected)

        assert "n_flips" in metrics
        assert "mean_correlation_before" in metrics
        assert "mean_correlation_after" in metrics
        assert isinstance(metrics["n_flips"], (int, np.integer, jnp.integer))
        assert metrics["mean_correlation_after"] >= metrics["mean_correlation_before"]


class TestSignFlipPreservesMagnitude:
    """Only signs change, not magnitudes."""

    def test_sign_flip_preserves_magnitude(self):
        n_parcels = 6
        data = _make_consistent_data(jr.PRNGKey(33), n_subjects=2, n_parcels=n_parcels)
        signs_list = find_sign_flips(data)

        for d, signs in zip(data, signs_list):
            flipped = apply_sign_flips(d, signs)
            np.testing.assert_allclose(
                jnp.abs(flipped), jnp.abs(d), atol=1e-6,
                err_msg="Magnitudes should be preserved after sign flipping",
            )


class TestApplySignFlips:
    """Unit tests for the apply_sign_flips helper."""

    def test_identity_signs(self):
        d = jr.normal(jr.PRNGKey(0), (5, 100))
        result = apply_sign_flips(d, jnp.ones(5))
        np.testing.assert_array_equal(result, d)

    def test_negate_all(self):
        d = jr.normal(jr.PRNGKey(0), (5, 100))
        result = apply_sign_flips(d, -jnp.ones(5))
        np.testing.assert_allclose(result, -d, atol=1e-7)

    def test_selective_flip(self):
        d = jnp.ones((3, 10))
        signs = jnp.array([1.0, -1.0, 1.0])
        result = apply_sign_flips(d, signs)
        expected = jnp.array([[1.0]*10, [-1.0]*10, [1.0]*10])
        np.testing.assert_array_equal(result, expected)


class TestComputeTemplateCovariance:
    """Unit tests for compute_template_covariance."""

    def test_shape(self):
        data = _make_consistent_data(jr.PRNGKey(0), n_subjects=3, n_parcels=5)
        cov = compute_template_covariance(data)
        assert cov.shape == (5, 5)

    def test_symmetric(self):
        data = _make_consistent_data(jr.PRNGKey(0), n_subjects=3, n_parcels=5)
        cov = compute_template_covariance(data)
        np.testing.assert_allclose(cov, cov.T, atol=1e-6)
