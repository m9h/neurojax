"""Automatic bad-channel detection (OSL preprocessing parity).

OSL flags bad channels by amplitude/variance outliers and by low correlation to
the rest of the montage. This adds the same: flat channels, robust-z variance
outliers, and channels poorly correlated with their peers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.preprocessing.bad_channels import detect_bad_channels


def _cohort(seed=0, T=2000):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(T)
    good = np.stack([latent + 0.3 * rng.standard_normal(T) for _ in range(6)])  # 0..5
    flat = np.zeros((1, T))                                  # ch 6: dead/flat
    noisy = 6.0 * rng.standard_normal((1, T))               # ch 7: high-var, uncorrelated
    data = np.concatenate([good, flat, noisy], axis=0)
    return jnp.asarray(data)


def test_flat_and_noisy_channels_flagged():
    data = _cohort()
    res = detect_bad_channels(data)
    assert 6 in res["bad"]      # flat
    assert 7 in res["bad"]      # high-variance / uncorrelated


def test_good_channels_not_flagged():
    data = _cohort()
    res = detect_bad_channels(data)
    assert all(g not in res["bad"] for g in range(6))


def test_reasons_are_reported():
    data = _cohort()
    res = detect_bad_channels(data)
    # the flat channel shows up under the flat criterion; structure is per-reason
    assert 6 in res["flat"]
    assert isinstance(res["bad"], list)
