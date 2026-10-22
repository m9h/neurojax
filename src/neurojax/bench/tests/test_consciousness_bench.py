"""Tests for the consciousness complexity benchmark harness on synthetic
two-condition recordings (narrow-alpha 'awake' vs broadband 'drug')."""

import jax
import numpy as np
import pytest

from neurojax.analysis.nltsa import MEASURES
from neurojax.bench.consciousness_bench import (
    Recording,
    contrast,
    decompose_contrast,
    epoch_recording,
    score_recordings,
)

FS = 100.0


def _ar2(seed, n, f0, r, n_ch):
    rng = np.random.RandomState(seed)
    a1, a2 = 2 * r * np.cos(2 * np.pi * f0 / FS), -(r ** 2)
    x = rng.randn(n_ch, n + 200)
    for t in range(2, x.shape[1]):
        x[:, t] += a1 * x[:, t - 1] + a2 * x[:, t - 2]
    return x[:, 200:].astype(np.float32)


SUBSET = {k: MEASURES[k] for k in ("lz76_mean", "permutation_entropy", "hjorth_mobility", "pair_correlation")}


@pytest.fixture(scope="module")
def recordings():
    recs = []
    for s in range(6):
        recs.append(Recording("synth", f"s{s}", "awake", _ar2(10 + s, 3000, 10, 0.97, 3), FS))
        recs.append(Recording("synth", f"s{s}", "drug", _ar2(20 + s, 3000, 10, 0.6, 3), FS))
    return recs


def test_epoching_drops_partial_epoch():
    rec = Recording("synth", "s0", "awake", np.zeros((3, 1050), np.float32), FS)
    ep = epoch_recording(rec, epoch_s=2.0)
    assert ep.shape == (5, 3, 200)
    assert epoch_recording(rec, epoch_s=2.0, max_epochs=2).shape[0] == 2


def test_score_table_shape_and_columns(recordings):
    df = score_recordings(recordings, SUBSET, epoch_s=5.0, max_epochs=3)
    assert set(df.columns) >= {"dataset", "subject", "condition", "measure", "value", "sd", "n_epochs"}
    assert len(df) == len(recordings) * len(SUBSET)
    assert (df["n_epochs"] == 3).all()
    assert np.isfinite(df["value"]).all()


def test_contrast_detects_broadband_increase(recordings):
    df = score_recordings(recordings, SUBSET, epoch_s=5.0, max_epochs=3)
    c = contrast(df, "awake", "drug").set_index("measure")
    assert set(c.columns) >= {"n", "cohen_d", "p_wilcoxon", "auc", "frac_increase"}
    assert c.loc["lz76_mean", "cohen_d"] > 1.0
    assert c.loc["lz76_mean", "auc"] > 0.9
    assert c.loc["lz76_mean", "n"] == 6


def test_decomposition_is_spectral_for_linear_shift(recordings):
    d = decompose_contrast(recordings[:2], SUBSET, "awake", "drug", key=jax.random.PRNGKey(0),
                           epoch_s=5.0, max_epochs=3, n_surrogates=10)
    row = d.set_index("measure").loc["lz76_mean"]
    assert row["spectral_fraction"] > 0.7
    assert abs(row["phasic"]) < 3 * row["phasic_std"]
