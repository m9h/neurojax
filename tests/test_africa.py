"""AFRICA-style automatic artefact-IC classification + rejection.

OSL's AFRICA runs ICA, identifies artefact components (ocular/cardiac/line),
and back-projects the rest. neurojax already has FastICA; this is the missing
classify-and-reject layer: correlate components with EOG/ECG references, score
line-noise power, flag, and reconstruct without the flagged components. (For EEG,
mne_icalabel's ICLabel is an alternative classifier; the heuristic core here is
montage-free and what AFRICA uses for MEG.)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.preprocessing.africa import (
    classify_artifact_components,
    correlation_scores,
    line_noise_scores,
    reject_components,
)


def _planted(seed=0, T=1000, fs=256.0):  # fs > 2*50 so the line tone is resolved
    rng = np.random.default_rng(seed)
    eog = rng.standard_normal(T)                       # reference signal
    t = np.arange(T) / fs
    line = np.sin(2 * np.pi * 50.0 * t)                # 50 Hz line-noise source
    sources = np.stack([eog.copy(), line, rng.standard_normal(T)])  # comp 0=ocular, 1=line
    mixing = rng.standard_normal((4, 3))               # 4 sensors, 3 comps
    return jnp.asarray(sources), jnp.asarray(mixing), jnp.asarray(eog), fs


def test_correlation_scores_flag_the_reference_component():
    sources, _, eog, _ = _planted()
    s = correlation_scores(sources, eog)
    assert s.shape == (3,)
    assert float(s[0]) > 0.99          # component 0 IS the reference
    assert float(s[1]) < 0.3 and float(s[2]) < 0.3


def test_line_noise_scores_flag_the_50hz_component():
    sources, _, _, fs = _planted()
    s = line_noise_scores(sources, fs, line_freq=50.0, bw=1.0)
    assert s.shape == (3,)
    assert float(s[1]) > 0.8           # component 1 is nearly all 50 Hz
    assert float(s[0]) < 0.3 and float(s[2]) < 0.3


def test_classify_unions_ocular_and_line():
    sources, _, eog, fs = _planted()
    res = classify_artifact_components(
        sources, fs, eog=eog, corr_thresh=0.5, line_freq=50.0, line_thresh=0.3
    )
    assert 0 in res["artifacts"]       # ocular
    assert 1 in res["artifacts"]       # line
    assert 2 not in res["artifacts"]   # clean


def test_reject_components_backprojects_without_flagged():
    sources, mixing, _, _ = _planted()
    clean = reject_components(sources, mixing, [0, 1])
    expected = mixing @ sources.at[jnp.array([0, 1])].set(0.0)
    assert clean.shape == (4, sources.shape[1])
    assert jnp.allclose(clean, expected, atol=1e-6)
