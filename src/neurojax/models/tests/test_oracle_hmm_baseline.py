"""Validity tests for the osl-dynamics HMM oracle baseline (red-green TDD).

These are deliberately NOT guarded by ``skipif``: building the oracle is the
task, so absence of the artifacts is a RED failure, not a skip.  They assert the
baseline is internally consistent AND that osl-dynamics actually recovered the
generating **state time course** — i.e. the baseline is trustworthy as a parity
target for the JAX reimplementation (see ``test_hmm_oracle_parity.py``).

We validate state *inference*, not the learned mean *parameters*: osl-dynamics
learns means by gradient descent and shrinks them toward zero in this regime,
so its gamma/occupancy recover the truth even though its mean parameters do not.

Produce the artifacts with the oracle (CPU/TF osl-dynamics)::

    DATA_DIR=tests/data/oracle_osl/hmm \\
      python containers/scripts/generate_synthetic.py
    DATA_DIR=tests/data/oracle_osl/hmm \\
      python containers/scripts/run_hmm_baseline.py
"""

import numpy as np
import pytest

from neurojax.models.tests.test_hmm_oracle_parity import (
    ORACLE_DIR,
    _fractional_occupancy,
    _label_match_accuracy,
)

_BASE = ORACLE_DIR / "osl_baseline"

_SIM_FILES = {
    "timeseries": ORACLE_DIR / "osl_sim_timeseries.npy",
    "states_true": ORACLE_DIR / "osl_sim_states_true.npy",
    "means_true": ORACLE_DIR / "osl_sim_means_true.npy",
    "covs_true": ORACLE_DIR / "osl_sim_covs_true.npy",
}
_FIT_FILES = {
    "gamma": _BASE / "gamma.npy",
    "means": _BASE / "means.npy",
    "covariances": _BASE / "covariances.npy",
    "trans_prob": _BASE / "trans_prob.npy",
}


@pytest.fixture(scope="module")
def art():
    """Load all oracle artifacts; missing files fail loudly (RED)."""
    missing = [str(p) for p in {**_SIM_FILES, **_FIT_FILES}.values() if not p.exists()]
    if missing:
        pytest.fail(
            "Oracle baseline not built — missing:\n  "
            + "\n  ".join(missing)
            + "\nRun containers/scripts/{generate_synthetic,run_hmm_baseline}.py "
            "with DATA_DIR=tests/data/oracle_osl/hmm."
        )
    return {k: np.load(p) for k, p in {**_SIM_FILES, **_FIT_FILES}.items()}


def test_shapes_consistent(art):
    T, C = art["timeseries"].shape
    S = art["means"].shape[0]
    assert art["means"].shape == (S, C)
    assert art["covariances"].shape == (S, C, C)
    assert art["trans_prob"].shape == (S, S)
    assert art["gamma"].shape == (T, S)
    assert art["means_true"].shape == (S, C)


def test_all_finite(art):
    for k, v in art.items():
        assert np.all(np.isfinite(v)), f"non-finite values in {k}"


def test_trans_prob_is_stochastic(art):
    P = art["trans_prob"]
    assert np.all(P >= -1e-8), "negative transition probabilities"
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-4)


def test_covariances_spd(art):
    covs = art["covariances"]
    for s, cov in enumerate(covs):
        np.testing.assert_allclose(cov, cov.T, atol=1e-5, err_msg=f"cov[{s}] not symmetric")
        assert np.linalg.eigvalsh(cov).min() > 0, f"cov[{s}] not positive-definite"


def test_gamma_normalized(art):
    np.testing.assert_allclose(art["gamma"].sum(axis=1), 1.0, atol=1e-4)


def test_oracle_recovers_true_states(art):
    """The point of the oracle: osl-dynamics must recover the generating state
    time course (its gamma must match the simulated states)."""
    true = art["states_true"].argmax(axis=1)
    est = art["gamma"].argmax(axis=1)
    S = art["means"].shape[0]
    acc = _label_match_accuracy(true, est, S)
    assert acc > 0.9, f"oracle state-recovery accuracy {acc:.3f} below 0.9"


def test_oracle_fractional_occupancy_matches_truth(art):
    true = art["states_true"].argmax(axis=1)
    S = art["means"].shape[0]
    fo_true = _fractional_occupancy(true, S)
    fo_oracle = art["gamma"].mean(axis=0)
    np.testing.assert_allclose(np.sort(fo_oracle), np.sort(fo_true), atol=0.05)
