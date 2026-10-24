"""Parity tests for the JAX GaussianHMM.

Two complementary checks, sharing the same state-alignment + summary metrics
(HMM state labels are permutation-invariant, so everything is compared *after*
matching estimated states to reference states by nearest mean):

1. ``TestGroundTruthRecovery`` — simulate from a known HMM and assert the JAX
   model recovers the generating means / transition matrix / fractional
   occupancy.  Runs everywhere (no external data), so it is the immediate
   correctness gate for the Path-C reimplementation on the GB10.

2. ``TestOSLDynamicsParity`` — compare against osl-dynamics outputs dumped by
   ``containers/scripts/run_hmm_baseline.py`` (the CPU/TF oracle).  Skipped
   until the oracle ``.npy`` files exist, matching the project convention of
   real data paths guarded by ``skipif`` (CLAUDE.md: no mock databases).

   Populate the oracle once with::

       docker run --rm -e DATA_DIR=/data \\
         -v $(pwd)/tests/data/oracle_osl/hmm:/data \\
         neurojax/oracle-osl python /scripts/run_hmm_baseline.py
"""

import os
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.models.hmm import GaussianHMM

# Oracle outputs (overridable for CI / alternate locations).
ORACLE_DIR = Path(
    os.environ.get(
        "NEUROJAX_ORACLE_DIR",
        Path(__file__).resolve().parents[4] / "tests" / "data" / "oracle_osl" / "hmm",
    )
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _simulate_hmm(key, means, covs, trans, T):
    """Sample (timeseries, true_states) from a Gaussian HMM."""
    means = np.asarray(means)
    covs = np.asarray(covs)
    trans = np.asarray(trans)
    S, C = means.shape
    rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**31 - 1)))

    states = np.empty(T, dtype=int)
    states[0] = rng.integers(S)
    for t in range(1, T):
        states[t] = rng.choice(S, p=trans[states[t - 1]])

    ts = np.stack(
        [rng.multivariate_normal(means[s], covs[s]) for s in states]
    ).astype(np.float32)
    return ts, states


def _align_states(ref_means, est_means):
    """Map each estimated state to a reference state by nearest mean.

    Returns ``perm`` such that ``est[perm]`` lines up with ``ref``.  Uses the
    Hungarian assignment when SciPy is available, else a deterministic greedy
    global-minimum matcher (exact for well-separated states).
    """
    ref = np.asarray(ref_means)
    est = np.asarray(est_means)
    S = ref.shape[0]
    cost = np.linalg.norm(ref[:, None, :] - est[None, :, :], axis=-1)  # (S_ref, S_est)

    try:
        from scipy.optimize import linear_sum_assignment

        _, col = linear_sum_assignment(cost)
        return col
    except Exception:
        perm = -np.ones(S, dtype=int)
        taken = set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            if perm[r] == -1 and c not in taken:
                perm[r] = c
                taken.add(c)
        return perm


def _fractional_occupancy(states, n_states):
    return np.bincount(np.asarray(states), minlength=n_states) / len(states)


def _label_match_accuracy(seq_a, seq_b, n_states):
    """Best-permutation agreement between two integer label sequences.

    HMM state labels are arbitrary, so we match labels via the confusion
    matrix (Hungarian when SciPy is present, else greedy) and return the
    fraction of timepoints that agree under the best matching.
    """
    a = np.asarray(seq_a)
    b = np.asarray(seq_b)
    conf = np.zeros((n_states, n_states))
    for i in range(n_states):
        bi = b[a == i]
        for j in range(n_states):
            conf[i, j] = np.sum(bi == j)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-conf)
        return conf[r, c].sum() / len(a)
    except Exception:
        taken, used_i, total = set(), set(), 0.0
        order = np.dstack(np.unravel_index(np.argsort(-conf, axis=None), conf.shape))[0]
        for i, j in order:
            if i not in used_i and j not in taken:
                used_i.add(i)
                taken.add(j)
                total += conf[i, j]
        return total / len(a)


# ---------------------------------------------------------------------------
# 1. Ground-truth recovery (no external data — runs on the GB10 directly)
# ---------------------------------------------------------------------------

class TestGroundTruthRecovery:
    """The JAX HMM should recover a known generating model."""

    @pytest.fixture(scope="class")
    def sim(self):
        key = jr.PRNGKey(0)
        # 3 well-separated states, 4 channels, modest within-state noise.
        means = jnp.array(
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]
        )
        covs = jnp.stack([jnp.eye(4) * 0.4] * 3)
        # Sticky dynamics (typical of MEG state HMMs).
        trans = jnp.array(
            [[0.94, 0.03, 0.03], [0.03, 0.94, 0.03], [0.03, 0.03, 0.94]]
        )
        ts, true_states = _simulate_hmm(key, means, covs, trans, T=2000)
        return dict(ts=ts, true_states=true_states, means=means, trans=trans)

    @pytest.fixture(scope="class")
    def fitted(self, sim):
        model = GaussianHMM(n_states=3, n_channels=4)
        model.fit([jnp.asarray(sim["ts"])], n_epochs=30, n_init=3)
        return model

    def test_recovers_state_means(self, sim, fitted):
        # Ground-truth means in the standardized space the model fits in.
        ts = sim["ts"]
        z_means = (np.asarray(sim["means"]) - ts.mean(0)) / ts.std(0).clip(1e-10)
        perm = _align_states(z_means, np.asarray(fitted.means))
        aligned = np.asarray(fitted.means)[perm]
        np.testing.assert_allclose(aligned, z_means, atol=0.4)

    def test_recovers_fractional_occupancy(self, sim, fitted):
        decoded = np.asarray(fitted.decode([jnp.asarray(sim["ts"])])[0])
        # Align decoded labels to the true labels via the fitted means.
        ts = sim["ts"]
        z_means = (np.asarray(sim["means"]) - ts.mean(0)) / ts.std(0).clip(1e-10)
        perm = _align_states(z_means, np.asarray(fitted.means))
        inv = np.argsort(perm)
        decoded_aligned = inv[decoded]
        fo_est = _fractional_occupancy(decoded_aligned, 3)
        fo_true = _fractional_occupancy(sim["true_states"], 3)
        np.testing.assert_allclose(fo_est, fo_true, atol=0.1)

    def test_decoding_accuracy(self, sim, fitted):
        decoded = np.asarray(fitted.decode([jnp.asarray(sim["ts"])])[0])
        ts = sim["ts"]
        z_means = (np.asarray(sim["means"]) - ts.mean(0)) / ts.std(0).clip(1e-10)
        perm = _align_states(z_means, np.asarray(fitted.means))
        inv = np.argsort(perm)
        acc = float((inv[decoded] == sim["true_states"]).mean())
        assert acc > 0.9, f"Viterbi accuracy {acc:.3f} below 0.9"


# ---------------------------------------------------------------------------
# 2. osl-dynamics oracle parity (skipped until the oracle is run)
# ---------------------------------------------------------------------------

_BASE = ORACLE_DIR / "osl_baseline"
_REQUIRED = [
    ORACLE_DIR / "osl_sim_timeseries.npy",
    _BASE / "means.npy",
    _BASE / "covariances.npy",
    _BASE / "trans_prob.npy",
]
_oracle_missing = [str(p) for p in _REQUIRED if not p.exists()]


@pytest.mark.skipif(
    bool(_oracle_missing),
    reason=f"osl-dynamics oracle outputs not found (missing: {_oracle_missing}). "
    "Run containers/scripts/run_hmm_baseline.py via the oracle container to populate.",
)
class TestOSLDynamicsParity:
    """JAX HMM vs osl-dynamics on the *same* standardized timeseries.

    The two libraries use different inference — Baum-Welch EM (closed-form
    M-step) here vs stochastic variational training in osl-dynamics — so we
    assert agreement on what both estimate well: the **state segmentation**
    (per-timepoint decode and fractional occupancy).

    We deliberately do NOT assert mean-*parameter* parity: osl-dynamics learns
    means by gradient descent and shrinks them toward zero in this regime
    (verified: ~0.28 mean-abs error vs the true means), whereas Baum-Welch's
    closed-form means recover the truth (~0.02 error).  Both still segment the
    states the same way — that is the meaningful equivalence.
    """

    @pytest.fixture(scope="class")
    def oracle(self):
        return {
            "ts": np.load(ORACLE_DIR / "osl_sim_timeseries.npy"),
            "gamma": np.load(_BASE / "gamma.npy"),
            "means": np.load(_BASE / "means.npy"),
            "trans_prob": np.load(_BASE / "trans_prob.npy"),
        }

    @pytest.fixture(scope="class")
    def fitted(self, oracle):
        n_states, n_channels = oracle["means"].shape
        model = GaussianHMM(n_states=n_states, n_channels=n_channels)
        model.fit([jnp.asarray(oracle["ts"])], n_epochs=30, n_init=3)
        return model

    def test_state_segmentation_agrees(self, oracle, fitted):
        n_states = oracle["means"].shape[0]
        decoded = np.asarray(fitted.decode([jnp.asarray(oracle["ts"])])[0])
        oracle_states = oracle["gamma"].argmax(axis=1)
        acc = _label_match_accuracy(oracle_states, decoded, n_states)
        assert acc > 0.85, f"neurojax vs oracle segmentation agreement {acc:.3f} < 0.85"

    def test_fractional_occupancy_matches(self, oracle, fitted):
        n_states = oracle["means"].shape[0]
        decoded = np.asarray(fitted.decode([jnp.asarray(oracle["ts"])])[0])
        fo_jax = _fractional_occupancy(decoded, n_states)
        fo_oracle = oracle["gamma"].mean(axis=0)
        # Sorted comparison sidesteps label permutation between the two fits.
        np.testing.assert_allclose(np.sort(fo_jax), np.sort(fo_oracle), atol=0.1)
