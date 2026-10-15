"""Numerical comparison of the JAX CMC-CSD port against SPM's forward model.

The reference ``tests/data/dcm/cmc_csd_spm25_oracle.mat`` was produced by live
SPM 25.01 (``scripts/spm_oracle``): ``spm_csd_mtf`` for the single- and
four-node LFP architectures of SPM 26's DCM M/EEG test 6, at the prior mean
and at a perturbed parameter set.  The port must reproduce the cross-spectra
to 1e-6 relative and SPM's priors exactly.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import loadmat

ORACLE = os.path.join(os.path.dirname(__file__), "data", "dcm", "cmc_csd_spm25_oracle.mat")


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


@pytest.fixture(scope="module")
def oracle():
    return loadmat(ORACLE, squeeze_me=True, struct_as_record=False)


def _ours(arch, perturb):
    from neurojax.dcm import cmc_priors, four_node_test_architecture
    if arch == "single":
        pE, _ = cmc_priors([np.zeros((1, 1))] * 3, [np.zeros((1, 1))], np.ones((1, 1))); X = np.zeros((1, 0))
    else:
        A, B, C, X = four_node_test_architecture(); pE, _ = cmc_priors(A, B, C)
    P = dict(pE)
    if perturb:
        P["G"] = P["G"] + 0.2; P["T"] = P["T"] - 0.1; P["D"] = P["D"] + 0.3; P["a"] = P["a"] + 0.5
    return P, X


@pytest.mark.parametrize("arch", ["single", "four"])
@pytest.mark.parametrize("perturb", [0, 1])
def test_priors_match_spm(oracle, arch, perturb):
    r = oracle[f"{arch}_{perturb}"]
    P, _ = _ours(arch, perturb)
    for f in ("T", "G", "D", "S", "a", "b", "c", "d", "L", "J"):
        np.testing.assert_allclose(np.asarray(P[f]).reshape(-1), _dense(getattr(r.P, f)).reshape(-1), atol=1e-12, err_msg=f)
    for i in range(4):
        theirs = _dense(r.P.A[i]) if isinstance(r.P.A, np.ndarray) and r.P.A.dtype == object else _dense(r.P.A)
        np.testing.assert_allclose(np.asarray(P["A"][i]), theirs, atol=1e-12)


@pytest.mark.parametrize("arch", ["single", "four"])
@pytest.mark.parametrize("perturb", [0, 1])
def test_csd_matches_spm(oracle, arch, perturb):
    import jax
    jax.config.update("jax_enable_x64", True)
    from neurojax.dcm import cmc_csd
    r = oracle[f"{arch}_{perturb}"]
    P, X = _ours(arch, perturb)
    w = _dense(r.Hz).astype(float).reshape(-1)
    ours = cmc_csd(P, w, X)
    ys = r.y if isinstance(r.y, np.ndarray) and r.y.dtype == object else [r.y]
    assert len(ours) == len(ys)
    for G, H in zip(ours, ys):
        H = _dense(H); G = np.asarray(G)
        if H.ndim == 1:
            H = H.reshape(-1, 1, 1)
        assert G.shape == H.shape
        rel = np.abs(G - H).max() / np.abs(H).max()
        assert rel < 1e-6, rel
