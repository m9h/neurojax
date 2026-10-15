"""Numerical comparison of the JAX CMC-CSD port against live SPM (needs spm-python + MATLAB Runtime).

Skipped unless ``NEUROJAX_SPM_PYTHON`` points at a Python with ``spm-python``
and a working MATLAB Runtime.  The oracle is produced by running SPM's own
``spm_dcm_neural_priors`` / ``spm_L_priors`` / ``spm_ssr_priors`` /
``spm_dcm_x_neural`` / ``spm_csd_mtf`` in a subprocess for the single-source
and four-node LFP architectures of SPM 26's test 6, at the prior mean and at
a perturbed parameter set, and comparing the cross-spectra with ours.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

SPM_PY = os.environ.get("NEUROJAX_SPM_PYTHON", "")
pytestmark = pytest.mark.skipif(not SPM_PY or not os.path.exists(SPM_PY), reason="set NEUROJAX_SPM_PYTHON to a python with spm-python + MATLAB Runtime")

ORACLE_SCRIPT = r'''
import json, sys, numpy as np
from spm import *
spm('defaults', 'eeg')
def run(arch, perturb):
    if arch == "single":
        A = Cell([np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1))]); B = Cell([np.zeros((1,1))]); C = np.ones((1,1)); X = np.zeros((1,0))
    else:
        A1 = np.zeros((4,4)); A1[1,0]=1; A1[3,2]=1
        A2 = np.zeros((4,4)); A2[0,1]=1; A2[2,3]=1
        A3 = np.zeros((4,4)); A3[2,0]=A3[0,2]=A3[3,1]=A3[1,3]=1
        A = Cell([A1, A2, A3]); B = Cell([((A1+A2+A3)>0).astype(float)]); C = np.array([[1.],[0.],[1.],[0.]]); X = np.array([[0.],[1.]])
    P, pC = spm_dcm_neural_priors(A, B, C, 'CMC', nargout=2)
    Ns = C.shape[0]
    dipfit = Struct(); dipfit.model='CMC'; dipfit.type='LFP'; dipfit.location=0; dipfit.Ns=Ns; dipfit.Nc=Ns
    P, pC = spm_L_priors(dipfit, P, pC, nargout=2)
    P = spm_ssr_priors(P, pC)
    if perturb:
        P.G = np.asarray(P.G) + 0.2; P.T = np.asarray(P.T) - 0.1; P.D = np.asarray(P.D) + 0.3; P.a = np.asarray(P.a) + 0.5
    x, f = spm_dcm_x_neural(P, 'CMC', nargout=2)
    M = Struct(); M.model='CMC'; M.f=f; M.g='spm_gx_erp'; M.x=x; M.n=int(np.size(x)); M.m=Ns; M.u=np.zeros((Ns,1)); M.l=Ns
    M.Hz = np.arange(2,65,dtype=float).reshape(-1,1); M.dt=1/128; M.dipfit=dipfit
    U = Struct(); U.X = X
    y = spm_csd_mtf(P, M, U)
    ys = [np.asarray(t) for t in (y if isinstance(y, (list, tuple, Cell)) else [y])]
    return [{"re": np.real(t).tolist(), "im": np.imag(t).tolist()} for t in ys]
out = {}
for arch in ("single", "four"):
    for perturb in (False, True):
        out[f"{arch}_{int(perturb)}"] = run(arch, perturb)
json.dump(out, open(sys.argv[1], "w"))
'''


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    out = tmp_path_factory.mktemp("spm") / "oracle.json"
    script = tmp_path_factory.mktemp("spm") / "oracle.py"
    script.write_text(ORACLE_SCRIPT)
    r = subprocess.run([SPM_PY, str(script), str(out)], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        pytest.skip(f"SPM oracle failed: {r.stderr[-800:]}")
    return json.load(open(out))


@pytest.mark.parametrize("arch", ["single", "four"])
@pytest.mark.parametrize("perturb", [0, 1])
def test_csd_matches_spm(oracle, arch, perturb):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from neurojax.dcm import cmc_csd, cmc_priors, four_node_test_architecture
    if arch == "single":
        pE, _ = cmc_priors([np.zeros((1, 1))] * 3, [np.zeros((1, 1))], np.ones((1, 1))); X = np.zeros((1, 0))
    else:
        A, B, C, X = four_node_test_architecture(); pE, _ = cmc_priors(A, B, C)
    P = dict(pE)
    if perturb:
        P["G"] = P["G"] + 0.2; P["T"] = P["T"] - 0.1; P["D"] = P["D"] + 0.3; P["a"] = P["a"] + 0.5
    ours = cmc_csd(P, np.arange(2, 65, dtype=float), X)
    ref = oracle[f"{arch}_{perturb}"]
    assert len(ours) == len(ref)
    for G, H in zip(ours, ref):
        H = np.asarray(H["re"]) + 1j * np.asarray(H["im"])
        G = np.asarray(G)
        assert G.shape == H.shape
        rel = np.abs(G - H).max() / np.abs(H).max()
        assert rel < 1e-4, rel
