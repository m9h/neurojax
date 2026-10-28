"""Second-level (PEB) analysis of per-recording spectral-DCM posteriors.

Reads the ``posterior_<subject>_<condition>.npz`` files written by
``dcm_propofol.py`` / ``dcm_propofol_sources.py`` and fits a parametric
empirical Bayes model with design ``[group mean, mild, moderate, recovery, centred subject indicators]``
(recordings as rows; the subject columns absorb the pairing), reporting each sedation effect on every parameter
with its posterior sd and the Bayesian-model-reduction evidence (nats) for
keeping it.

Every design column has the prior ``N(0, pC)``, so a column must not carry an
offset that the data put far from the prior: uncentred subject indicators
(4 recordings each) are shrunk hard toward the prior mean, and the offset common
to every subject then leaks into the sedation columns (20 recordings each) as a
spurious effect of every drug level, recovery included.  The intercept carries
that offset (80 recordings, negligible shrinkage); the subject columns are
centred so they carry only deviations from it.

The between-subject covariance is one component scaled by a single
log-precision (as ``spm_dcm_peb``); ``--per-parameter-gamma`` estimates one
per parameter (slow beyond ~20 parameters).

Usage::

    python scripts/consciousness_bench/dcm_peb.py /data/datasets/consciousness-bench/propofol-chennu2016/dcm_cmc [--per-parameter-gamma]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

from neurojax.inverse.peb import bmr_effects, peb  # noqa: E402

COND = ["mild", "moderate", "recovery"]


def main(d: Path, shared_gamma: bool = True):
    files = sorted(d.glob("posterior_*.npz"))
    rows, Ep, Cp = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        subject, cond = f.stem.split("_")[1:3]
        rows.append((subject, cond)); Ep.append(z["Ep"]); Cp.append(z["Cp"])
        pE, pC, names = z["pE"], z["pC"], list(z["names"])
    Ep, Cp = np.stack(Ep), np.stack(Cp)
    subjects = sorted({s for s, _ in rows})
    # group mean, sedation levels relative to baseline, centred subject indicators (pairing)
    S = np.array([[1.0 if s == sub else 0.0 for sub in subjects] for s, _ in rows])
    X = np.column_stack([np.ones(len(rows))] + [[1.0 if c == k else 0.0 for _, c in rows] for k in COND]
                        + [S - S.mean(axis=0)])
    print(f"{len(rows)} recordings, {Ep.shape[1]} parameters, design {X.shape}", flush=True)
    res = peb(jnp.asarray(Ep), jnp.asarray(Cp), jnp.asarray(pE), jnp.asarray(pC), jnp.asarray(X), shared_gamma=shared_gamma)
    dF = bmr_effects(res, covariates=list(range(1, 1 + len(COND))))  # sedation effects only
    sd = np.sqrt(np.diag(np.asarray(res.Cb))).reshape(res.Eb.shape)
    out = []
    for ci, cname in enumerate(["mean"] + COND + [f"subject_{s}" for s in subjects]):
        for k, name in enumerate(names):
            out.append(dict(effect=cname, param=name, mean=float(res.Eb[ci, k]), sd=float(sd[ci, k]), dF_bmr=float(dF[ci, k])))
    tab = pd.DataFrame(out)
    tab.to_csv(d / "peb_effects.csv", index=False)
    print(f"F = {float(res.F):.1f}; between-subject sd (first 6): {np.sqrt(np.diag(np.asarray(res.Sigma)))[:6].round(3)}; clipped eigenvalues: {res.n_clipped}")
    sel = tab[tab.effect.isin(COND) & (~tab.param.str.startswith(("a_", "b_")))]
    print(sel[sel.dF_bmr > 3].sort_values(["effect", "dF_bmr"], ascending=[True, False]).round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), shared_gamma="--per-parameter-gamma" not in sys.argv[2:])
