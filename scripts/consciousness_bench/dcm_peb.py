"""Second-level (PEB) analysis of per-recording spectral-DCM posteriors.

Reads the ``posterior_<subject>_<condition>.npz`` files written by
``dcm_propofol.py`` / ``dcm_propofol_sources.py`` and fits a parametric
empirical Bayes model with design ``[subject indicators, mild, moderate, recovery]``
(recordings as rows; the subject columns absorb the pairing), reporting each sedation effect on every parameter
with its posterior sd and the Bayesian-model-reduction evidence (nats) for
keeping it.

Usage::

    python scripts/consciousness_bench/dcm_peb.py /data/datasets/consciousness-bench/propofol-chennu2016/dcm_cmc
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


def main(d: Path):
    files = sorted(d.glob("posterior_*.npz"))
    rows, Ep, Cp = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        subject, cond = f.stem.split("_")[1:3]
        rows.append((subject, cond)); Ep.append(z["Ep"]); Cp.append(z["Cp"])
        pE, pC, names = z["pE"], z["pC"], list(z["names"])
    Ep, Cp = np.stack(Ep), np.stack(Cp)
    subjects = sorted({s for s, _ in rows})
    # subject indicator columns (fixed effects for the pairing) + sedation-level columns; no intercept
    X = np.column_stack([[1.0 if s == sub else 0.0 for s, _ in rows] for sub in subjects]
                        + [[1.0 if c == k else 0.0 for _, c in rows] for k in COND])
    n_sub = len(subjects)
    print(f"{len(rows)} recordings, {Ep.shape[1]} parameters, design {X.shape}", flush=True)
    res = peb(jnp.asarray(Ep), jnp.asarray(Cp), jnp.asarray(pE), jnp.asarray(pC), jnp.asarray(X))
    dF = bmr_effects(res)
    sd = np.sqrt(np.diag(np.asarray(res.Cb))).reshape(res.Eb.shape)
    out = []
    for ci, cname in enumerate([f"subject_{s}" for s in subjects] + COND):
        for k, name in enumerate(names):
            out.append(dict(effect=cname, param=name, mean=float(res.Eb[ci, k]), sd=float(sd[ci, k]), dF_bmr=float(dF[ci, k])))
    tab = pd.DataFrame(out)
    tab.to_csv(d / "peb_effects.csv", index=False)
    print(f"F = {float(res.F):.1f}; between-subject sd (first 6): {np.sqrt(np.diag(np.asarray(res.Sigma)))[:6].round(3)}; clipped eigenvalues: {res.n_clipped}")
    sel = tab[tab.effect.isin(COND) & (~tab.param.str.startswith(("a_", "b_")))]
    print(sel[sel.dF_bmr > 3].sort_values(["effect", "dF_bmr"], ascending=[True, False]).round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
