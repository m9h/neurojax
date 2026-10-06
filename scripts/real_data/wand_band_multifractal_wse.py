#!/usr/bin/env python
"""Band-resolved WSE multifractal analysis of WAND resting MEG -- the multifractal
counterpart of ``wand_band_langevin.py``.

The band Langevin fit characterizes drift DIRECTIONALITY (gradient vs solenoidal):
delta drift is 64% rotational, gamma 2% (see docs/WAND_THREE_LEG_CROSSTEST.md). It
says nothing about whether the underlying envelope process itself is monofractal
(Gaussian/fGn-like, as the Langevin noise model assumes) or genuinely multifractal
(intermittent/multiplicative). This applies the weak-scaling-exponent formalism
(Dumeur, Saes, Abry, Ciuciu, Wendt, Jaffard 2025, arXiv:2503.16892 -- see also
``wand_multifractal_wse.py`` for the harmonic-coefficient version) to each band's
parcel-averaged amplitude envelope, to check whether multifractal character tracks
rotational strength.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_band_multifractal_wse.py
"""

import os

import jax.numpy as jnp
import numpy as np
import pymultifracs as pmf

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 50.0
THETA, OMEGA = 0.5, 1
Q = np.linspace(-4, 4, 17)
# solenoidal fraction of drift from wand_band_langevin.py (docs/WAND_THREE_LEG_CROSSTEST.md)
SOL_TOT = {"delta": 0.64, "theta": 0.26, "alpha": 0.13, "beta": 0.05, "gamma": 0.02}


def band_envelope(X, lo, hi):
    """Morlet envelope of one band, averaged across parcels -> scalar series @ ENV_FS."""
    fc = jnp.linspace(lo, hi, 6)
    C = morlet_cwt(jnp.asarray(X.T), FS, fc, eeglab_cycles(fc, 3.0, 0.5))   # (68,6,T)
    env = np.asarray(jnp.mean(jnp.abs(C), axis=1))                          # (68, T)
    ds = int(FS / ENV_FS)
    env = env[:, : (env.shape[1] // ds) * ds].reshape(68, -1, ds).mean(2)   # -> ENV_FS
    mean_env = env.mean(0)                                                  # (T,) network envelope
    return (mean_env - mean_env.mean()) / (mean_env.std() + 1e-9)


def main():
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)
    print(f"raw source parcels: {X.shape} @ {FS:.0f} Hz")
    print(f"\n=====  Band-resolved WSE multifractal analysis (envelope {ENV_FS:.0f} Hz)  =====")
    print(f"{'band':>7} {'Hz':>9} {'sol/tot':>8} {'c1(H)':>9} {'c2(MF)':>9} {'width':>8}")

    rows = []
    for name, (lo, hi) in BANDS.items():
        x = band_envelope(X, lo, hi)
        WT = pmf.wavelet_analysis(x, wt_name="db3", normalization=1)
        wse = WT.get_wse(theta=THETA, omega=OMEGA, gamint=0)
        out = pmf.mfa(wse, [(3, 12)], q=Q, n_cumul=3, estimates="cms")
        c1, c2, c3 = np.asarray(out.cumulants.log_cumulants).squeeze()
        hq = np.asarray(out.spectrum.hq).squeeze()
        Dq = np.asarray(out.spectrum.Dq).squeeze()
        mask = Dq > 0
        width = hq[mask].max() - hq[mask].min() if mask.any() else 0.0
        rows.append((name, c1, c2, width))
        print(f"{name:>7} {f'{lo}-{hi}':>9} {SOL_TOT[name]:>8.2f} {c1:>+9.4f} "
              f"{c2:>+9.4f} {width:>8.4f}", flush=True)

    sol = np.array([SOL_TOT[r[0]] for r in rows])
    c2s = np.array([r[2] for r in rows])
    widths = np.array([r[3] for r in rows])
    rho_c2 = float(np.corrcoef(sol, c2s)[0, 1])
    rho_w = float(np.corrcoef(sol, widths)[0, 1])
    print(f"\ncorr(sol/tot, c2)          = {rho_c2:+.3f}")
    print(f"corr(sol/tot, spectrum width) = {rho_w:+.3f}")
    print("  -> if |rho| is small, rotational strength and multifractal character are "
          "independent axes; a large |rho| would mean the same bands driving the "
          "circulation are also the most (or least) multifractal.")


if __name__ == "__main__":
    main()
