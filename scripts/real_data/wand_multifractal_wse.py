#!/usr/bin/env python
"""Weak-scaling-exponent (WSE) multifractal analysis of the WAND connectome-harmonic
coefficient time series (same ``harmonic_coeffs.npy`` used for the TE directed-coupling
analysis, `wand_jax_te.py`).

Dumeur, Saes, Abry, Ciuciu, Wendt, Jaffard (2025, arXiv:2503.16892) introduce a
(theta,omega)-leader / weak-scaling-exponent multifractal formalism that -- unlike the
standard wavelet-leader or p-leader formalisms -- needs no a priori global regularity
assumption on the signal, which the paper shows real MEG sensor time series routinely
violate (negative estimated H_min breaks the p-leader approach outright without an ad
hoc large fractional-integration order). We check here whether the WAND harmonic
coefficients hit the same problem, and whether the low-order harmonics driving the
~0.1 Hz solenoidal circulation are monofractal (c2~0, consistent with the current
Gaussian/fGn-style Langevin noise model) or genuinely multifractal (c2<0, intermittent/
multiplicative structure not captured by that model).

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_multifractal_wse.py
"""

import os

import numpy as np
import pymultifracs as pmf

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
THETA, OMEGA = 0.5, 1
SCALING_RANGE = [(3, 12)]           # dyadic scales 2^3..2^12 samples @ 25 Hz (~0.3-160s)
Q = np.linspace(-4, 4, 17)


def main():
    env = np.load(os.path.join(OUT, "harmonic_coeffs.npy"))    # (T, 15) @ 25 Hz, z-scored
    T, n = env.shape
    print(f"harmonic coeffs {env.shape}; WSE multifractal analysis per harmonic "
          f"(theta={THETA}, omega={OMEGA})", flush=True)

    rows = []
    for h in range(n):
        WT = pmf.wavelet_analysis(env[:, h], wt_name="db3", normalization=1)
        wse = WT.get_wse(theta=THETA, omega=OMEGA, gamint=0)
        out = pmf.mfa(wse, SCALING_RANGE, q=Q, n_cumul=3, estimates="cms")
        c1, c2, c3 = np.asarray(out.cumulants.log_cumulants).squeeze()
        hq = np.asarray(out.spectrum.hq).squeeze()
        Dq = np.asarray(out.spectrum.Dq).squeeze()
        mask = Dq > 0
        width = hq[mask].max() - hq[mask].min() if mask.any() else 0.0
        rows.append((h + 1, c1, c2, c3, width))
        print(f"  H{h + 1:>2d}: c1(H)={c1:+.4f}  c2(multifractality)={c2:+.4f}  "
              f"c3={c3:+.4f}  spectrum width={width:.4f}", flush=True)

    rows = np.array([(r[1], r[2], r[3], r[4]) for r in rows])
    c1, c2, c3, width = rows.T
    print("\n=====  summary across 15 harmonics  =====", flush=True)
    print(f"c1 (H, self-similarity): mean={c1.mean():+.4f} range=[{c1.min():+.4f}, "
          f"{c1.max():+.4f}]  ({int((c1 < 0).sum())}/{n} negative)", flush=True)
    print(f"c2 (multifractality):    mean={c2.mean():+.4f} range=[{c2.min():+.4f}, "
          f"{c2.max():+.4f}]  ({int((c2 < 0).sum())}/{n} negative)", flush=True)
    print(f"spectrum width:          mean={width.mean():.4f} range=[{width.min():.4f}, "
          f"{width.max():.4f}]", flush=True)
    print("\nnegative c1 (H_min-like) would have broken the standard p-leader formalism "
          "without an ad hoc large fractional-integration order (Dumeur et al. 2025); "
          "the WSE formalism sidesteps that requirement entirely.", flush=True)
    print("c2 << 0 => genuine (intermittent/multiplicative) multifractality, not captured "
          "by the current Gaussian/fGn-style Langevin noise model; c2~=0 => monofractal, "
          "consistent with it.", flush=True)


if __name__ == "__main__":
    main()
