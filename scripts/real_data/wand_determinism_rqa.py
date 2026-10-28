#!/usr/bin/env python
"""Determinism axis for the WAND bands — RQA + recurrence-network dimension oracle.

Irreversibility ≠ determinism: the committed "stochastic cycle, not a limit cycle"
claim was an *absence of evidence* (DMD/SINDy/DYSCO ≈ 0 rotation).  This is the
*positive* test.  Recurrence quantification (pyunicorn) measures determinism
directly: DET (fraction of recurrence points on diagonal lines) and L_max are high
for a deterministic limit cycle/chaos, low for a stochastic process; the
recurrence-network transitivity dimension estimates the *nonlinear* effective
dimension (vs the linear PCA rank).

Crucially DET is *not* zero for a stochastic signal (autocorrelation makes short
diagonals), so the test is the **excess** determinism over an IAAFT surrogate
(spectrum+marginal matched, via the TimeseriesSurrogates.jl oracle): DET_data ≈
DET_surrogate ⇒ no deterministic skeleton beyond the linear autocorrelation ⇒
positively confirms the stochastic-cycle reading.  DET_data ≫ surrogate would
instead reveal hidden determinism the linear methods missed.

Multivariate RQA on the rank-r embedding, windowed (the full T×T recurrence matrix
is infeasible at T≈180k) with a fixed recurrence rate; run under the CPU oracle.

    PYTHONPATH= .venv-oracle/bin/python scripts/real_data/wand_determinism_rqa.py
"""

import os
import subprocess

import numpy as np
from pyunicorn.timeseries import RecurrencePlot, RecurrenceNetwork

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
SCR = os.environ.get("SCRATCH", "/tmp/claude-1000/-home-mhough-dev-neurojax/"
                     "a44a9232-d44b-4b8b-8812-560682c446fa/scratchpad")
ORACLE = os.path.join(os.path.dirname(__file__), "oracle_surrogates")
JULIA = os.path.expanduser("~/.juliaup/bin/julia")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
DS = 5                  # 50 Hz -> 10 Hz (reduce tangential motion / autocorrelation)
WIN = 3000             # window length (samples @ 10 Hz = 300 s)
N_WIN = 8              # windows averaged
RR = 0.05             # fixed recurrence rate (reported)
N_SURR = 20


def rqa(window):
    """DET, LAM, L_max, ENTR, transitivity-dimension for one (W, r) window."""
    rp = RecurrencePlot(window, metric="supremum", recurrence_rate=RR,
                        silence_level=2)
    rn = RecurrenceNetwork(window, metric="supremum", recurrence_rate=RR,
                           silence_level=2)
    return (rp.determinism(l_min=2), rp.laminarity(v_min=2),
            rp.max_diaglength(), rp.diag_entropy(l_min=2),
            rn.transitivity_dim_single_scale())


def windows(Z):
    Zd = Z[::DS]
    starts = np.linspace(0, len(Zd) - WIN, N_WIN).astype(int)
    return [Zd[s:s + WIN] for s in starts]


def surrogate_det(win, seed):
    """DET of IAAFT surrogates of one window (via the Julia oracle)."""
    fin, fout = os.path.join(SCR, "rqa_in.npy"), os.path.join(SCR, "rqa_out.npy")
    np.save(fin, win.astype(np.float64))
    subprocess.run([JULIA, "--startup-file=no",
                    os.path.join(ORACLE, "gen_surrogates.jl"),
                    fin, fout, str(N_SURR), "iaaft", str(seed)],
                   check=True, capture_output=True,
                   env=dict(os.environ, JULIA_PROJECT=ORACLE))
    S = np.load(fout)
    return np.array([RecurrencePlot(S[i], metric="supremum", recurrence_rate=RR,
                                    silence_level=2).determinism(l_min=2)
                     for i in range(len(S))])


def main():
    print(f"=====  WAND determinism (RQA, RR={RR}, {WIN}@{50//DS}Hz × {N_WIN} win)  =====")
    print(f"{'band':>7} {'DET':>7} {'surrDET':>8} {'z':>6} {'LAM':>7} "
          f"{'Lmax':>6} {'ENTR':>6} {'transD':>7}")
    for name in BANDS:
        path = os.path.join(OUT, f"band_emb_{name}.npy")
        if not os.path.exists(path):
            print(f"{name:>7}  (missing {path} — run wand_cache_embeddings.py first)")
            continue
        Z = np.load(path)
        wins = windows(Z)
        stats = np.array([rqa(w) for w in wins])            # (N_WIN, 5)
        det, lam, lmax, entr, transd = stats.mean(0)
        sdet = surrogate_det(wins[len(wins) // 2], 0)        # mid window
        z = (det - sdet.mean()) / (sdet.std() + 1e-12)
        print(f"{name:>7} {det:>7.3f} {sdet.mean():>8.3f} {z:>6.2f} {lam:>7.3f} "
              f"{lmax:>6.0f} {entr:>6.3f} {transd:>7.2f}")
    print("  -> DET≈surrDET (low z) ⇔ no deterministic skeleton beyond autocorrelation")
    print("     ⇒ positively confirms the stochastic cycle; transD = nonlinear dim vs PCA rank.")


if __name__ == "__main__":
    main()
