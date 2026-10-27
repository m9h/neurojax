"""Do the fitted spectral DCMs reproduce the complexity-measure contrasts?

For every ``posterior_<subject>_<condition>.npz`` written by
``dcm_propofol_sources.py`` the posterior-mean canonical-microcircuit model
is simulated at the sensors (the subject's free-moment leadfield on all
channels), the simulated recordings are scored with the bench measures and
contrasted exactly as the real data were (``run_bench.py``: 10 evenly
spaced channels, 10 s epochs, 6 epochs), and the simulated paired Cohen's d
per measure is set beside the observed one.

``--amplitudes`` selects the simulations: ``0`` is the linear-Gaussian
process with the fitted cross-spectrum (what any spectral surrogate also
has), a positive value integrates the nonlinear network with innovations of
that amplitude (the CSD fit does not identify it; see
``SpectralDCM.simulate``).  If a measure's observed contrast survives in the
linear simulation it is spectral; if it appears only at a finite amplitude
the fitted model's nonlinearity accounts for it; if it appears in neither it
is outside the model.

Outputs under the posterior directory: sim_scores.csv (long, with an
``amplitude`` column), sim_contrasts.csv and sim_vs_observed.csv.

Usage::

    python scripts/consciousness_bench/dcm_simulate_measures.py \
        /data/datasets/consciousness-bench/propofol-chennu2016/dcm_sources_free --amplitudes 0 1 3 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent))
from dcm_propofol import CONDITIONS, FREQS  # noqa: E402

from neurojax.analysis.nltsa import MEASURES  # noqa: E402
from neurojax.analysis.spectral_baselines import SPECTRAL_MEASURES  # noqa: E402
from neurojax.bench.consciousness_bench import Recording, contrast, score_recordings  # noqa: E402
from neurojax.inverse.csd_dcm import CMCSpec, SpectralDCM  # noqa: E402

ALL = {**MEASURES, **SPECTRAL_MEASURES}
# the measures whose propofol contrast was not spectral (docs/CONSCIOUSNESS_BENCH.md) plus spectral controls
DEFAULT_MEASURES = [
    "mse_sum", "lyapunov", "chaos_01", "pair_correlation", "branching_ratio", "dfa_alpha_envelope",
    "sample_entropy", "lz76_mean", "permutation_entropy", "hjorth_mobility", "aperiodic_exponent", "higuchi_fd",
    "relpow_alpha", "relpow_beta", "spectral_entropy",
]


def simulate_recording(d: Path, f: Path, amplitude: float, duration_s: float, fs: float, n_channels: int, seed: int):
    jax.config.update("jax_enable_x64", True)  # joblib workers do not inherit the parent's JAX config
    subject, cond = f.stem.split("_")[1:3]
    z = np.load(f, allow_pickle=True)
    lf = np.load(d / f"leadfield_{subject}.npz", allow_pickle=True)
    if "A_forward" not in lf:  # fits from before 2026-10-26 used pre-SPM extrinsic semantics; the simulator would not match them
        raise ValueError(f"{d}: leadfield files carry no network; refit with the current dcm_propofol_sources.py")
    spec = CMCSpec(n_regions=4, A_forward=jnp.asarray(lf["A_forward"]), A_backward=jnp.asarray(lf["A_backward"]),
                   extrinsic_var=float(lf["extrinsic_var"]))
    dcm = SpectralDCM(spec, leadfield=jnp.asarray(lf["L3"]), orientation=jnp.asarray(lf["nn"]), freqs=jnp.asarray(FREQS),
                      noise_fraction=float(lf["noise_fraction"]) if "noise_fraction" in lf else 0.01,
                      noise_var=float(lf["noise_var"]) if "noise_var" in lf else 1 / 128)
    names = list(z["names"])
    if names != dcm.names:
        # the channel-space model has one channel-noise amplitude per channel, the mode-space fit one per mode:
        # map the fitted parameters by name and give the channels the fitted mean per-mode noise log-amplitude
        ep = dict(zip(names, np.asarray(z["Ep"])))
        a_n = np.mean([v for k, v in ep.items() if k.startswith("a_n")])
        Ep = jnp.asarray([ep.get(k, a_n if k.startswith("a_n") else 0.0) for k in dcm.names])
    else:
        Ep = jnp.asarray(z["Ep"])
    key = jax.random.PRNGKey(seed + 1000 * int(subject) + CONDITIONS.index(cond))
    x = dcm.simulate(Ep, fs=fs, duration_s=duration_s, key=key, nonlinear=amplitude > 0,
                     amplitude=amplitude if amplitude > 0 else 1.0)
    idx = np.linspace(0, x.shape[0] - 1, n_channels).round().astype(int)
    return Recording(f"sim-amp{amplitude:g}", subject, cond, np.asarray(x[idx], dtype=np.float64), fs,
                     [str(c) for c in np.asarray(lf["channels"])[idx]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[0.0, 1.0, 3.0, 10.0])
    ap.add_argument("--measures", default=",".join(DEFAULT_MEASURES), help="comma list or 'all'")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--n-channels", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--observed", type=Path, default=None, help="directory with contrast_baseline_vs_<cond>.csv (default: parent)")
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    measures = ALL if a.measures == "all" else {k: ALL[k] for k in a.measures.split(",")}
    files = sorted(a.dir.glob("posterior_*.npz"))
    if a.subjects:
        files = [f for f in files if f.stem.split("_")[1] in a.subjects]
    print(f"{len(files)} posteriors, amplitudes {a.amplitudes}, {len(measures)} measures", flush=True)
    scores, cons = [], []
    for amp in a.amplitudes:
        t0 = time.time()
        recs = Parallel(n_jobs=a.n_jobs)(
            delayed(simulate_recording)(a.dir, f, amp, a.duration, a.fs, a.n_channels, a.seed) for f in files
        )
        print(f"amplitude {amp:g}: simulated in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        df = score_recordings(recs, measures, epoch_s=10.0, max_epochs=6, n_jobs=a.n_jobs, batched=False)
        df["amplitude"] = amp
        scores.append(df)
        for cond in CONDITIONS[1:]:
            c = contrast(df, "baseline", cond)
            c["amplitude"], c["contrast"] = amp, f"baseline_vs_{cond}"
            cons.append(c)
        print(f"amplitude {amp:g}: scored in {time.time() - t0:.0f}s", flush=True)
    scores = pd.concat(scores, ignore_index=True)
    cons = pd.concat(cons, ignore_index=True)
    scores.to_csv(a.dir / "sim_scores.csv", index=False)
    cons.to_csv(a.dir / "sim_contrasts.csv", index=False)
    obs_dir = a.observed or a.dir.parent
    rows = []
    for cond in CONDITIONS[1:]:
        p = obs_dir / f"contrast_baseline_vs_{cond}.csv"
        if not p.exists():
            continue
        obs = pd.read_csv(p).set_index("measure")
        for meas in measures:
            if meas not in obs.index:
                continue
            row = dict(contrast=f"baseline_vs_{cond}", measure=meas, d_observed=obs.loc[meas, "cohen_d"])
            for amp in a.amplitudes:
                sel = cons[(cons.contrast == f"baseline_vs_{cond}") & (cons.amplitude == amp) & (cons.measure == meas)]
                row[f"d_sim_amp{amp:g}"] = float(sel.cohen_d.iloc[0]) if len(sel) else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(a.dir / "sim_vs_observed.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
