"""Multi-source spectral DCM on the Chennu 2016 propofol data (CMC, real leadfield).

Four cortical sources (left/right occipital, left/right frontal; fsaverage
ico-5 vertices nearest to the MNI targets) are projected to the 91-channel
EGI montage through the fsaverage three-layer BEM (MNE, fiducials estimated
from the montage).  By default each source's dipole moment is a free
parameter with the cortical normal as prior mean (as the ECD moments of
``spm_L_priors``); ``--fixed-orientation`` pins it to the normal.  As in
``spm_dcm_csd`` the sensor data are reduced to ``--n-modes`` spatial modes
(PCA of each subject's covariance pooled over levels; the same modes for
every level) and the leadfield is projected onto them.  A canonical-microcircuit
network with occipital -> frontal forward and frontal -> occipital backward
connections is inverted per subject x sedation level.

Outputs under ``$CB_OUT/propofol-chennu2016/dcm_sources_free/`` (or
``dcm_sources/`` with ``--fixed-orientation``, ``dcm_sources_stable/`` with
``--stability-margin``): posteriors.csv, fits.csv,
spectra.csv (observed vs predicted log power of mode 0), contrasts.csv,
leadfield_<subject>.npz (free-moment leadfield ``L3`` (C, 4, 3), normals
``nn``, modes ``U``) and posterior_<subject>_<condition>.npz.

Usage::

    python scripts/consciousness_bench/dcm_propofol_sources.py [--subjects 02] [--n-jobs 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.datasets import fetch_fsaverage

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent))
from dcm_propofol import CONDITIONS, FREQS, contrasts  # noqa: E402

from neurojax.analysis.multitaper import multitaper_cpsd  # noqa: E402
from neurojax.bench.consciousness_bench import Recording, epoch_recording  # noqa: E402
from neurojax.inverse.csd_dcm import CMCSpec, SpectralDCM  # noqa: E402
from neurojax.io.consciousness_bench import load_chennu_propofol  # noqa: E402

# MNI targets (mm): occipital (visual alpha) and frontal (propofol frontal alpha/beta)
SOURCES = {"LO": (-25, -90, 5), "RO": (25, -90, 5), "LF": (-30, 40, 30), "RF": (30, 40, 30)}
# extrinsic gains at SPM's defaults (spm_fx_cmc E = 200): occipital -> frontal forward, frontal -> occipital backward
A_FORWARD = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [200, 0, 0, 0], [0, 200, 0, 0]], float)  # row = target
A_BACKWARD = np.array([[0, 0, 200, 0], [0, 0, 0, 200], [0, 0, 0, 0], [0, 0, 0, 0]], float)
# homotopic lateral connections (LO<->RO, LF<->RF): in the CMC a lateral connection is a reciprocal
# forward + backward pair, which spm_fx_cmc reduces by 1/5
A_LATERAL = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float) * 200


def network(lateral: bool) -> tuple[np.ndarray, np.ndarray]:
    if lateral:
        return A_FORWARD + A_LATERAL, A_BACKWARD + A_LATERAL
    return A_FORWARD, A_BACKWARD


def build_leadfield(info: mne.Info) -> tuple[np.ndarray, np.ndarray, dict]:
    """Free-moment EEG leadfield ``(n_channels, 4, 3)``, cortical normals ``(4, 3)`` and picked positions."""
    fs_dir = Path(fetch_fsaverage(verbose="ERROR"))
    subjects_dir = fs_dir.parent
    src_all = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif", verbose="ERROR")
    rr = np.concatenate([s["rr"] for s in src_all])  # MRI (surface RAS) m; fsaverage ~ MNI
    nn = np.concatenate([s["nn"] for s in src_all])
    used = np.concatenate([s["vertno"] + (0 if k == 0 else src_all[0]["np"]) for k, s in enumerate(src_all)])
    pos, ori, picked = [], [], {}
    for name, mni in SOURCES.items():
        d = np.linalg.norm(rr[used] - np.asarray(mni) / 1000.0, axis=1)
        v = used[np.argmin(d)]
        pos.append(rr[v]); ori.append(nn[v]); picked[name] = (rr[v] * 1000).round(1).tolist()
    vol = mne.setup_volume_source_space(
        "fsaverage", pos=dict(rr=np.asarray(pos), nn=np.asarray(ori)), subjects_dir=subjects_dir, verbose="ERROR"
    )
    coreg = mne.coreg.Coregistration(info, "fsaverage", subjects_dir, fiducials="estimated")
    coreg.fit_fiducials(verbose="ERROR")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="ERROR")
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    fwd = mne.make_forward_solution(info, coreg.trans, vol, bem, meg=False, eeg=True, verbose="ERROR")
    L3 = fwd["sol"]["data"].reshape(info["nchan"], len(SOURCES), 3)  # average-referenced by MNE for EEG
    return L3, np.asarray(ori), picked


def mode_csd(data: np.ndarray, fs: float, U: np.ndarray, epoch_s: float = 10.0, bandwidth: float = 2.0) -> np.ndarray:
    """Multitaper CSD (n_freq, k, k) of the spatial modes ``U`` (C, k), averaged over epochs.

    ``bandwidth`` is the half-bandwidth in Hz: 1.0 is a single taper, 2.0 three tapers.
    """
    ep = epoch_recording(Recording("x", "x", "x", data, fs), epoch_s=epoch_s)  # (E, C, T)
    acc = None
    for e in ep:
        cpsd, f = multitaper_cpsd(jnp.asarray((U.T @ e).T), fs=fs, bandwidth=bandwidth)
        acc = np.asarray(cpsd) if acc is None else acc + np.asarray(cpsd)
    acc = acc / len(ep)
    f = np.asarray(f)
    out = np.empty((FREQS.size,) + acc.shape[1:], complex)
    for i in range(acc.shape[1]):
        for j in range(acc.shape[2]):
            out[:, i, j] = np.interp(FREQS, f, acc[:, i, j].real) + 1j * np.interp(FREQS, f, acc[:, i, j].imag)
    return out


def fit_subject(subject: str, recs: list, n_modes: int, n_iter: int, n_starts: int, free_orientation: bool, out: Path,
                stability_margin: float | None = None, lateral: bool = False, extrinsic_var: float = 1 / 8,
                bandwidth: float = 2.0, randomise_moments: float = 0.5,
                noise_fraction: float = 0.01, noise_var: float = 1 / 128, delays: tuple[float, float] | None = None):
    jax.config.update("jax_enable_x64", True)  # joblib workers do not inherit the parent's JAX config
    by_cond = {r.condition: r for r in recs}
    if "baseline" not in by_cond:
        return [], [], []
    base = by_cond["baseline"]
    info = mne.create_info(base.channels, base.fs, "eeg")
    montage = mne.io.read_epochs_eeglab(
        Path(os.environ.get("CB_ROOT", "/data/raw/consciousness-bench")) / "propofol-chennu2016" / "unzipped"
        / "Sedation-RestingState" / f"{base.meta['file']}.set", verbose="ERROR").get_montage()
    info.set_montage(montage)
    L3, nn, picked = build_leadfield(info)
    # spatial modes from the covariance pooled over all of the subject's levels: modes
    # taken from the baseline alone make the baseline CSD "aligned" with the modes
    # and every other level not, which PEB then reads as a baseline-vs-rest effect
    cov = sum((lambda X: X @ X.T / X.shape[1])(r.data - r.data.mean(axis=1, keepdims=True)) for r in by_cond.values())
    U, s, _ = np.linalg.svd(cov / len(by_cond))
    U = U[:, :n_modes]
    A_fwd, A_bwd = network(lateral)
    np.savez(out / f"leadfield_{subject}.npz", L3=L3, nn=nn, U=U, explained=(s[:n_modes].sum() / s.sum()), channels=base.channels,
             sources=list(SOURCES), positions_mm=[picked[k] for k in SOURCES], A_forward=A_fwd, A_backward=A_bwd,
             extrinsic_var=extrinsic_var, bandwidth=bandwidth, noise_fraction=noise_fraction, noise_var=noise_var,
             delays=np.asarray(delays if delays is not None else []))
    spec = CMCSpec(n_regions=4, A_forward=jnp.asarray(A_fwd), A_backward=jnp.asarray(A_bwd), extrinsic_var=extrinsic_var,
                   delays=None if delays is None else tuple(delays))
    if free_orientation:
        dcm = SpectralDCM(spec, leadfield=jnp.einsum("cm,cik->mik", U, L3), orientation=jnp.asarray(nn), freqs=jnp.asarray(FREQS),
                          stability_margin=stability_margin, noise_fraction=noise_fraction, noise_var=noise_var)
    else:
        dcm = SpectralDCM(spec, leadfield=jnp.asarray(U.T @ np.einsum("cik,ik->ci", L3, nn)), freqs=jnp.asarray(FREQS),
                          stability_margin=stability_margin, noise_fraction=noise_fraction, noise_var=noise_var)
    post, fits, spectra = [], [], []
    for cond in CONDITIONS:
        if cond not in by_cond:
            continue
        r = by_cond[cond]
        csd = mode_csd(r.data, r.fs, U, bandwidth=bandwidth)
        t0 = time.time()
        res = dcm.invert(jnp.asarray(csd), n_iter=n_iter, n_starts=n_starts, randomise_moments=randomise_moments)
        dt = time.time() - t0
        sd = np.sqrt(np.diag(np.asarray(res.Cp)))
        np.savez(out / f"posterior_{subject}_{cond}.npz", Ep=np.asarray(res.Ep), Cp=np.asarray(res.Cp), pE=np.asarray(dcm.prior_mean),
                 pC=np.asarray(dcm.prior_cov), names=dcm.names, F=float(res.F))
        for name, m, sdev in zip(dcm.names, np.asarray(res.Ep), sd):
            post.append(dict(subject=subject, condition=cond, propofol_ugml=r.meta.get("propofol_ugml"), param=name,
                             mean=float(m), sd=float(sdev)))
        obs = np.log(np.real(np.diagonal(csd, axis1=1, axis2=2)) * res.scale)
        pred = np.log(np.real(np.diagonal(np.asarray(res.prediction), axis1=1, axis2=2)) * res.scale)
        rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
        fits.append(dict(subject=subject, condition=cond, propofol_ugml=r.meta.get("propofol_ugml"), F=float(res.F),
                         h=float(res.h), scale=float(res.scale), n_iter=len(res.F_trace), rmse_logpower=rmse,
                         F_starts=" ".join(f"{x:.1f}" for x in res.F_starts), seconds=dt))
        for f, o, p in zip(FREQS, obs[:, 0], pred[:, 0]):
            spectra.append(dict(subject=subject, condition=cond, f=float(f), observed=float(o), predicted=float(p)))
        ep = dict(zip(dcm.names, np.asarray(res.Ep)))
        max_re = float(dcm.max_real_eigenvalue(res.Ep))
        fits[-1]["max_re_eig"] = max_re
        mom = dcm.moments(res.Ep)
        mom_txt = "" if mom is None else " |m|=" + ",".join(f"{x:.2f}" for x in np.linalg.norm(np.asarray(mom), axis=1))
        print(f"{subject} {cond:9s} F={float(res.F):9.1f} rmse={rmse:.2f} T_ii(LO,LF)={ep['T[0, 2]']:+.2f},{ep['T[2, 2]']:+.2f} "
              f"A_fwd(LF<-LO)={ep['A_forward[2, 0]']:+.2f}{mom_txt} maxRe={max_re:+.1f} {dt:6.0f}s", flush=True)
    return post, fits, spectra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=48)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--fixed-orientation", action="store_true", help="pin dipole moments to the cortical normal")
    ap.add_argument("--lateral", action="store_true", help="add homotopic LO<->RO and LF<->RF reciprocal connections")
    ap.add_argument("--extrinsic-var", type=float, default=1 / 8, help="prior variance of the extrinsic log-gains (SPM 1/8)")
    ap.add_argument("--bandwidth", type=float, default=2.0, help="multitaper half-bandwidth in Hz (1.0 = one taper, 2.0 = three)")
    ap.add_argument("--randomise-moments", type=float, default=0.5,
                    help="fraction of the extra starts that also draw the dipole moments (1 = all, 0 = none)")
    ap.add_argument("--noise-fraction", type=float, default=0.01, help="prior channel-noise power as a fraction of the neural prediction")
    ap.add_argument("--noise-var", type=float, default=1 / 128, help="prior variance of the channel-noise log-amplitudes (SPM 1/128)")
    ap.add_argument("--delays", type=float, nargs=2, default=None, metavar=("INTRINSIC", "EXTRINSIC"),
                    help="conduction delays in s (SPM: 0.001 0.016); default none")
    ap.add_argument("--name", default=None, help="output subdirectory name (default dcm_sources_stable / dcm_sources / dcm_sources_free)")
    ap.add_argument("--stability-margin", type=float, default=None,
                    help="soft barrier keeping max Re eigenvalue below -margin (1/s); outputs go to dcm_sources_stable/")
    ap.add_argument("--out", default=os.environ.get("CB_OUT", "/data/datasets/consciousness-bench"))
    a = ap.parse_args()
    out = Path(a.out) / "propofol-chennu2016" / (a.name or (
        "dcm_sources_stable" if a.stability_margin is not None else "dcm_sources" if a.fixed_orientation else "dcm_sources_free"))
    out.mkdir(parents=True, exist_ok=True)
    recs = load_chennu_propofol(subjects=a.subjects)
    subjects = sorted({r.subject for r in recs})
    print(f"{len(recs)} recordings, {len(subjects)} subjects, {a.n_modes} modes", flush=True)
    results = Parallel(n_jobs=a.n_jobs)(
        delayed(fit_subject)(s, [r for r in recs if r.subject == s], a.n_modes, a.n_iter, a.n_starts, not a.fixed_orientation, out,
                             a.stability_margin, a.lateral, a.extrinsic_var, a.bandwidth, a.randomise_moments,
                             a.noise_fraction, a.noise_var, a.delays)
        for s in subjects
    )
    post = pd.DataFrame([x for r in results for x in r[0]])
    fits = pd.DataFrame([x for r in results for x in r[1]])
    spectra = pd.DataFrame([x for r in results for x in r[2]])
    post.to_csv(out / "posteriors.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    spectra.to_csv(out / "spectra.csv", index=False)
    con = contrasts(post)
    con.to_csv(out / "contrasts.csv", index=False)
    if con.empty:
        print("fewer than 3 subjects: no contrasts")
    else:
        print(con.sort_values(["contrast", "cohen_d"]).to_string(index=False))


if __name__ == "__main__":
    main()
