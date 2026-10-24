#!/usr/bin/env python3
"""Individualized HBN EMEG Recon for one subject (real-data front-end).

Wires the subject's own anatomy + EEG into
:func:`neurojax.pipeline.recon.recon_directed_connectivity`:

    HBN EEG  ─┐
              ├─ coregister montage ↔ FreeSurfer subjects_dir
    FS recon ─┘        │
                       ▼  BEM/FEM forward  →  gain (leadfield)
                       ▼
        recon_directed_connectivity(gain, eeg) → source PDC/DTF + leakage

Default subject: sub-NDARAD481FXF (complete modern FreeSurfer from the Legion run;
EEG in HBN BIDS_EEG/cmi_bids_NC; FS6 is also adequate — thickness r=0.98 vs modern).

Requires MNE (+ a BEM-capable forward) and the staged data:
  * EEG:  aws s3 sync --no-sign-request \\
            s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_NC/sub-<S>/ <eeg_dir>/
  * FS :  the subject's FreeSurfer subjects_dir (modern from the Legion, or
          s3://.../derivatives/Freesurfer_version6.0.0/<NDAR>/ — FS6 is adequate).

This is the staging/forward front-end; the inverse→connectivity core is tested in
tests/test_recon_pipeline.py. Run it in an MNE-capable env once the data is staged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from neurojax.pipeline.recon import recon_directed_connectivity


def build_gain(raw, subject: str, subjects_dir: str, trans):
    """BEM forward → (gain (n_sensors, n_sources), src). MNE-based forward."""
    import mne

    src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir,
                                 add_dist=False)
    model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir,
                               conductivity=(0.3, 0.006, 0.3))  # 3-layer EEG
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                    eeg=True, meg=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    return np.asarray(fwd["sol"]["data"]), src


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eeg", required=True, help="HBN EEG file (.set/.fif/BIDS)")
    p.add_argument("--subject", default="sub-NDARAD481FXF")
    p.add_argument("--subjects-dir", required=True, help="FreeSurfer SUBJECTS_DIR")
    p.add_argument("--trans", default="fsaverage",
                   help="coregistration trans (fiducial-based) or a -trans.fif")
    p.add_argument("--fmax", type=float, default=45.0)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--out", default="hbn_recon_connectivity.npz")
    a = p.parse_args()

    import mne

    raw = mne.io.read_raw(a.eeg, preload=True).pick("eeg")
    raw.set_eeg_reference("average", projection=False)
    fs = raw.info["sfreq"]

    # coregister: align EEG montage to the subject's FreeSurfer head (fiducials).
    # neurojax.geometry.bem.coregister_montage writes the -trans; here we accept a
    # precomputed trans (or fsaverage as a template fallback).
    gain, src = build_gain(raw, a.subject, a.subjects_dir, a.trans)

    eeg = jnp.asarray(raw.get_data())                       # (n_sensors, n_times)
    freqs = jnp.linspace(1.0, a.fmax, 60)
    out = recon_directed_connectivity(jnp.asarray(gain), eeg, freqs, fs,
                                      order=a.order)

    np.savez_compressed(
        a.out,
        pdc=np.asarray(out["pdc"]), dtf=np.asarray(out["dtf"]),
        leakage=np.asarray(out["leakage"]), freqs=np.asarray(freqs),
    )
    pdc = np.asarray(out["pdc"])
    print(f"[hbn_recon] {a.subject}: sources={out['sources'].shape} "
          f"pdc={pdc.shape}  mean|PDC|={np.abs(pdc).mean():.3f}  -> {a.out}")
    print(f"[hbn_recon] max off-diagonal leakage = "
          f"{float(np.max(np.asarray(out['leakage']) - np.eye(pdc.shape[1]))):.3f} "
          "(high => prefer a HIGGS inverse; see neurojax.source.higgs)")


if __name__ == "__main__":
    main()
