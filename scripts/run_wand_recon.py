#!/usr/bin/env python3
"""First INDIVIDUAL MEG source-connectivity map for a WAND subject.

CTF resting MEG -> individual FreeSurfer source space (the recon we just computed)
+ single-sphere conductor (MEG sees through the skull, so no BEM/watershed needed)
+ head-digitization coregistration -> MNE inverse -> aparc parcels -> neurojax
directed_source_connectivity (PDC/DTF). This is the individual-head-model recon we
could NOT do for HBN (no ARM FreeSurfer) -- here the source space + coreg are the
subject's own.  Usage: run_wand_recon.py [sub-XXXXX]
"""
import os
import sys

import jax.numpy as jnp
import mne
import numpy as np

sys.path.insert(0, os.path.expanduser("~/dev/neurojax/src"))
from neurojax.analysis.source_connectivity import directed_source_connectivity

mne.set_log_level("WARNING")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub-01187"
SES_MEG, FS_SUBJ = "ses-01", f"{SUB}_ses-02"
SUBJECTS_DIR = "/data/raw/wand/derivatives/freesurfer"
DS = f"/data/raw/wand/{SUB}/{SES_MEG}/meg/{SUB}_{SES_MEG}_task-resting.ds"
OUT = os.path.expanduser(f"~/wand_recon_{SUB}.npz")

print("[1] read CTF resting MEG", flush=True)
raw = mne.io.read_raw_ctf(DS, preload=True, clean_names=True)
raw.pick_types(meg=True, ref_meg=False, exclude="bads")
raw.filter(1.0, 45.0); raw.resample(250); raw.crop(tmax=min(60.0, raw.times[-1]))
print(f"    {len(raw.ch_names)} MEG ch, sfreq {raw.info['sfreq']}, "
      f"{int(raw.times[-1])}s", flush=True)

print("[2] fiducial coreg + individual source space + sphere", flush=True)
# fiducial-only head->MRI trans (sphere MEG needs no head surface / watershed):
# estimate MRI fiducials from the subject's own FS surfaces, align the CTF
# head-coord cardinals (ident 1/2/3 = LPA/Nasion/RPA) to them.
mri_fids = mne.coreg.get_mni_fiducials(FS_SUBJ, subjects_dir=SUBJECTS_DIR)
mri_by_id = {f["ident"]: np.asarray(f["r"], float) for f in mri_fids}
head_by_id = {d["ident"]: np.asarray(d["r"], float)
              for d in raw.info["dig"] if d["kind"] == 1}
order = [1, 2, 3]
head_pts = np.array([head_by_id[i] for i in order])
mri_pts = np.array([mri_by_id[i] for i in order])
trans_mat = mne.coreg.fit_matched_points(head_pts, mri_pts, out="trans")
trans = mne.transforms.Transform("head", "mri", trans_mat)
resid = np.linalg.norm(mne.transforms.apply_trans(trans_mat, head_pts) - mri_pts,
                       axis=1) * 1e3
print(f"    fiducial coreg residual {resid.mean():.1f} mm", flush=True)
src = mne.setup_source_space(FS_SUBJ, spacing="oct6", subjects_dir=SUBJECTS_DIR,
                             add_dist=False)
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)  # standard MEG sphere
fwd = mne.make_forward_solution(raw.info, trans, src, sphere, meg=True, eeg=False)
print(f"    forward: {fwd['nsource']} individual sources x {fwd['nchan']} MEG",
      flush=True)

print("[3] MNE inverse -> aparc parcels", flush=True)
cov = mne.make_ad_hoc_cov(raw.info)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0 / 9.0, method="MNE")
labels = [l for l in mne.read_labels_from_annot(FS_SUBJ, "aparc",
          subjects_dir=SUBJECTS_DIR) if "unknown" not in l.name.lower()]
ltc = mne.extract_label_time_course(stc, labels, src, mode="mean_flip")
print(f"    {ltc.shape[0]} parcels x {ltc.shape[1]} samples", flush=True)

print("[4] neurojax directed source connectivity (PDC/DTF)", flush=True)
freqs = jnp.linspace(1.0, 45.0, 45)
out = directed_source_connectivity(jnp.asarray(ltc), freqs, fs=raw.info["sfreq"],
                                   order=6)
pdc = np.asarray(out["pdc"])
alpha = (np.asarray(freqs) >= 8) & (np.asarray(freqs) <= 12)
P = pdc[alpha].mean(0); np.fill_diagonal(P, 0.0)
names = [l.name for l in labels]
idx = np.dstack(np.unravel_index(np.argsort(P, axis=None)[::-1], P.shape))[0][:8]
print("    top alpha-band directed edges (source -> target, PDC):", flush=True)
for i, j in idx:
    print(f"      {names[j]:>24} -> {names[i]:<24} {P[i, j]:.3f}", flush=True)
np.savez_compressed(OUT, pdc=pdc, dtf=np.asarray(out["dtf"]),
                    freqs=np.asarray(freqs), parcels=np.array(names))
print(f"[done] {SUB}: individual FS source space + sphere + dig-coreg -> {OUT}",
      flush=True)
