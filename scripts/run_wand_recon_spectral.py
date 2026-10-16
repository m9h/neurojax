#!/usr/bin/env python3
"""CiftiStorm-equivalent spectral recon for ONE WAND subject (handoff step 5).

CTF resting MEG -> individual FreeSurfer source space + sphere forward (as in
run_wand_recon.py) -> neurojax ``Leadfield`` (depth correction + QC) -> aparc
parcel gain -> multitaper sensor CSD -> ``higgs_spectral`` (joint source CSD +
sparse Hermitian source precision per frequency) -> coherence / imaginary
coherency / partial coherence on the source CSD + HIGGS leakage diagnostic.
Usage: run_wand_recon_spectral.py [sub-XXXXX]
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import mne
import numpy as np

sys.path.insert(0, os.path.expanduser("~/dev/neurojax/src"))
from neurojax.analysis.connectivity_spectral import (
    imaginary_coherency_from_cpsd,
    partial_coherence_from_cpsd,
)
from neurojax.analysis.multitaper import multitaper_cpsd
from neurojax.analysis.source_connectivity import leakage_index
from neurojax.analysis.state_spectra import coherence_from_cpsd
from neurojax.io.cifti import save_pconn
from neurojax.source.higgs import higgs_select_alpha
from neurojax.source.leadfield import Leadfield, leadfield_qc

mne.set_log_level("WARNING")
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub-01187"
SES_MEG, FS_SUBJ = "ses-01", f"{SUB}_ses-02"
SUBJECTS_DIR = "/data/raw/wand/derivatives/freesurfer"
DS = f"/data/raw/wand/{SUB}/{SES_MEG}/meg/{SUB}_{SES_MEG}_task-resting.ds"
# NOISE_ALPHA=<float> in the environment switches on the full hgLASSO noise GGM
NOISE_ALPHA = float(os.environ["NOISE_ALPHA"]) if os.environ.get("NOISE_ALPHA") else None
OUT = os.path.expanduser(f"~/wand_recon_spectral_{SUB}{'_noiseggm' if NOISE_ALPHA else ''}.npz")
PCONN_DIR = f"/data/raw/wand/derivatives/neurojax-meg/{SUB}/{SES_MEG}"
# reported choices (never hidden): depth correction, HIGGS penalty, EM iters,
# multitaper window / bandwidth, frequency grid
DEPTH, N_ITER = 0.8, 20
ALPHAS, EBIC_GAMMA = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0], 0.5
WIN_S, BW_HZ = 2.0, 2.0
FREQS = np.arange(1.0, 46.0, 1.0)

print("[1] read CTF resting MEG", flush=True)
raw = mne.io.read_raw_ctf(DS, preload=True, clean_names=True)
raw.pick_types(meg=True, ref_meg=False, exclude="bads")
raw.filter(1.0, 45.0); raw.resample(250); raw.crop(tmax=min(120.0, raw.times[-1]))
fs = raw.info["sfreq"]
print(f"    {len(raw.ch_names)} MEG ch, sfreq {fs}, {int(raw.times[-1])}s", flush=True)

print("[2] fiducial coreg + individual source space + sphere forward", flush=True)
mri_fids = mne.coreg.get_mni_fiducials(FS_SUBJ, subjects_dir=SUBJECTS_DIR)
mri_by_id = {f["ident"]: np.asarray(f["r"], float) for f in mri_fids}
head_by_id = {d["ident"]: np.asarray(d["r"], float)
              for d in raw.info["dig"] if d["kind"] == 1}
head_pts = np.array([head_by_id[i] for i in (1, 2, 3)])
mri_pts = np.array([mri_by_id[i] for i in (1, 2, 3)])
trans = mne.transforms.Transform(
    "head", "mri", mne.coreg.fit_matched_points(head_pts, mri_pts, out="trans"))
src = None
for spacing in ("oct6", "oct5", "oct4"):
    try:
        src = mne.setup_source_space(FS_SUBJ, spacing=spacing, subjects_dir=SUBJECTS_DIR,
                                     add_dist=False)
        break
    except RuntimeError as e:
        print(f"    spacing={spacing} failed ({e}); falling back", flush=True)
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
fwd = mne.make_forward_solution(raw.info, trans, src, sphere, meg=True, eeg=False)
print(f"    forward: {fwd['nsource']} sources ({spacing}) x {fwd['nchan']} MEG", flush=True)

print(f"[3] neurojax Leadfield (depth={DEPTH}) + QC", flush=True)
lf = Leadfield.from_mne_forward(fwd, depth=DEPTH)
G = np.asarray(lf.gain())
qc = leadfield_qc(G)
print(f"    vertex gain {G.shape}: cond={qc['condition_number']:.3g}, "
      f"rank={qc['effective_rank']}/{min(G.shape)} at tol={qc['rank_tol']:.3g}, "
      f"{qc['depth_outliers'].size} depth outliers", flush=True)

# aparc parcel gain: mean fixed-orientation column within each label (the same
# parcel topographies run_wand_recon.py uses for its leakage flag)
labels = [l for l in mne.read_labels_from_annot(FS_SUBJ, "aparc", subjects_dir=SUBJECTS_DIR)
          if "unknown" not in l.name.lower()]
vert = [s["vertno"] for s in fwd["src"]]
offset = [0, len(vert[0])]
cols, names = [], []
for lab in labels:
    h = 0 if lab.hemi == "lh" else 1
    idx = np.flatnonzero(np.isin(vert[h], lab.vertices)) + offset[h]
    if idx.size == 0:
        print(f"    dropping empty parcel {lab.name}", flush=True)
        continue
    cols.append(G[:, idx].mean(1)); names.append(lab.name)
G_parc = np.stack(cols, 1)
# HIGGS is scale-sensitive (alpha, the noise floor and the unit source prior all
# live on the gain's scale): bring the MEG gain (~1e-6 T/Am per parcel) to unit
# mean column norm so unit-variance sources map to O(1) sensor power like the
# z-scored data. The scale is reported and saved.
G_SCALE = float(np.linalg.norm(G_parc, axis=0).mean())
G_parc = jnp.asarray(G_parc / G_SCALE)
qc_p = leadfield_qc(G_parc)
print(f"    parcel gain rescaled by 1/{G_SCALE:.3g} (unit mean column norm)", flush=True)
print(f"    parcel gain {G_parc.shape}: cond={qc_p['condition_number']:.3g}, "
      f"rank={qc_p['effective_rank']} at tol={qc_p['rank_tol']:.3g}", flush=True)

print(f"[4] multitaper sensor CSD ({WIN_S}s windows, bandwidth {BW_HZ} Hz)", flush=True)
X = raw.get_data().T  # (T, C)
X = (X - X.mean(0)) / X.std(0)  # unit-variance channels; G columns are relative anyway
n_win = int(len(X) // (WIN_S * fs))
L = int(WIN_S * fs)
csd_acc = None
for w in range(n_win):
    cpsd, f_axis = multitaper_cpsd(jnp.asarray(X[w * L:(w + 1) * L]), fs=fs, bandwidth=BW_HZ)
    csd_acc = cpsd if csd_acc is None else csd_acc + cpsd
csd = csd_acc / n_win
f_idx = np.array([int(np.argmin(np.abs(np.asarray(f_axis) - f))) for f in FREQS])
csd = csd[f_idx]
print(f"    sensor CSD {csd.shape} from {n_win} windows", flush=True)

n_tapers = int(2 * BW_HZ - 1)
N_EFF = n_win * n_tapers  # independent CSD estimates for the EBIC
print(f"[5] higgs_select_alpha over {ALPHAS} (EBIC gamma={EBIC_GAMMA}, n_eff={N_EFF}, "
      f"n_iter={N_ITER}) on {jax.devices()[0].platform}", flush=True)
t0 = time.time()
print(f"    noise model: {'full GGM, noise_alpha=%g' % NOISE_ALPHA if NOISE_ALPHA else 'diagonal'}", flush=True)
sel = higgs_select_alpha(csd, G_parc, ALPHAS, n_eff=N_EFF, gamma=EBIC_GAMMA, n_iter=N_ITER,
                         noise_alpha=NOISE_ALPHA)
ALPHA = sel["alpha"]
n_pairs = len(names) * (len(names) - 1) / 2 * len(FREQS)
for a, e, s_ in zip(ALPHAS, np.asarray(sel["n_edges"]), np.asarray(sel["ebic"])):
    print(f"    alpha={a:<5} edge density={e / n_pairs:.3f} EBIC={s_:.0f}"
          f"{'  <-- selected' if a == ALPHA else ''}", flush=True)
out = {k: np.asarray(v) for k, v in sel["result"].items()}
print(f"    {time.time() - t0:.1f}s; nll first/last = {out['nll'][0]:.4g} / {out['nll'][-1]:.4g}",
      flush=True)
S_x = jnp.asarray(out["source_csd"])
coh = np.asarray(coherence_from_cpsd(S_x))
icoh = np.asarray(imaginary_coherency_from_cpsd(S_x))
pcoh = np.asarray(partial_coherence_from_cpsd(S_x))
Theta = out["precision"]
edges = (np.abs(Theta) > 1e-6)
np.einsum("fii->fi", edges)[:] = False
leak = np.stack([np.asarray(leakage_index(jnp.asarray(out["kernel"][f]) @ G_parc))
                 for f in range(len(FREQS))])
alpha_band = (FREQS >= 8) & (FREQS <= 12)
print(f"    alpha-band precision edges / pair: {edges[alpha_band].mean():.3f}; "
      f"mean off-diag leakage {np.mean(leak[alpha_band][:, ~np.eye(len(names), dtype=bool)]):.3f}",
      flush=True)
P = pcoh[alpha_band].mean(0); np.fill_diagonal(P, 0.0)
top = np.dstack(np.unravel_index(np.argsort(P, axis=None)[::-1], P.shape))[0][:16:2]
print("    top alpha partial-coherence pairs (leakage in brackets):", flush=True)
for i, j in top:
    print(f"      {names[i]:>22} -- {names[j]:<22} pcoh={P[i, j]:.3f} "
          f"[{leak[alpha_band][:, i, j].mean():.2f}]", flush=True)

# CIFTI pconn (step 7): alpha-band means on the aparc parcels of the subject's
# own FreeSurfer surfaces (fs_LR resampling is not done here, so the parcel
# vertex ids refer to the individual white surface of {FS_SUBJ})
os.makedirs(PCONN_DIR, exist_ok=True)
hemi_struct = {"lh": "CIFTI_STRUCTURE_CORTEX_LEFT", "rh": "CIFTI_STRUCTURE_CORTEX_RIGHT"}
parcels = [(lab.name, {hemi_struct[lab.hemi]: lab.vertices}) for lab in labels if lab.name in names]
n_mesh = {hemi_struct["lh"]: int(fwd["src"][0]["np"]), hemi_struct["rh"]: int(fwd["src"][1]["np"])}
for key, mat in (("coh", coh), ("imcoh", np.abs(icoh)), ("pcoh", pcoh)):
    save_pconn(f"{PCONN_DIR}/{SUB}_{SES_MEG}_task-resting_band-alpha_desc-higgs{key}"
               f"{'noiseggm' if NOISE_ALPHA else ''}.pconn.nii",
               mat[alpha_band].mean(0), parcels, n_mesh_vertices=n_mesh)
print(f"    wrote 3 alpha-band pconn files to {PCONN_DIR}", flush=True)
np.savez_compressed(OUT, ebic_alphas=np.array(ALPHAS), ebic=np.asarray(sel["ebic"]), freqs=FREQS, parcels=np.array(names), sensor_csd=np.asarray(csd),
                    source_csd=out["source_csd"],
                    precision=Theta, coherence=coh, imag_coherency=icoh,
                    partial_coherence=pcoh, leakage=leak, nll=out["nll"],
                    gain_parcel=np.asarray(G_parc), gain_scale=G_SCALE, depth=DEPTH, alpha=ALPHA,
                    qc_cond_vertex=qc["condition_number"], qc_cond_parcel=qc_p["condition_number"])
print(f"[done] {SUB} -> {OUT}", flush=True)
