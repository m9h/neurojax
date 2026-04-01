#!/usr/bin/env python3
"""
Validate thalamic connectivity segmentation against 7T quantitative MRI.

Uses WAND ses-06 7T data:
  - Multi-echo GRE (0.67mm, 7 echoes) → T2* map (iron sensitivity)
  - MP2RAGE (0.7mm) → T1 map proxy (myelin/iron sensitivity)

Plus ses-04 7T hires FreeSurfer ThalamicNuclei segmentation.

Runs inside freesurfer-tracula container.

Usage:
    python3 /scripts/30_thalamic_7T_validation.py sub-08033

References:
    Johansen-Berg et al. 2005, Cerebral Cortex 15(1):31-39
    Behrens et al. 2003, Nature Neuroscience 6(7):750-757
"""

import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "sub-08033"
SES = "ses-02"
FS_SUBJ = f"{SUBJECT}_{SES}"

WAND_ROOT = Path("/data/raw/wand")
FS_DIR = Path(f"/subjects/{FS_SUBJ}")
THAL_DIR = Path(f"/subjects/thalamic_segmentation/{FS_SUBJ}")
RESULTS = THAL_DIR / "results"
QC = THAL_DIR / "qc"

# Connectivity labels from find_the_biggest (alphabetical filename order)
# Need to verify actual ordering from the seeds_to_* files
CONN_LABELS = {}

# ---------------------------------------------------------------
# Step 1: Determine find_the_biggest label ordering
# ---------------------------------------------------------------
print("=== Thalamic 7T Validation ===")
print(f"Subject: {FS_SUBJ}\n")

print(">>> Determining connectivity label order")
import glob
for hemi in ["lh", "rh"]:
    seeds = sorted(glob.glob(
        str(THAL_DIR / f"probtrackx/{hemi}/seeds_to_*")))
    print(f"  {hemi}:")
    for i, s in enumerate(seeds, 1):
        name = Path(s).name.replace(f"seeds_to_{hemi}_", "").replace("_diff.nii.gz", "")
        print(f"    {i} = {name}")
        CONN_LABELS[(hemi, i)] = name

# ---------------------------------------------------------------
# Step 2: Fit T2* from multi-echo GRE
# ---------------------------------------------------------------
print("\n>>> Step 2: T2* fitting from 7T multi-echo GRE")

t2star_file = THAL_DIR / "qc" / "t2star_map.nii.gz"

if not t2star_file.exists():
    echo_times = []
    echo_data = []
    for i in range(1, 8):
        efile = WAND_ROOT / f"{SUBJECT}/ses-06/anat/{SUBJECT}_ses-06_echo-0{i}_part-mag_MEGRE.nii.gz"
        if efile.exists():
            img = nib.load(str(efile))
            echo_data.append(img.get_fdata())
            echo_times.append(i * 0.005)  # 5ms spacing starting at 5ms
            if i == 1:
                affine = img.affine
                header = img.header

    echo_times = np.array(echo_times)
    # Handle possible dimension mismatch across echoes (truncate to min slices)
    min_slices = min(d.shape[2] for d in echo_data)
    echo_data = np.array([d[:, :, :min_slices] for d in echo_data])
    print(f"  {len(echo_times)} echoes, TEs: {echo_times*1000} ms, slices: {min_slices}")

    # Log-linear T2* fit: log(S) = log(S0) - TE/T2*
    mask = echo_data[0] > np.percentile(echo_data[0][echo_data[0] > 0], 5)
    log_data = np.zeros_like(echo_data)
    log_data[:, mask] = np.log(echo_data[:, mask] + 1e-10)

    # Least squares fit
    A = np.column_stack([np.ones_like(echo_times), -echo_times])
    shape = echo_data.shape[1:]
    log_flat = log_data.reshape(len(echo_times), -1)

    params = np.linalg.lstsq(A, log_flat, rcond=None)[0]
    r2star = params[1].reshape(shape)  # 1/T2*
    r2star[~mask] = 0
    r2star[r2star < 0] = 0
    r2star[r2star > 200] = 0  # clip extreme values (Hz)

    t2star = np.zeros_like(r2star)
    valid = r2star > 0
    t2star[valid] = 1000.0 / r2star[valid]  # T2* in ms
    t2star[t2star > 100] = 0  # clip unreasonable values

    nib.save(nib.Nifti1Image(t2star, affine, header), str(t2star_file))
    print(f"  T2* map saved: {t2star_file}")
    print(f"  Range: {t2star[valid].min():.1f} - {t2star[valid].max():.1f} ms")
else:
    print(f"  T2* map already exists")
    t2star_img = nib.load(str(t2star_file))
    t2star = t2star_img.get_fdata()

# ---------------------------------------------------------------
# Step 3: Load connectivity segmentation and FS nuclei
# ---------------------------------------------------------------
print("\n>>> Step 3: Load segmentations")

conn_segs = {}
for hemi in ["lh", "rh"]:
    seg_file = RESULTS / f"{hemi}_thalamus_seg_anat.nii.gz"
    if seg_file.exists():
        conn_segs[hemi] = nib.load(str(seg_file)).get_fdata()
        print(f"  {hemi} connectivity seg loaded")

# FS ThalamicNuclei from ses-02 (3T)
fs_thal_3t = FS_DIR / "mri" / "ThalamicNuclei.v13.T1.mgz"
# FS ThalamicNuclei from ses-04 (7T hires)
fs_thal_7t = Path(f"/subjects/{SUBJECT}_ses-04/mri/ThalamicNuclei.v13.T1.mgz")

for label, path in [("3T", fs_thal_3t), ("7T", fs_thal_7t)]:
    if path.exists():
        print(f"  FS ThalamicNuclei ({label}): found")
    else:
        print(f"  FS ThalamicNuclei ({label}): not found")

# ---------------------------------------------------------------
# Step 4: Map FS nuclei to functional groups for comparison
# ---------------------------------------------------------------
print("\n>>> Step 4: FS nuclei → functional group mapping")

# FreeSurfer ThalamicNuclei label IDs → Johansen-Berg functional groups
# Based on known thalamo-cortical projection patterns
FS_NUCLEI_TO_FUNCTION = {
    # Prefrontal (MD complex)
    "prefrontal": {
        "Left": [8109, 8110],   # Left-MDm, Left-MDl
        "Right": [8209, 8210],  # Right-MDm, Right-MDl
    },
    # Premotor (VA complex)
    "premotor": {
        "Left": [8115, 8116],   # Left-VA, Left-VAmc
        "Right": [8215, 8216],
    },
    # Primary Motor (VL complex)
    "motor": {
        "Left": [8112, 8113],   # Left-VLa, Left-VLp
        "Right": [8212, 8213],
    },
    # Somatosensory (VP complex)
    "somatosensory": {
        "Left": [8111],         # Left-VPL
        "Right": [8211],
    },
    # Posterior Parietal (Pulvinar + LP)
    "posterior_parietal": {
        "Left": [8104, 8105, 8106, 8107, 8120],  # PuA, PuM, PuL, PuI, LD
        "Right": [8204, 8205, 8206, 8207, 8220],
    },
    # Temporal (MGN + part of pulvinar)
    "temporal": {
        "Left": [8103],         # Left-MGN
        "Right": [8203],
    },
    # Occipital (LGN)
    "occipital": {
        "Left": [8102],         # Left-LGN
        "Right": [8202],
    },
}

# ---------------------------------------------------------------
# Step 5: Compute comparison metrics
# ---------------------------------------------------------------
print("\n>>> Step 5: Quantitative comparison")

if fs_thal_3t.exists() and conn_segs:
    fs_img = nib.load(str(fs_thal_3t))
    fs_data = fs_img.get_fdata()

    # Resample connectivity seg to match FS ThalamicNuclei grid if needed
    conn_segs_resampled = {}
    for hemi in ["lh", "rh"]:
        seg_file = RESULTS / f"{hemi}_thalamus_seg_anat.nii.gz"
        seg_img = nib.load(str(seg_file))
        if seg_img.shape != fs_img.shape:
            from scipy.ndimage import affine_transform
            # Compute voxel mapping between the two volumes
            # Both should be in FreeSurfer orig space, just different grids
            seg2fs_vox = np.linalg.inv(fs_img.affine) @ seg_img.affine
            conn_segs_resampled[hemi] = affine_transform(
                seg_img.get_fdata(), np.linalg.inv(seg2fs_vox[:3, :3]),
                offset=-np.linalg.inv(seg2fs_vox[:3, :3]) @ seg2fs_vox[:3, 3],
                output_shape=fs_img.shape, order=0)
        else:
            conn_segs_resampled[hemi] = seg_img.get_fdata()

    print("\n  Connectivity vs FS Nuclei overlap (Dice):")
    print(f"  {'Region':<20} {'lh Dice':<10} {'rh Dice':<10}")
    print(f"  {'-'*40}")

    results_csv = []
    for func_name, nuclei_dict in FS_NUCLEI_TO_FUNCTION.items():
        for hemi in ["lh", "rh"]:
            if hemi not in conn_segs_resampled:
                continue
            side = "Left" if hemi == "lh" else "Right"
            label_ids = nuclei_dict[side]

            # FS nuclei mask
            fs_mask = np.isin(fs_data, label_ids)

            # Connectivity label for this function
            conn_label = None
            for (h, idx), name in CONN_LABELS.items():
                if h == hemi and name == func_name:
                    conn_label = idx
                    break
            if conn_label is None:
                continue

            conn_mask = conn_segs_resampled[hemi] == conn_label

            # Dice
            intersection = np.sum(fs_mask & conn_mask)
            dice = 2 * intersection / (np.sum(fs_mask) + np.sum(conn_mask) + 1e-10)

            results_csv.append({
                "hemi": hemi, "region": func_name,
                "fs_voxels": int(np.sum(fs_mask)),
                "conn_voxels": int(np.sum(conn_mask)),
                "dice": float(dice),
            })

    # Print results grouped
    for func_name in FS_NUCLEI_TO_FUNCTION:
        lh_dice = [r["dice"] for r in results_csv if r["region"] == func_name and r["hemi"] == "lh"]
        rh_dice = [r["dice"] for r in results_csv if r["region"] == func_name and r["hemi"] == "rh"]
        lh_str = f"{lh_dice[0]:.3f}" if lh_dice else "n/a"
        rh_str = f"{rh_dice[0]:.3f}" if rh_dice else "n/a"
        print(f"  {func_name:<20} {lh_str:<10} {rh_str:<10}")

    # Save CSV
    csv_path = RESULTS / "fs_nuclei_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("hemi,region,fs_voxels,conn_voxels,dice\n")
        for r in results_csv:
            f.write(f"{r['hemi']},{r['region']},{r['fs_voxels']},{r['conn_voxels']},{r['dice']:.4f}\n")
    print(f"\n  Saved: {csv_path}")

# ---------------------------------------------------------------
# Step 6: T2* values per connectivity region (iron validation)
# ---------------------------------------------------------------
print("\n>>> Step 6: T2* per connectivity region (iron contrast)")
print("  Expected: Motor regions → shorter T2* (more iron)")
print("            Prefrontal (MD) → longer T2* (less iron)\n")

# T2* map is in ses-06 7T native space — need to register to ses-02 anat
# For now, report that registration is needed
# TODO: Register ses-06 MEGRE to ses-02 T1w via ses-04 7T as intermediate
print("  NOTE: T2* sampling requires cross-session registration (ses-06 → ses-02)")
print("  Available chain: ses-06 7T → ses-04 7T (same scanner) → ses-02 3T")
print("  This will be implemented as a follow-up step.")

# ---------------------------------------------------------------
# Step 7: Compare 3T vs 7T FS ThalamicNuclei
# ---------------------------------------------------------------
print("\n>>> Step 7: 3T vs 7T FreeSurfer ThalamicNuclei comparison")

if fs_thal_3t.exists() and fs_thal_7t.exists():
    fs3t = nib.load(str(fs_thal_3t)).get_fdata()
    fs7t = nib.load(str(fs_thal_7t)).get_fdata()

    # Both are in their respective FreeSurfer spaces
    # Direct comparison requires registration
    print("  Both segmentations available.")
    print("  NOTE: Direct voxel comparison requires ses-04 → ses-02 registration.")
    print("  Volume comparison (resolution-independent):")

    # Read volume files
    for label, path_stem in [("3T", FS_DIR), ("7T", Path(f"/subjects/{SUBJECT}_ses-04"))]:
        vol_file = path_stem / "mri" / "ThalamicNuclei.v13.T1.volumes.txt"
        if vol_file.exists():
            print(f"\n  {label} ThalamicNuclei volumes:")
            with open(vol_file) as f:
                for line in f:
                    if "Whole" in line or "Left" in line.split()[0][:4] or "Right" in line.split()[0][:5]:
                        if "Whole" in line:
                            print(f"    {line.strip()}")
        else:
            print(f"  {label} volumes file not found")

print("\n=== Validation complete ===")
