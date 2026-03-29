#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
# ]
# ///
"""Compare WMH segmentation methods on WAND data.

Methods:
  1. FreeSurfer WMH-SynthSeg (DL, T1w-only, unsupervised)
  2. FSL BIANCA (supervised classifier, T1w + MNI features)

WAND limitation: No FLAIR acquisition. Both methods run on T1w only,
which reduces WMH sensitivity. The comparison quantifies agreement
between methods and spatial distribution of detected lesions.

For BIANCA without training labels: uses LOO across WAND subjects or
the self-training approach (Griffanti et al. 2016).

Outputs:
  - Overlap metrics (Dice, volume agreement)
  - Spatial distribution (periventricular vs deep)
  - Per-method volume estimates
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def run_bianca_self_training(subject, wand_root, fsl_anat_dir):
    """Run BIANCA in self-training mode using WMH-SynthSeg as pseudo-labels.

    Since we don't have manual WMH labels, we use WMH-SynthSeg output
    as training labels for BIANCA. This tests whether BIANCA (with its
    spatial features) agrees with the DL method when given the same
    "ground truth" — i.e., does the spatial model add information?
    """
    deriv = Path(wand_root) / "derivatives"
    adv_dir = deriv / "advanced-freesurfer" / subject
    bianca_dir = deriv / "bianca" / subject
    bianca_dir.mkdir(parents=True, exist_ok=True)

    anat_dir = Path(fsl_anat_dir)

    # BIANCA needs: T1 brain, brain mask, MNI transform, and training labels
    t1_brain = anat_dir / "T1_biascorr_brain.nii.gz"
    brain_mask = anat_dir / "T1_biascorr_brain_mask.nii.gz"
    mni_mat = anat_dir / "T1_to_MNI_lin.mat"
    wmh_synthseg = adv_dir / "wmh_seg.nii.gz"

    if not all(p.exists() for p in [t1_brain, brain_mask, mni_mat]):
        print(f"  Missing fsl_anat outputs for BIANCA")
        return None

    if not wmh_synthseg.exists():
        print(f"  WMH-SynthSeg output not found, skipping BIANCA comparison")
        return None

    # Threshold WMH-SynthSeg to binary mask (it outputs segmentation labels)
    # WMH-SynthSeg labels: 77 = WMH
    print("  Creating BIANCA training mask from WMH-SynthSeg...")
    wmh_img = nib.load(str(wmh_synthseg))
    wmh_data = wmh_img.get_fdata()
    # WMH-SynthSeg outputs labels; WMH is typically label 77
    wmh_binary = (wmh_data == 77).astype(np.float32)
    if wmh_binary.sum() == 0:
        # Try other possible WMH labels or threshold
        wmh_binary = (wmh_data > 0.5).astype(np.float32)
    wmh_mask_path = bianca_dir / "wmh_synthseg_binary.nii.gz"
    nib.save(nib.Nifti1Image(wmh_binary, wmh_img.affine), str(wmh_mask_path))
    wmh_vol_synthseg = float(wmh_binary.sum() * np.prod(wmh_img.header.get_zooms()[:3]))
    print(f"  WMH-SynthSeg volume: {wmh_vol_synthseg:.1f} mm³ "
          f"({wmh_binary.sum():.0f} voxels)")

    # Create BIANCA masterfile (LOO with single subject = self-training)
    masterfile = bianca_dir / "masterfile.txt"
    with open(masterfile, "w") as f:
        f.write(f"{t1_brain} {brain_mask} {wmh_mask_path} {mni_mat}\n")

    # Run BIANCA
    print("  Running BIANCA...")
    bianca_out = bianca_dir / "bianca_output.nii.gz"
    cmd = [
        "bianca",
        f"--singlefile={masterfile}",
        "--querysubjectnum=1",
        "--brainmaskfeaturenum=2",
        "--labelfeaturenum=3",
        "--matfeaturenum=4",
        f"--featuresubset=1,2",
        "--trainingpts=2000",
        "--nonlespts=10000",
        f"--saveclassifierdata={bianca_dir}/classifier",
        f"-o", str(bianca_out),
        "-v",
    ]

    env = os.environ.copy()
    fsldir = env.get("FSLDIR", "/Users/mhough/fsl")
    env["FSLDIR"] = fsldir
    env["PATH"] = f"{fsldir}/bin:" + env.get("PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  BIANCA failed: {result.stderr[-300:]}")
        return None

    print(f"  BIANCA output: {bianca_out}")

    # Threshold BIANCA probability map
    bianca_img = nib.load(str(bianca_out))
    bianca_prob = bianca_img.get_fdata()
    bianca_binary = (bianca_prob > 0.9).astype(np.float32)
    bianca_mask_path = bianca_dir / "bianca_binary_0.9.nii.gz"
    nib.save(nib.Nifti1Image(bianca_binary, bianca_img.affine),
             str(bianca_mask_path))
    bianca_vol = float(bianca_binary.sum() * np.prod(bianca_img.header.get_zooms()[:3]))
    print(f"  BIANCA volume (thr=0.9): {bianca_vol:.1f} mm³")

    # Classify periventricular vs deep
    print("  Classifying periventricular vs deep WMH...")
    pv_cmd = [
        "bianca_perivent_deep",
        f"--biession={bianca_mask_path}",
        f"--querysubjectnum=1",
        f"--brainmaskfeaturenum=2",
        f"--matfeaturenum=4",
        f"--singlefile={masterfile}",
        f"-o", str(bianca_dir / "perivent_deep"),
    ]
    subprocess.run(pv_cmd, capture_output=True, text=True, env=env)

    return {
        "bianca_binary_path": str(bianca_mask_path),
        "bianca_prob_path": str(bianca_out),
        "bianca_volume_mm3": bianca_vol,
        "synthseg_binary_path": str(wmh_mask_path),
        "synthseg_volume_mm3": wmh_vol_synthseg,
    }


def compute_overlap(mask1, mask2):
    """Compute overlap metrics between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)
    sum_vols = np.sum(mask1) + np.sum(mask2)

    dice = 2 * intersection / sum_vols if sum_vols > 0 else 0
    jaccard = intersection / union if union > 0 else 0

    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "intersection_voxels": int(intersection),
        "mask1_voxels": int(np.sum(mask1)),
        "mask2_voxels": int(np.sum(mask2)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare WMH segmentation methods"
    )
    parser.add_argument("subject", help="Subject ID")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    args = parser.parse_args()

    subject = args.subject
    wand = Path(args.wand_root)
    deriv = wand / "derivatives"

    print(f"=== WMH Comparison: {subject} ===")

    # fsl_anat directory
    fsl_anat = (deriv / "fsl-anat" / subject / "ses-03" / "anat" /
                f"{subject}_ses-03_T1w.anat")

    if not fsl_anat.exists():
        print(f"ERROR: fsl_anat not found at {fsl_anat}")
        sys.exit(1)

    # Run BIANCA with WMH-SynthSeg as pseudo-labels
    results = run_bianca_self_training(subject, str(wand), str(fsl_anat))

    if results is None:
        print("  Could not run comparison (missing inputs)")
        return

    # Compute overlap
    print("\n--- Overlap Metrics ---")
    synthseg = nib.load(results["synthseg_binary_path"]).get_fdata()
    bianca = nib.load(results["bianca_binary_path"]).get_fdata()

    overlap = compute_overlap(synthseg, bianca)
    results["overlap"] = overlap
    print(f"  Dice coefficient: {overlap['dice']:.3f}")
    print(f"  Jaccard index: {overlap['jaccard']:.3f}")
    print(f"  WMH-SynthSeg voxels: {overlap['mask1_voxels']}")
    print(f"  BIANCA voxels: {overlap['mask2_voxels']}")
    print(f"  Intersection: {overlap['intersection_voxels']}")

    # Volume comparison
    print("\n--- Volume Comparison ---")
    vol_ratio = (results["bianca_volume_mm3"] /
                 results["synthseg_volume_mm3"]
                 if results["synthseg_volume_mm3"] > 0 else float("inf"))
    print(f"  WMH-SynthSeg: {results['synthseg_volume_mm3']:.1f} mm³")
    print(f"  BIANCA:       {results['bianca_volume_mm3']:.1f} mm³")
    print(f"  Ratio (BIANCA/SynthSeg): {vol_ratio:.2f}")
    results["volume_ratio"] = float(vol_ratio)

    # Save results
    comp_dir = deriv / "bianca" / subject
    with open(comp_dir / "wmh_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {comp_dir / 'wmh_comparison.json'}")

    print("\n=== Summary ===")
    print("Note: WAND has no FLAIR — both methods use T1w only.")
    print("BIANCA trained on WMH-SynthSeg labels (self-training),")
    print("so agreement tests whether spatial features add information.")
    print("True validation requires manual expert labels.")


if __name__ == "__main__":
    main()
