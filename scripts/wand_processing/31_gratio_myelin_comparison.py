#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
# ]
# ///
"""WAND myelin proxy comparison: T1w/T2w vs QMT BPF vs T2* on cortical surface.

Computes:
  1. g-ratio map from QMT BPF (proxy for myelin volume fraction)
  2. Vertex-wise correlation: T1w/T2w vs QMT BPF on cortical surface
  3. Regional analysis: mean per Desikan ROI
  4. Conduction velocity proxy from g-ratio

Validates the T1w/T2w myelin assumption used in Valdes-Sosa ξ-αNET.

Outputs to: derivatives/qmri/{subject}/gratio/ and derivatives/qmri/{subject}/myelin_comparison/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Desikan-Killiany ROI labels (FreeSurfer aparc)
DESIKAN_ROIS = {
    1: "bankssts", 2: "caudalanteriorcingulate", 3: "caudalmiddlefrontal",
    5: "cuneus", 6: "entorhinal", 7: "fusiform", 8: "inferiorparietal",
    9: "inferiortemporal", 10: "isthmuscingulate", 11: "lateraloccipital",
    12: "lateralorbitofrontal", 13: "lingual", 14: "medialorbitofrontal",
    15: "middletemporal", 16: "parahippocampal", 17: "paracentral",
    18: "parsopercularis", 19: "parsorbitalis", 20: "parstriangularis",
    21: "pericalcarine", 22: "postcentral", 23: "posteriorcingulate",
    24: "precentral", 25: "precuneus", 26: "rostralanteriorcingulate",
    27: "rostralmiddlefrontal", 28: "superiorfrontal", 29: "superiorparietal",
    30: "superiortemporal", 31: "supramarginal", 32: "frontalpole",
    33: "temporalpole", 34: "transversetemporal", 35: "insula",
}


def load_surface_data(mgz_path):
    """Load FreeSurfer surface overlay (.mgz or .mgh)."""
    img = nib.load(str(mgz_path))
    return img.get_fdata().squeeze()


def project_vol_to_surface(vol_path, subject, subjects_dir, hemi,
                            proj_frac_start=0.2, proj_frac_end=0.8,
                            proj_frac_step=0.1):
    """Project volume to cortical surface using mri_vol2surf."""
    import subprocess
    import tempfile

    out_file = tempfile.NamedTemporaryFile(suffix=".mgz", delete=False)
    out_path = out_file.name
    out_file.close()

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = subjects_dir
    fs_home = env.get("FREESURFER_HOME", "/Applications/freesurfer/8.2.0")
    env["FREESURFER_HOME"] = fs_home

    cmd = [
        f"{fs_home}/bin/mri_vol2surf",
        "--mov", str(vol_path),
        "--regheader", subject,
        "--hemi", hemi,
        "--projfrac-avg",
        str(proj_frac_start), str(proj_frac_end), str(proj_frac_step),
        "--o", out_path,
        "--cortex",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  WARNING: mri_vol2surf failed for {hemi}: {result.stderr[-200:]}")
        return None

    data = load_surface_data(out_path)
    os.unlink(out_path)
    return data


def load_annot(annot_path):
    """Load FreeSurfer annotation file."""
    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
    return labels, names


def compute_roi_means(surf_data, annot_labels, roi_names):
    """Compute mean of surface data per ROI."""
    results = {}
    for label_idx in range(len(roi_names)):
        mask = annot_labels == label_idx
        if mask.sum() > 10 and label_idx > 0:  # skip unknown
            name = roi_names[label_idx].decode() if isinstance(
                roi_names[label_idx], bytes) else roi_names[label_idx]
            vals = surf_data[mask]
            valid = vals[vals != 0]
            if len(valid) > 5:
                results[name] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "n_vertices": int(len(valid)),
                }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="WAND myelin proxy comparison"
    )
    parser.add_argument("subject", help="Subject ID")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    args = parser.parse_args()

    subject = args.subject
    wand = Path(args.wand_root)
    deriv = wand / "derivatives"
    fs_dir = deriv / "freesurfer" / subject
    adv_dir = deriv / "advanced-freesurfer" / subject
    qmri_dir = deriv / "qmri" / subject
    gratio_dir = qmri_dir / "gratio"
    comp_dir = qmri_dir / "myelin_comparison"
    subjects_dir = str(deriv / "freesurfer")

    gratio_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    print("=== Myelin Proxy Comparison ===")
    print(f"Subject: {subject}")

    # --- 1. Load available maps ---
    maps = {}

    # T1w/T2w ratio (already computed in advanced-freesurfer)
    t1t2_path = adv_dir / "myelin" / "T1w_T2w_ratio.nii.gz"
    if t1t2_path.exists():
        maps["T1w_T2w"] = nib.load(str(t1t2_path))
        print(f"  T1w/T2w ratio: {t1t2_path}")

    # QMT BPF
    bpf_path = qmri_dir / "ses-02" / "QMT_bpf.nii.gz"
    if bpf_path.exists():
        maps["QMT_BPF"] = nib.load(str(bpf_path))
        print(f"  QMT BPF: {bpf_path}")

    # VFA T1
    t1_path = qmri_dir / "ses-02" / "T1map.nii.gz"
    if t1_path.exists():
        maps["VFA_T1"] = nib.load(str(t1_path))
        print(f"  VFA T1: {t1_path}")

    # T2*
    t2star_path = qmri_dir / "ses-06" / "T2star_map.nii.gz"
    if t2star_path.exists():
        maps["T2star"] = nib.load(str(t2star_path))
        print(f"  T2*: {t2star_path}")

    # R2*
    r2star_path = qmri_dir / "ses-06" / "R2star_map.nii.gz"
    if r2star_path.exists():
        maps["R2star"] = nib.load(str(r2star_path))
        print(f"  R2*: {r2star_path}")

    # --- 2. Compute g-ratio from QMT BPF ---
    print("\n--- g-ratio Computation ---")
    if "QMT_BPF" in maps:
        bpf_data = maps["QMT_BPF"].get_fdata().astype(np.float64)
        bpf_affine = maps["QMT_BPF"].affine

        # MVF ≈ BPF * calibration_factor
        # Sled & Pike (2001): BPF relates to bound proton fraction
        # MVF/BPF ratio ~1.5-2.5 depending on calibration
        # Using Stikov et al. (2015) calibration: MVF ≈ BPF * 1.86
        mvf = np.clip(bpf_data * 1.86, 0, 0.50)

        # g-ratio = sqrt(1 - MVF)
        g_ratio = np.zeros_like(mvf)
        valid = mvf > 0.01
        g_ratio[valid] = np.sqrt(1 - mvf[valid])

        # Clip to physiological range [0.5, 1.0]
        g_ratio[valid] = np.clip(g_ratio[valid], 0.5, 1.0)

        nib.save(nib.Nifti1Image(g_ratio, bpf_affine),
                 str(gratio_dir / "g_ratio_proxy.nii.gz"))
        nib.save(nib.Nifti1Image(mvf, bpf_affine),
                 str(gratio_dir / "mvf.nii.gz"))

        g_valid = g_ratio[valid]
        print(f"  g-ratio range: [{g_valid.min():.3f}, {g_valid.max():.3f}]")
        print(f"  Median g-ratio: {np.median(g_valid):.3f}")
        print(f"  Expected WM g-ratio: ~0.6-0.7 (Stikov 2015)")
        print(f"  Saved: {gratio_dir / 'g_ratio_proxy.nii.gz'}")

        # Conduction velocity proxy: v = k * d / g
        # where d = axon diameter (from AxCaliber, not available yet)
        # For now, save MVF which is the myelin volume fraction
    else:
        print("  QMT BPF not available, skipping g-ratio")

    # --- 3. Surface projection + correlation ---
    print("\n--- Cortical Surface Projection ---")

    # Check FreeSurfer surfaces exist
    lh_white = fs_dir / "surf" / "lh.white"
    if not lh_white.exists():
        print("  FreeSurfer surfaces not found, skipping surface analysis")
        return

    surface_maps = {}
    for name, vol_img in maps.items():
        vol_path = vol_img.get_filename()
        print(f"  Projecting {name}...")
        for hemi in ["lh", "rh"]:
            surf_data = project_vol_to_surface(
                vol_path, subject, subjects_dir, hemi
            )
            if surf_data is not None:
                key = f"{hemi}_{name}"
                surface_maps[key] = surf_data
                # Save surface overlay
                out_mgz = comp_dir / f"{hemi}.{name}.mgz"
                # Create a minimal overlay image
                nib.save(
                    nib.MGHImage(
                        surf_data.reshape(-1, 1, 1).astype(np.float32),
                        np.eye(4)
                    ),
                    str(out_mgz)
                )

    # --- 4. Vertex-wise correlations ---
    print("\n--- Vertex-wise Correlations ---")
    results = {"subject": subject, "correlations": {}}

    ref_key = "T1w_T2w"
    compare_keys = ["QMT_BPF", "VFA_T1", "T2star", "R2star"]

    for hemi in ["lh", "rh"]:
        ref_name = f"{hemi}_{ref_key}"
        if ref_name not in surface_maps:
            continue

        ref_data = surface_maps[ref_name]

        for comp_key in compare_keys:
            comp_name = f"{hemi}_{comp_key}"
            if comp_name not in surface_maps:
                continue

            comp_data = surface_maps[comp_name]

            # Only correlate where both are valid
            valid = (ref_data != 0) & (comp_data != 0) & np.isfinite(ref_data) & np.isfinite(comp_data)
            if valid.sum() < 100:
                continue

            r_pearson, p_pearson = pearsonr(ref_data[valid], comp_data[valid])
            r_spearman, p_spearman = spearmanr(ref_data[valid], comp_data[valid])

            pair = f"{hemi}: T1w/T2w vs {comp_key}"
            results["correlations"][pair] = {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
                "n_vertices": int(valid.sum()),
            }
            print(f"  {pair}: r={r_pearson:.3f} (p={p_pearson:.2e}), "
                  f"rho={r_spearman:.3f}, n={valid.sum()}")

    # --- 5. Regional analysis (Desikan ROIs) ---
    print("\n--- Regional Analysis (Desikan ROIs) ---")
    roi_results = {}

    for hemi in ["lh", "rh"]:
        annot_path = fs_dir / "label" / f"{hemi}.aparc.annot"
        if not annot_path.exists():
            continue

        labels, names = load_annot(annot_path)

        for map_name in [ref_key] + compare_keys:
            key = f"{hemi}_{map_name}"
            if key not in surface_maps:
                continue
            roi_means = compute_roi_means(surface_maps[key], labels, names)
            roi_results[key] = roi_means

    # Cross-modal ROI correlation
    if roi_results:
        print("\n  ROI-level correlations (T1w/T2w vs others):")
        for hemi in ["lh", "rh"]:
            ref_rois = roi_results.get(f"{hemi}_{ref_key}", {})
            if not ref_rois:
                continue
            for comp_key in compare_keys:
                comp_rois = roi_results.get(f"{hemi}_{comp_key}", {})
                if not comp_rois:
                    continue
                # Match ROIs present in both
                common = set(ref_rois.keys()) & set(comp_rois.keys())
                if len(common) < 5:
                    continue
                ref_vals = [ref_rois[r]["mean"] for r in sorted(common)]
                comp_vals = [comp_rois[r]["mean"] for r in sorted(common)]
                r, p = pearsonr(ref_vals, comp_vals)
                print(f"    {hemi}: T1w/T2w vs {comp_key}: "
                      f"r={r:.3f} (p={p:.3e}, {len(common)} ROIs)")
                results["correlations"][f"{hemi}_ROI: T1w/T2w vs {comp_key}"] = {
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "n_rois": len(common),
                }

    # Save results
    results["roi_data"] = roi_results
    with open(comp_dir / "myelin_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {comp_dir / 'myelin_comparison.json'}")
    print(f"Surface overlays: {comp_dir}/*.mgz")

    # --- Summary ---
    print("\n=== Summary ===")
    print("Key question: How well does T1w/T2w proxy correlate with QMT BPF?")
    if "lh: T1w/T2w vs QMT_BPF" in results["correlations"]:
        r = results["correlations"]["lh: T1w/T2w vs QMT_BPF"]["pearson_r"]
        print(f"  LH vertex-wise: r = {r:.3f}")
    if "rh: T1w/T2w vs QMT_BPF" in results["correlations"]:
        r = results["correlations"]["rh: T1w/T2w vs QMT_BPF"]["pearson_r"]
        print(f"  RH vertex-wise: r = {r:.3f}")
    print("\nExpected: moderate-strong correlation (r~0.5-0.8) with regional")
    print("bias in high-iron areas (basal ganglia) where T2* confounds T1w/T2w.")


if __name__ == "__main__":
    main()
