#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
# ]
# ///
"""Automated ROI analysis across all WAND modalities.

Produces paper-ready CSV tables:
  - Three-way T1 comparison (QUIT vs FreeSurfer vs Python)
  - Per-Desikan-ROI T1 statistics
  - QUIT QMT tissue-specific parameters
  - Perfusion summary
  - DTI summary

Usage:
    uv run 41_auto_roi_analysis.py sub-08033
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add neurojax to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from neurojax.qmri.io import load_nifti
from neurojax.qmri.roi import (
    extract_roi_stats, extract_tissue_stats,
    compare_tools, compare_tools_to_csv, to_csv,
    load_segmentation, DESIKAN_LABELS
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", default="sub-08033", nargs="?")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    args = parser.parse_args()

    wand = Path(args.wand_root)
    sub = args.subject
    qmri = wand / "derivatives" / "qmri" / sub / "ses-02"
    out = qmri / "roi_analysis"
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== Auto ROI Analysis: {sub} ===")

    # Load segmentation
    seg_path = qmri / "aparc_aseg_spgr.nii.gz"
    if not seg_path.exists():
        print(f"ERROR: segmentation not found at {seg_path}")
        print("Run mri_convert to create it first.")
        return
    seg = load_segmentation(str(seg_path))

    # 1. Three-way T1
    t1_maps = {}
    for name, rel_path in [
        ("QUIT_DESPOT1", "quit/D1_D1_T1.nii.gz"),
        ("FreeSurfer", "freesurfer/T1_in_spgr.nii.gz"),
        ("Python_linear", "T1map.nii.gz"),
    ]:
        path = qmri / rel_path
        if path.exists():
            data, _, _ = load_nifti(str(path))
            arr = np.asarray(data)
            if "FreeSurfer" in name and np.median(arr[arr > 100]) > 100:
                arr = arr / 1000.0
            t1_maps[name] = arr

    if len(t1_maps) >= 2:
        comparison = compare_tools(t1_maps, seg, valid_range=(0.1, 5.0))
        compare_tools_to_csv(comparison, str(out / "t1_three_way.csv"))
        print(f"  t1_three_way.csv: {len(t1_maps)} tools × {len(comparison)} tissues")

    # 2. Per-ROI T1
    if "QUIT_DESPOT1" in t1_maps:
        roi_stats = extract_roi_stats(
            t1_maps["QUIT_DESPOT1"], seg,
            labels=DESIKAN_LABELS, valid_range=(0.1, 5.0)
        )
        to_csv(roi_stats, str(out / "quit_t1_per_roi.csv"))
        print(f"  quit_t1_per_roi.csv: {len(roi_stats)} ROIs")

    # 3. QMT tissue stats
    bpf_path = qmri / "quit" / "QMT_QMT_f_b.nii.gz"
    t1q_path = qmri / "quit" / "D1_T1_in_QMT.nii.gz"
    if bpf_path.exists() and t1q_path.exists():
        bpf = np.asarray(load_nifti(str(bpf_path))[0])
        t1q = np.asarray(load_nifti(str(t1q_path))[0])
        wm = (t1q > 0.3) & (t1q < 1.0) & (bpf > 0.01)
        gm = (t1q >= 1.0) & (t1q < 2.5) & (bpf > 0.01)

        params = {
            "BPF": ("quit/QMT_QMT_f_b.nii.gz", (0.01, 0.30), 1),
            "k_bf": ("quit/QMT_QMT_k_bf.nii.gz", (0.01, 20), 1),
            "T1_f": ("quit/QMT_QMT_T1_f.nii.gz", (0.1, 5.0), 1),
            "T2_f": ("quit/QMT_QMT_T2_f.nii.gz", (0.005, 1.0), 1000),
            "T2_b": ("quit/QMT_QMT_T2_b.nii.gz", (1e-7, 5e-5), 1e6),
        }

        rows = []
        for name, (rel, vrange, scale) in params.items():
            path = qmri / rel
            if path.exists():
                d = np.asarray(load_nifti(str(path))[0])
                wm_v = d[wm & (d > vrange[0]) & (d < vrange[1])] * scale
                gm_v = d[gm & (d > vrange[0]) & (d < vrange[1])] * scale
                if len(wm_v) > 100:
                    rows.append({
                        "param": name,
                        "WM_median": f"{np.median(wm_v):.4g}",
                        "GM_median": f"{np.median(gm_v):.4g}",
                        "WM_n": str(len(wm_v)),
                        "GM_n": str(len(gm_v)),
                    })

        if rows:
            with open(out / "quit_qmt_tissue.csv", "w") as f:
                f.write(",".join(rows[0].keys()) + "\n")
                for r in rows:
                    f.write(",".join(r.values()) + "\n")
            print(f"  quit_qmt_tissue.csv: {len(rows)} parameters")

    print(f"\n  Output: {out}/")
    print(f"  Files: {sorted(os.listdir(out))}")


if __name__ == "__main__":
    main()
