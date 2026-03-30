"""Automatic ROI extraction from FreeSurfer segmentations.

Extracts tissue-specific statistics from quantitative parameter maps
using FreeSurfer aparc+aseg parcellations. Supports multi-tool
comparison for oracle-driven model development.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional

# FreeSurfer standard subcortical + tissue labels
# Full LUT at $FREESURFER_HOME/FreeSurferColorLUT.txt
TISSUE_LABELS = {
    "WM": [2, 41, 77, 251, 252, 253, 254, 255],  # L/R cerebral WM + WM hypointensities + CC
    "GM": list(range(1001, 1036)) + list(range(2001, 2036)),  # L/R cortical (Desikan)
    "CSF": [4, 5, 14, 15, 24, 43, 44],  # L/R lateral ventricles + 3rd/4th + CSF
}

SUBCORTICAL_LABELS = {
    "Left-Thalamus": 10, "Right-Thalamus": 49,
    "Left-Caudate": 11, "Right-Caudate": 50,
    "Left-Putamen": 12, "Right-Putamen": 51,
    "Left-Pallidum": 13, "Right-Pallidum": 52,
    "Left-Hippocampus": 17, "Right-Hippocampus": 53,
    "Left-Amygdala": 18, "Right-Amygdala": 54,
    "Left-Accumbens": 26, "Right-Accumbens": 58,
    "Brain-Stem": 16,
}

DESIKAN_LABELS = {
    1001: "lh-bankssts", 1002: "lh-caudalanteriorcingulate",
    1003: "lh-caudalmiddlefrontal", 1005: "lh-cuneus",
    1006: "lh-entorhinal", 1007: "lh-fusiform",
    1008: "lh-inferiorparietal", 1009: "lh-inferiortemporal",
    1010: "lh-isthmuscingulate", 1011: "lh-lateraloccipital",
    1012: "lh-lateralorbitofrontal", 1013: "lh-lingual",
    1014: "lh-medialorbitofrontal", 1015: "lh-middletemporal",
    1016: "lh-parahippocampal", 1017: "lh-paracentral",
    1018: "lh-parsopercularis", 1019: "lh-parsorbitalis",
    1020: "lh-parstriangularis", 1021: "lh-pericalcarine",
    1022: "lh-postcentral", 1023: "lh-posteriorcingulate",
    1024: "lh-precentral", 1025: "lh-precuneus",
    1026: "lh-rostralanteriorcingulate", 1027: "lh-rostralmiddlefrontal",
    1028: "lh-superiorfrontal", 1029: "lh-superiorparietal",
    1030: "lh-superiortemporal", 1031: "lh-supramarginal",
    1032: "lh-frontalpole", 1033: "lh-temporalpole",
    1034: "lh-transversetemporal", 1035: "lh-insula",
    # Right hemisphere: same names, IDs + 1000
    **{k + 1000: v.replace("lh-", "rh-") for k, v in {
        1001: "lh-bankssts", 1002: "lh-caudalanteriorcingulate",
        1003: "lh-caudalmiddlefrontal", 1005: "lh-cuneus",
        1006: "lh-entorhinal", 1007: "lh-fusiform",
        1008: "lh-inferiorparietal", 1009: "lh-inferiortemporal",
        1010: "lh-isthmuscingulate", 1011: "lh-lateraloccipital",
        1012: "lh-lateralorbitofrontal", 1013: "lh-lingual",
        1014: "lh-medialorbitofrontal", 1015: "lh-middletemporal",
        1016: "lh-parahippocampal", 1017: "lh-paracentral",
        1018: "lh-parsopercularis", 1019: "lh-parsorbitalis",
        1020: "lh-parstriangularis", 1021: "lh-pericalcarine",
        1022: "lh-postcentral", 1023: "lh-posteriorcingulate",
        1024: "lh-precentral", 1025: "lh-precuneus",
        1026: "lh-rostralanteriorcingulate", 1027: "lh-rostralmiddlefrontal",
        1028: "lh-superiorfrontal", 1029: "lh-superiorparietal",
        1030: "lh-superiortemporal", 1031: "lh-supramarginal",
        1032: "lh-frontalpole", 1033: "lh-temporalpole",
        1034: "lh-transversetemporal", 1035: "lh-insula",
    }.items()},
}

ALL_LABELS = {**{v: k for k, v in SUBCORTICAL_LABELS.items()},
              **{v: k for k, v in DESIKAN_LABELS.items()}}


def load_segmentation(path: str) -> np.ndarray:
    """Load FreeSurfer segmentation (aparc+aseg.mgz or aseg.mgz).

    Returns integer label array.
    """
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.int32)


def extract_roi_stats(param_map: np.ndarray, segmentation: np.ndarray,
                       labels: Optional[dict] = None,
                       valid_range: Optional[tuple] = None) -> list:
    """Extract per-ROI statistics from a parameter map.

    Args:
        param_map: 3D parameter map (T1, BPF, FA, etc.)
        segmentation: 3D integer label array (same shape or broadcastable)
        labels: Dict of {label_id: name}. Default: Desikan + subcortical.
        valid_range: (min, max) tuple to exclude outliers

    Returns:
        List of dicts with keys: name, label_id, mean, median, std, n_voxels
    """
    if labels is None:
        labels = {**DESIKAN_LABELS, **{v: k for k, v in SUBCORTICAL_LABELS.items()}}

    param = np.asarray(param_map)
    seg = np.asarray(segmentation)

    # Validate shapes match
    if param.shape != seg.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs seg {seg.shape}")

    results = []
    for label_id, name in sorted(labels.items(), key=lambda x: x[1]):
        mask = seg == label_id
        if mask.sum() == 0:
            continue

        vals = param[mask]
        valid = np.isfinite(vals) & (vals != 0)
        if valid_range is not None:
            valid &= (vals >= valid_range[0]) & (vals <= valid_range[1])
        vals = vals[valid]

        if len(vals) < 5:
            continue

        results.append({
            "name": name,
            "label_id": int(label_id),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "n_voxels": int(len(vals)),
        })

    return results


def extract_tissue_stats(param_map: np.ndarray,
                          segmentation: np.ndarray,
                          valid_range: Optional[tuple] = None) -> dict:
    """Extract WM, GM, CSF statistics from a parameter map.

    Args:
        param_map: 3D parameter map
        segmentation: 3D FreeSurfer aparc+aseg labels

    Returns:
        Dict with WM/GM/CSF keys, each containing mean/median/std/n_voxels.
    """
    param = np.asarray(param_map)
    seg = np.asarray(segmentation)
    results = {}

    for tissue, label_ids in TISSUE_LABELS.items():
        mask = np.isin(seg, label_ids)
        vals = param[mask]
        valid = np.isfinite(vals) & (vals != 0)
        if valid_range is not None:
            valid &= (vals >= valid_range[0]) & (vals <= valid_range[1])
        vals = vals[valid]

        if len(vals) < 10:
            results[tissue] = {"mean": None, "median": None, "std": None, "n_voxels": 0}
            continue

        results[tissue] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "n_voxels": int(len(vals)),
        }

    return results


def compare_tools(maps: dict, segmentation: np.ndarray,
                   valid_range: Optional[tuple] = None) -> dict:
    """Compare multiple tool outputs on the same ROIs.

    Args:
        maps: Dict of {tool_name: param_map_3d} e.g.
              {"QUIT": quit_t1, "FreeSurfer": fs_t1, "Python": py_t1}
        segmentation: FreeSurfer aparc+aseg
        valid_range: (min, max) for valid parameter values

    Returns:
        Dict of {tissue: {tool: {mean, median, std, n_voxels}}}
    """
    results = {}
    for tool_name, param_map in maps.items():
        tissue_stats = extract_tissue_stats(param_map, segmentation, valid_range)
        for tissue, stats in tissue_stats.items():
            if tissue not in results:
                results[tissue] = {}
            results[tissue][tool_name] = stats

    return results


def to_csv(stats: list, path: str):
    """Export ROI stats list to CSV.

    Args:
        stats: Output from extract_roi_stats()
        path: Output CSV path
    """
    if not stats:
        return

    keys = stats[0].keys()
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in stats:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def compare_tools_to_csv(comparison: dict, path: str):
    """Export multi-tool comparison to CSV.

    Args:
        comparison: Output from compare_tools()
        path: Output CSV path
    """
    rows = []
    for tissue, tools in comparison.items():
        for tool, stats in tools.items():
            if stats.get("mean") is not None:
                rows.append({
                    "tissue": tissue,
                    "tool": tool,
                    **stats,
                })

    if not rows:
        return

    keys = rows[0].keys()
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
