"""CHARM-GEMS head segmentation and conductivity assignment.

Wraps FreeSurfer SAMSEG with the extended CHARM atlas to produce
multi-tissue head models for FEM/BEM forward solutions. The CHARM
atlas provides 60 tissue labels including differentiated skull
(cortical bone, cancellous bone), vasculature (arteries, veins),
and extra-cerebral tissues (skin, mucosa, sinus, eye fluid).

Usage:
    seg = run_charm_segmentation(t1_path, output_dir)
    tissues = assign_conductivities(seg)
    surfaces = extract_tissue_surfaces(seg, tissues)

Requirements:
    - FreeSurfer 8.2+ with samseg+charm atlas
    - charm-gems Python bindings (optional, for direct GEMS API)
"""

import subprocess
import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# CHARM tissue labels and their conductivities (S/m)
# Based on literature values: Gabriel et al. (1996), Dannhauer et al. (2011),
# Saturnino et al. (2019, SimNIBS)
CHARM_TISSUE_CONDUCTIVITY = {
    # Label ID: (name, conductivity S/m)
    0: ('Unknown', 0.0),
    911: ('Skin', 0.465),
    907: ('Other-Tissues', 0.25),
    915: ('Bone-Cortical', 0.006),
    916: ('Bone-Cancellous', 0.025),     # 4× more conductive than cortical
    24: ('CSF', 1.654),
    914: ('Vein', 0.7),
    902: ('Artery', 0.7),
    126: ('Spinal-Cord', 0.126),
    262: ('Sinus', 1.0),                  # air-filled → ~0, fluid-filled → ~1
    909: ('Mucosa', 0.35),
    908: ('Rectus-Muscles', 0.35),
    259: ('Eye-Fluid', 1.5),
    930: ('Optic-Nerve', 0.126),
    # Standard brain structures
    2: ('Left-Cerebral-WM', 0.126),
    41: ('Right-Cerebral-WM', 0.126),
    3: ('Left-Cerebral-Cortex', 0.276),
    42: ('Right-Cerebral-Cortex', 0.276),
    7: ('Left-Cerebellum-WM', 0.126),
    46: ('Right-Cerebellum-WM', 0.126),
    8: ('Left-Cerebellum-Cortex', 0.276),
    47: ('Right-Cerebellum-Cortex', 0.276),
    16: ('Brain-Stem', 0.126),
    10: ('Left-Thalamus', 0.276),
    49: ('Right-Thalamus', 0.276),
    11: ('Left-Caudate', 0.276),
    50: ('Right-Caudate', 0.276),
    12: ('Left-Putamen', 0.276),
    51: ('Right-Putamen', 0.276),
    13: ('Left-Pallidum', 0.126),
    52: ('Right-Pallidum', 0.126),
    17: ('Left-Hippocampus', 0.276),
    53: ('Right-Hippocampus', 0.276),
    18: ('Left-Amygdala', 0.276),
    54: ('Right-Amygdala', 0.276),
    26: ('Left-Accumbens', 0.276),
    58: ('Right-Accumbens', 0.276),
    28: ('Left-VentralDC', 0.276),
    60: ('Right-VentralDC', 0.276),
    4: ('Left-Lateral-Ventricle', 1.654),
    43: ('Right-Lateral-Ventricle', 1.654),
    5: ('Left-Inf-Lat-Vent', 1.654),
    44: ('Right-Inf-Lat-Vent', 1.654),
    14: ('3rd-Ventricle', 1.654),
    15: ('4th-Ventricle', 1.654),
    192: ('Corpus-Callosum', 0.126),
    77: ('WM-hypointensities', 0.126),
    85: ('Optic-Chiasm', 0.126),
    30: ('Left-vessel', 0.7),
    62: ('Right-vessel', 0.7),
    31: ('Left-choroid-plexus', 1.0),
    63: ('Right-choroid-plexus', 1.0),
    34: ('Left-WMCrowns', 0.126),
    66: ('Right-WMCrowns', 0.126),
    183: ('Left-Vermis-Area', 0.276),
    184: ('Right-Vermis-Area', 0.276),
    267: ('Pons-Belly-Area', 0.126),
    11300: ('ctx_lh_high_myelin', 0.276),
    12300: ('ctx_rh_high_myelin', 0.276),
    80: ('non-WM-hypointensities', 0.276),
}

# Simplified 5-compartment conductivity model for BEM
FIVE_COMPARTMENT = {
    'brain': 0.33,           # GM + WM average
    'csf': 1.654,
    'bone_cortical': 0.006,  # compact bone
    'bone_cancellous': 0.025,  # spongy bone (diploe)
    'scalp': 0.465,
}

# Default CHARM atlas path in FreeSurfer 8.2+
DEFAULT_ATLAS = Path('/Applications/freesurfer/8.2.0/average/samseg/'
                     'samseg+cc+pons+verm+charm+wmcrowns')


def run_charm_segmentation(t1_path: str,
                           output_dir: str,
                           atlas_dir: Optional[str] = None,
                           threads: int = 4,
                           timeout: int = 3600) -> str:
    """Run SAMSEG with CHARM atlas for multi-tissue head segmentation.

    Args:
        t1_path: path to T1-weighted image (NIfTI or MGZ)
        output_dir: directory for output segmentation
        atlas_dir: CHARM atlas directory (default: FS 8.2 built-in)
        threads: number of CPU threads
        timeout: max seconds to wait

    Returns:
        path to output segmentation volume (seg.mgz)

    Raises:
        RuntimeError: if SAMSEG fails
        FileNotFoundError: if atlas or input not found
    """
    atlas = Path(atlas_dir) if atlas_dir else DEFAULT_ATLAS
    if not atlas.exists():
        raise FileNotFoundError(f"CHARM atlas not found: {atlas}")
    if not os.path.exists(t1_path):
        raise FileNotFoundError(f"T1 image not found: {t1_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Estimate memory: SAMSEG typically uses 4-8 GB
    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    if avail_gb < 4.0:
        logger.warning(f"SAMSEG needs ~4-8 GB, only {avail_gb:.1f} GB available")

    cmd = [
        'run_samseg',
        '-i', str(t1_path),
        '-o', str(output_dir),
        '--atlas', str(atlas),
        '--threads', str(threads),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"SAMSEG failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )

    seg_path = os.path.join(output_dir, 'seg.mgz')
    if not os.path.exists(seg_path):
        raise RuntimeError(f"Expected output not found: {seg_path}")

    logger.info(f"Segmentation complete: {seg_path}")
    return seg_path


def load_segmentation(seg_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CHARM segmentation volume.

    Args:
        seg_path: path to seg.mgz

    Returns:
        (labels, affine): integer label volume and voxel-to-world transform
    """
    import nibabel as nib
    img = nib.load(seg_path)
    labels = np.asarray(img.dataobj).astype(np.int32)
    affine = img.affine
    return labels, affine


def assign_conductivities(labels: np.ndarray,
                          conductivity_map: Optional[Dict] = None
                          ) -> np.ndarray:
    """Convert label volume to conductivity volume.

    Args:
        labels: (x, y, z) integer label array from CHARM segmentation
        conductivity_map: optional override {label_id: conductivity_Sm}

    Returns:
        (x, y, z) float32 conductivity array in S/m
    """
    cmap = conductivity_map or {k: v[1] for k, v in CHARM_TISSUE_CONDUCTIVITY.items()}
    sigma = np.zeros(labels.shape, dtype=np.float32)
    for label_id, conductivity in cmap.items():
        sigma[labels == label_id] = conductivity
    return sigma


def segmentation_summary(labels: np.ndarray) -> Dict:
    """Summarise tissue volumes from a CHARM segmentation.

    Returns:
        dict with tissue name, voxel count, and volume (mL) per label
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    summary = {}
    for label_id, count in zip(unique_labels, counts):
        label_id = int(label_id)
        name = CHARM_TISSUE_CONDUCTIVITY.get(label_id, ('Unknown', 0.0))[0]
        sigma = CHARM_TISSUE_CONDUCTIVITY.get(label_id, ('Unknown', 0.0))[1]
        summary[label_id] = {
            'name': name,
            'voxels': int(count),
            'volume_mL': round(int(count) * 1.0, 1),  # assumes 1mm³ voxels
            'conductivity_Sm': sigma,
        }
    return summary


def extract_tissue_surfaces(labels: np.ndarray,
                            affine: np.ndarray,
                            tissue_groups: Optional[Dict] = None
                            ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract isosurfaces at tissue boundaries using marching cubes.

    Args:
        labels: (x, y, z) integer label array
        affine: voxel-to-world transform
        tissue_groups: dict mapping surface name to list of label IDs
            that define the "inside". Default groups: brain, skull, scalp.

    Returns:
        dict of {surface_name: (vertices_mm, faces)}
    """
    from skimage.measure import marching_cubes

    if tissue_groups is None:
        # Default 3-surface BEM model
        brain_labels = [2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
                        41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
                        192, 34, 66, 77, 85, 183, 184, 267, 126, 930,
                        11300, 12300, 80]
        csf_labels = [4, 5, 14, 15, 24, 43, 44, 31, 63]
        bone_labels = [915, 916]
        skin_labels = [911]

        tissue_groups = {
            'inner_skull': brain_labels + csf_labels,
            'outer_skull': brain_labels + csf_labels + bone_labels,
            'outer_skin': brain_labels + csf_labels + bone_labels + skin_labels,
        }

    surfaces = {}
    voxel_size = np.abs(np.diag(affine[:3, :3]))

    for name, inside_labels in tissue_groups.items():
        mask = np.isin(labels, inside_labels).astype(np.float32)

        if mask.sum() < 100:
            logger.warning(f"Surface '{name}': too few voxels ({mask.sum()})")
            continue

        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5,
                                                 spacing=tuple(voxel_size))
            # Transform to world coordinates
            verts_world = verts + affine[:3, 3]
            surfaces[name] = (verts_world.astype(np.float32),
                              faces.astype(np.int32))
            logger.info(f"Surface '{name}': {len(verts)} verts, {len(faces)} faces")
        except Exception as e:
            logger.warning(f"Surface '{name}' extraction failed: {e}")

    return surfaces
