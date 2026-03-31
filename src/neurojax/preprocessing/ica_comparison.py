"""Multi-oracle ICA comparison for MEG preprocessing.

Compares four independent ICA implementations:
  1. neurojax PICA (JAX GPU) — our implementation
  2. MNE ICA (Picard/FastICA) — standard CPU reference
  3. FSL MELODIC (via NIfTI bridge) — probabilistic ICA, gold-standard dim. est.
  4. neurojax Complex FastICA (JAX GPU) — for oscillatory sources

The key validation:
  - Do all oracles find the same components?
  - Do dimensionality estimates agree? (neurojax Laplace vs MELODIC Bayesian)
  - Does the PCA rank we use for TDE+PCA (80) match the data-driven estimate?

The MELODIC-as-NIfTI trick: reshape MEG (n_channels, n_times) → NIfTI
(n_channels, 1, 1, n_times). MELODIC treats channels as voxels and time as
volumes — its Bayesian model selection gives the intrinsic dimensionality.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DimensionalityEstimate:
    """Dimensionality estimate from one method."""
    method: str
    n_components: int
    evidence: Optional[float] = None  # log-evidence or similar score


@dataclass
class ICAComparisonResult:
    """Results from multi-oracle ICA comparison."""
    subject: str
    n_channels: int
    n_times: int
    sfreq: float

    # Dimensionality estimates
    dim_neurojax_laplace: int = 0
    dim_melodic_bayesian: int = 0
    dim_mne_default: int = 0

    # Component correlation between oracles
    neurojax_mne_corr: float = 0.0  # mean abs correlation of matched components
    neurojax_melodic_corr: float = 0.0
    mne_melodic_corr: float = 0.0

    # Artifact component counts
    mne_ecg_components: int = 0
    mne_eog_components: int = 0


# -------------------------------------------------------------------------
# MELODIC via NIfTI bridge
# -------------------------------------------------------------------------

def meg_to_nifti(data: np.ndarray, sfreq: float, out_path: str) -> None:
    """Save MEG data as NIfTI for MELODIC processing.

    Reshapes (n_channels, n_times) → (n_channels, 1, 1, n_times) NIfTI.
    MELODIC treats the first 3 dims as spatial and the 4th as temporal.

    Parameters
    ----------
    data : (n_channels, n_times) array
    sfreq : float — sampling frequency (stored as pixdim[4] = TR = 1/sfreq)
    out_path : str — output .nii.gz path
    """
    import nibabel as nib
    n_ch, n_t = data.shape
    # Reshape to 4D: (n_channels, 1, 1, n_times)
    img_data = data[:, np.newaxis, np.newaxis, :].astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    # Set TR = 1/sfreq so MELODIC knows the temporal resolution
    img.header["pixdim"][4] = 1.0 / sfreq
    nib.save(img, out_path)
    logger.info("Saved MEG as NIfTI: %s (%d channels × %d timepoints)", out_path, n_ch, n_t)


def run_melodic(
    nifti_path: str,
    output_dir: str,
    dim: int = 0,  # 0 = automatic dimensionality estimation
    approach: str = "concat",
) -> dict:
    """Run FSL MELODIC on a NIfTI file.

    Parameters
    ----------
    nifti_path : str
    output_dir : str
    dim : int — number of components (0 = auto-estimate via Bayesian model selection)
    approach : str — 'concat' or 'tica'

    Returns
    -------
    dict with: n_components, mixing_matrix, component_timecourses
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "melodic",
        "-i", nifti_path,
        "-o", output_dir,
        "--nobet",  # no brain extraction (it's already channel data)
        "--nomask",  # no masking
        "--tr", str(1.0),  # will be overridden by NIfTI header
        "-a", approach,
        "--Oall",  # output all files
        "--report",
    ]
    if dim > 0:
        cmd.extend(["-d", str(dim)])
    # else: MELODIC auto-estimates dimensionality

    logger.info("Running MELODIC: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("MELODIC failed: %s", result.stderr[-500:])
        return {"n_components": 0, "error": result.stderr[-500:]}

    # Parse outputs
    output = {"n_components": 0}

    # Mixing matrix: melodic_mix (n_times × n_components)
    mix_path = os.path.join(output_dir, "melodic_mix")
    if os.path.exists(mix_path):
        mix = np.loadtxt(mix_path)
        output["mixing_matrix"] = mix
        output["n_components"] = mix.shape[1]
        logger.info("MELODIC: %d components estimated", mix.shape[1])

    # IC spatial maps (in our case: channel loadings)
    ic_path = os.path.join(output_dir, "melodic_IC.nii.gz")
    if os.path.exists(ic_path):
        import nibabel as nib
        ic_img = nib.load(ic_path)
        output["ic_maps"] = ic_img.get_fdata().squeeze()  # (n_channels, n_components)

    return output


# -------------------------------------------------------------------------
# neurojax PICA dimensionality
# -------------------------------------------------------------------------

def estimate_dim_neurojax(data: np.ndarray) -> DimensionalityEstimate:
    """Estimate intrinsic dimensionality using neurojax Laplace approximation.

    Parameters
    ----------
    data : (n_channels, n_times) array

    Returns
    -------
    DimensionalityEstimate
    """
    import jax.numpy as jnp
    from neurojax.analysis.dimensionality import PPCA

    X = jnp.array(data, dtype=jnp.float32)
    evidence = PPCA.get_laplace_evidence(X)
    k = int(jnp.argmax(evidence)) + 1  # 1-indexed
    logger.info("neurojax Laplace: k=%d (max evidence=%.1f)", k, float(evidence[k - 1]))

    return DimensionalityEstimate(
        method="neurojax_laplace",
        n_components=k,
        evidence=float(evidence[k - 1]),
    )


# -------------------------------------------------------------------------
# MNE ICA
# -------------------------------------------------------------------------

def run_mne_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    method: str = "picard",
    max_iter: int = 500,
) -> tuple[mne.preprocessing.ICA, dict]:
    """Run MNE ICA with automatic ECG/EOG component detection.

    Parameters
    ----------
    raw : mne.io.Raw
    n_components : int or None (None = 0.999 variance explained)
    method : str — 'picard', 'fastica', or 'infomax'

    Returns
    -------
    ica : mne.preprocessing.ICA — fitted ICA object
    info : dict — n_components, ecg_indices, eog_indices
    """
    if n_components is None:
        # Let MNE decide based on explained variance
        n_components = 0.999

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=42,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    info = {
        "n_components": ica.n_components_,
        "method": method,
    }

    # Auto-detect artifact components
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
        info["ecg_indices"] = ecg_indices
        info["ecg_scores"] = ecg_scores.tolist() if hasattr(ecg_scores, 'tolist') else []
    except Exception:
        info["ecg_indices"] = []
        info["ecg_scores"] = []

    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
        info["eog_indices"] = eog_indices
        info["eog_scores"] = eog_scores.tolist() if hasattr(eog_scores, 'tolist') else []
    except Exception:
        info["eog_indices"] = []
        info["eog_scores"] = []

    logger.info("MNE ICA (%s): %d components, %d ECG, %d EOG",
                method, ica.n_components_, len(info["ecg_indices"]), len(info["eog_indices"]))

    return ica, info


# -------------------------------------------------------------------------
# Component matching across oracles
# -------------------------------------------------------------------------

def match_components(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """Match ICA components between two decompositions via correlation.

    Parameters
    ----------
    A : (n_channels, n_components_a) — mixing matrix from oracle A
    B : (n_channels, n_components_b) — mixing matrix from oracle B

    Returns
    -------
    mean_corr : float — mean absolute correlation of best-matched pairs
    matching : (min(na, nb), 2) — index pairs (a_idx, b_idx)
    """
    n_a = A.shape[1]
    n_b = B.shape[1]
    n_match = min(n_a, n_b)

    # Correlation matrix between all pairs
    # Normalize columns
    A_norm = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-10)
    corr = np.abs(A_norm.T @ B_norm)  # (n_a, n_b)

    # Greedy matching
    matching = []
    used_b = set()
    for _ in range(n_match):
        # Find best unmatched pair
        mask = np.ones_like(corr, dtype=bool)
        for _, b in matching:
            mask[:, b] = False
        masked_corr = corr * mask
        idx = np.unravel_index(np.argmax(masked_corr), corr.shape)
        matching.append(idx)

    mean_corr = np.mean([corr[a, b] for a, b in matching])
    return mean_corr, np.array(matching)


# -------------------------------------------------------------------------
# Full comparison
# -------------------------------------------------------------------------

def run_ica_comparison(
    raw: mne.io.Raw,
    subject: str,
    use_melodic: bool = True,
    fsl_dir: Optional[str] = None,
) -> ICAComparisonResult:
    """Run multi-oracle ICA comparison on MEG data.

    Parameters
    ----------
    raw : mne.io.Raw — preprocessed MEG data (filtered, resampled)
    subject : str
    use_melodic : bool — whether to run FSL MELODIC (requires FSL installed)
    fsl_dir : str — path to FSL installation

    Returns
    -------
    ICAComparisonResult
    """
    raw_meg = raw.copy().pick(picks="meg", exclude="bads")
    data = raw_meg.get_data()  # (n_channels, n_times)
    n_ch, n_t = data.shape
    sfreq = raw_meg.info["sfreq"]

    result = ICAComparisonResult(
        subject=subject,
        n_channels=n_ch,
        n_times=n_t,
        sfreq=sfreq,
    )

    # 1. neurojax dimensionality estimation
    logger.info("=== neurojax Laplace dimensionality ===")
    try:
        dim_nj = estimate_dim_neurojax(data)
        result.dim_neurojax_laplace = dim_nj.n_components
    except Exception as e:
        logger.warning("neurojax dim est failed: %s", e)

    # 2. MELODIC dimensionality estimation
    if use_melodic:
        logger.info("=== MELODIC (via NIfTI bridge) ===")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                nifti_path = os.path.join(tmpdir, "meg_data.nii.gz")
                melodic_dir = os.path.join(tmpdir, "melodic_out")
                meg_to_nifti(data, sfreq, nifti_path)
                mel_result = run_melodic(nifti_path, melodic_dir, dim=0)
                result.dim_melodic_bayesian = mel_result.get("n_components", 0)
        except Exception as e:
            logger.warning("MELODIC failed: %s", e)

    # 3. MNE ICA
    logger.info("=== MNE ICA (Picard) ===")
    try:
        ica_mne, ica_info = run_mne_ica(raw_meg)
        result.dim_mne_default = ica_info["n_components"]
        result.mne_ecg_components = len(ica_info.get("ecg_indices", []))
        result.mne_eog_components = len(ica_info.get("eog_indices", []))
    except Exception as e:
        logger.warning("MNE ICA failed: %s", e)

    # 4. Cross-oracle component matching (if we have mixing matrices)
    # TODO: extract mixing matrices from all oracles and run match_components

    logger.info("=== Dimensionality comparison ===")
    logger.info("  neurojax Laplace: %d", result.dim_neurojax_laplace)
    logger.info("  MELODIC Bayesian: %d", result.dim_melodic_bayesian)
    logger.info("  MNE (0.999 var):  %d", result.dim_mne_default)
    logger.info("  MNE ECG comps:    %d", result.mne_ecg_components)
    logger.info("  MNE EOG comps:    %d", result.mne_eog_components)

    return result
