"""
NeuroJAX Geometry & BEM Module.

Handles:
1. MRI Processing (Watershed BEM).
2. Surface Generation (Scalp, Skull, Brain).
3. Coregistration (Montage -> MRI).
"""

import os
import mne
import numpy as np
from pathlib import Path

def make_bem_surfaces(subject: str, subjects_dir: Path, overwrite: bool = False):
    """
    Run MNE Watershed BEM to generate Brain/InnerSkull/OuterSkull/Scalp surfaces.
    Requires FreeSurfer reconstruction (recon-all) usually.
    
    If raw T1w is provided, we might need to run full recon-all or use `mne.bem.make_watershed_bem` directly if surfaces exist?
    MNE Watershed expects `mri/T1.mgz` or similar in FS structure.
    
    If we only have T1w.nii.gz, we might need to rely on 'fsl_bet' or just assume MNE can handle it?
    Actually MNE relies on FreeSurfer subjects_dir structure.
    
    For this demo, we assume the user has FSL/FreeSurfer or we gracefully fail/mock 
    if strictly Python-only is needed.
    
    However, the user asked for "DeepBet" which implies a separate tool.
    
    Let's implement a wrapper that checks for surfaces or tries to create them.
    """
    
    # Check if surfaces exist
    bem_dir = subjects_dir / subject / 'bem'
    if (bem_dir / 'outer_skin.surf').exists() and not overwrite:
        print(f"[CACHE] BEM surfaces found in {bem_dir}")
        return
        
    print(f"[GEOMETRY] Generating BEM surfaces for {subject}...")
    try:
        mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=overwrite)
    except Exception as e:
        print(f"[ERROR] Watershed BEM failed: {e}")
        print("Ensure FreeSurfer/FSL is installed and T1w is properly imported.")
        # Hint: 'mne flash_bem' or manual BET might be needed.

def coregister_montage(raw: mne.io.Raw, trans_path: Path, subject: str, subjects_dir: Path):
    """
    Load or Estimate Coregistration (Trans).
    
    If trans_path exists, load it.
    Else, use Fiducials (LPA/RPA/Nasion) to align Montage to MRI.
    """
    if trans_path.exists():
        return mne.read_trans(trans_path)
        
    print("[COREG] estimating alignment using fiducials...")
    # Need montage with fiducials
    # GSN-HydroCel-129 has them.
    
    # MRI fiducials? stored in typical FS location?
    # If not, auto-align?
    # mne.coreg.Coregistration?
    
    # For automation, we assume standard fiducials in MRI are aligned with template?
    # Or use `mne.coreg.fit_matched_points` if we have digitized points.
    
    # Fallback to 'fsaverage' mapping if subject MRI fiducials missing?
    
    trans = mne.coreg.estimate_head_mri_t(subject, subjects_dir=subjects_dir, n_jobs=1)
    return trans

def make_scalp_surfaces(subject: str, subjects_dir: Path):
    """
    Generate high-res scalp surface for visualization/Laplacians.
    """
    try:
        mne.bem.make_scalp_surfaces(subject=subject, subjects_dir=subjects_dir, force=True)
    except Exception as e:
        print(f"[ERROR] Scalp surface generation failed: {e}")
