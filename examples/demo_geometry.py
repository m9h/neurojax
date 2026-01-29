
import os
import sys
import shutil

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import mne
import numpy as np
from pathlib import Path
from neurojax.io.cmi import CMILoader
from neurojax.geometry.bem import make_bem_surfaces, make_scalp_surfaces

SUBJECT_ID = "sub-NDARGU729WUR"
SUBJECTS_DIR = Path("data/cmi/subjects")

def demo_geometry():
    print("=== Geometry & Head Model Demo ===")
    
    # 1. Fetch T1w
    loader = CMILoader(SUBJECT_ID)
    try:
        t1_path = loader.load_anat(modality="T1w")
        print(f"T1w Volume: {t1_path}")
    except Exception as e:
        print(f"Skipping MRI fetch: {e}")
        return

    # 2. Setup FreeSurfer Structure
    # MNE expects SUBJECTS_DIR/sub/mri/T1.mgz
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_dir = SUBJECTS_DIR / SUBJECT_ID
    mri_dir = sub_dir / "mri"
    mri_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert/Copy T1w
    # mne requires .mgz usually? Or can read .nii.gz?
    # It reads .nii.gz if named properly or passed explicitly?
    # Standard: T1.mgz
    
    # We can use nibabel to convert if needed, or simple copy if MNE accepts it.
    # MNE's watershed BEM generally wraps FreeSurfer calls which expect T1.mgz.
    
    dest_t1 = mri_dir / "T1.nii.gz"
    if not dest_t1.exists():
        shutil.copy(t1_path, dest_t1)
        print(f"Copied to {dest_t1}")
        
    # Mocking FreeSurfer recon-all (Heavy)
    # Ideally we'd run: `recon-all -i T1.nii.gz -s sub -all`
    # That takes hours.
    # For this demo, we assume the user might have surfaces or we attempt a lightweight BEM if possible?
    # MNE doesn't ship a fast BEM solver from raw T1 without FS surfaces (usually).
    
    print("\n[NOTE] Full BEM generation requires FreeSurfer 'recon-all'.")
    print("Executing lightweight scalp extraction (Flash/Watershed check)...")
    
    try:
        # Check if we can run scalp extraction
        make_scalp_surfaces(SUBJECT_ID, SUBJECTS_DIR)
        print(" Scalp surface generated.")
    except Exception as e:
        print(f" Scalp gen failed (expected if no FS): {e}")
        
    # 3. Coregistration (Simulation)
    print("\n--- Coregistration ---")
    # Load EEG to get montage
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
        raw.set_montage('GSN-HydroCel-129')
        print("Montage loaded.")
        
        # Visualize alignment (screenshot)
        # mne.viz.plot_alignment(raw.info, subject=SUBJECT_ID, subjects_dir=SUBJECTS_DIR, dig=True)
        # Without surfaces, this will fail or show nothing.
        
    except Exception:
        pass

    print("[INFO] Geometry pipeline ready. Requires valid FreeSurfer reconstruction (recon-all) to proceed with full BEM.")
    print("DeepBet/Betsurf would be integrated here to refine skull/scalp boundaries.")

if __name__ == "__main__":
    demo_geometry()
