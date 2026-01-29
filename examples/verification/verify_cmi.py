
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import mne
from neurojax.io.cmi import CMILoader
import fsspec

SUBJECT_ID = "sub-NDARGU729WUR"

def verify_cmi_loader():
    print(f"\n--- Verifying CMI Loader for {SUBJECT_ID} ---")
    
    loader = CMILoader(SUBJECT_ID)
    
    # 1. Load Resting State
    try:
        print("Downloading/Loading Resting State...")
        raw_rest = loader.load_task("RestingState")
        print(f"[SUCCESS] Loaded Resting State: {raw_rest}")
        print(raw_rest.info)
    except Exception as e:
        print(f"[FAILURE] Resting State load failed: {e}")

    # 2. Load Task (Surround Suppression)
    try:
        print("\nDownloading/Loading Surround Suppression (Run 1)...")
        raw_task = loader.load_task("surroundSupp", run=1)
        print(f"[SUCCESS] Loaded Surround Task: {raw_task}")
    except Exception as e:
        print(f"[FAILURE] Task load failed: {e}")
        
    # 3. Check for MRI Data (T1w)
    print("\n--- Checking for MRI Data (T1w) ---")
    fs = fsspec.filesystem("s3", anon=True)
    # HBN BIDS structure: root/sub-ID/anat/
    # Need to check where MRI is stored in S3. Usually same root but different subfolder?
    # Inspect root again for structure?
    # Assuming standard BIDS:
    # fcp-indi/data/Projects/HBN/BIDS_Curated/sub-ID/anat ? or just BIDS/sub-ID/anat
    
    mri_root = "fcp-indi/data/Projects/HBN/BIDS_curated/"
    # Check if subject is in the list
    try:
        # Listing all subjects is slow (thousands). 
        # Check specific path existence.
        target_sub = f"{mri_root}{SUBJECT_ID}/anat/"
        if fs.exists(target_sub):
            print(f"Directory exists: {target_sub}")
            mri_files = fs.ls(target_sub)
            for f in mri_files:
                print(f" - {os.path.basename(f)}")
                if "T1w.nii.gz" in f:
                     print("--> [MATCH] T1w MRI found!")
        else:
            print(f"MRI Directory NOT found: {target_sub}")
            # Maybe it's in a different release? 
            # HBN MRI is huge. 
    except Exception as e:
        print(f"MRI probe failed: {e}")

if __name__ == "__main__":
    verify_cmi_loader()
