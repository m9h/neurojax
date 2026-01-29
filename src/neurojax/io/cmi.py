"""
CMI Healthy Brain Network (HBN) Loader for NeuroJAX.

Handles downloading and loading of EEG data from the FCP-INDI S3 bucket.
"""

import os
import fsspec
import mne
from pathlib import Path
from typing import Optional, List

# CMI Bucket Prefix
BUCKET_ROOT = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1"
LOCAL_DATA_DIR = Path("data/cmi")

class CMILoader:
    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.fs = fsspec.filesystem("s3", anon=True)
        self.base_path = f"{BUCKET_ROOT}/{subject_id}/eeg"
        self.local_dir = LOCAL_DATA_DIR / subject_id
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def list_eeg_files(self) -> List[str]:
        """List all available EEG files for the subject on S3."""
        return self.fs.ls(self.base_path)

    def download_file(self, remote_path: str, force: bool = False) -> Path:
        """Download a file from S3 to local storage."""
        fname = os.path.basename(remote_path)
        local_path = self.local_dir / fname
        
        if local_path.exists() and not force:
            print(f"[CACHE] Using existing {local_path}")
            return local_path
            
        print(f"[DOWNLOAD] Fetching {fname} from S3...")
        self.fs.get(remote_path, str(local_path))
        return local_path

    def load_task(self, task_name: str, run: int = 1) -> mne.io.Raw:
        """
        Load a specific task (e.g., 'RestingState', 'surroundSupp').
        Downloads the .set file (and .fdt if present).
        """
        # Find the .set file
        files = self.list_eeg_files()
        
        # Pattern matching
        # e.g. sub-NDAR..._task-RestingState_eeg.set
        # e.g. sub-NDAR..._task-surroundSupp_run-1_eeg.set
        
        candidates = []
        for f in files:
            if not f.endswith(".set"):
                continue
            if task_name in f:
                if run is not None:
                     if f"run-{run}" in f:
                         candidates.append(f)
                     elif "RestingState" in task_name: # Resting often has no run?
                         candidates.append(f)
                else:
                    candidates.append(f)
        
        if not candidates:
            # Try looser match for RestingState if no run specified
            if task_name == "RestingState":
                 for f in files:
                     if "RestingState" in f and f.endswith(".set"):
                         candidates.append(f)

        if not candidates:
            raise FileNotFoundError(f"No .set file found for task {task_name}, run {run}")
            
        # Prioritize exact match
        target_remote = candidates[0]
        
        # Download .set
        local_set = self.download_file(target_remote)
        
        # Check for .fdt matching the .set
        # (Assuming same basename but .fdt)
        remote_fdt = target_remote.replace(".set", ".fdt")
        # Check if exists on S3?
        if self.fs.exists(remote_fdt):
            self.download_file(remote_fdt)
        else:
            print(f"[INFO] No .fdt companion found for {os.path.basename(target_remote)}. Assuming embedded data.")
            
        # Load with MNE
        try:
            raw = mne.io.read_raw_eeglab(local_set, preload=True)
            return raw
        except Exception as e:
            print(f"[ERROR] Failed to load EEGLAB file: {e}")
            raise


    def load_anat(self, modality: str = "T1w") -> Path:
        """
        Load anatomical MRI volume (T1w or T2w).
        Downloads from `BIDS_curated` directory on S3 (requires searching for session).
        
        Args:
            modality: "T1w" or "T2w".
            
        Returns:
            Path to local .nii.gz file.
        """
        # HBN Structure: fcp-indi/data/Projects/HBN/BIDS_curated/{sub}/{ses}/anat/
        curated_root = "s3://fcp-indi/data/Projects/HBN/BIDS_curated"
        sub_root = f"{curated_root}/{self.subject_id}"
        
        try:
            # List subject dir to find session (e.g., ses-HBNsiteRU)
            sessions = self.fs.ls(sub_root)
        except Exception: 
            # Fallback or error
            raise FileNotFoundError(f"Subject {self.subject_id} not found in BIDS_curated ({sub_root})")
            
        # Filter for ses-* directories
        ses_candidates = [s for s in sessions if "ses-" in os.path.basename(s)]
        if not ses_candidates:
            raise FileNotFoundError(f"No session found for {self.subject_id} in {sub_root}")
            
        # Use first session
        target_ses = ses_candidates[0] # Full path e.g. fcp-indi/.../ses-HBNsiteRU
        
        anat_path = f"s3://{target_ses}/anat" # fs.ls returns path without s3:// usually? 
        # fs.ls output: 'fcp-indi/...' (no s3:// prefix). fsspec usually handles this.
        # But for constructing next path, we need to be careful.
        # fs.ls returns list of paths relative to fs root or absolute buckets.
        
        # Let's list the anat dir inside the session
        # target_ses from fs.ls usually looks like 'fcp-indi/data/...'
        # We need to prepend 's3://' if using glob, but self.fs.ls() takes stripped path?
        # Actually self.fs is s3 filesystem. It takes paths without s3:// usually or handles both.
        # Let's use the string returned by ls directly.
        
        anat_dir = f"{target_ses}/anat"
        try:
            files = self.fs.ls(anat_dir)
        except FileNotFoundError:
            raise FileNotFoundError(f"No anat directory in {anat_dir}")
            
        # Find T1w
        candidates = [f for f in files if modality in f and f.endswith(".nii.gz") and not ".json" in f]
        if not candidates:
             raise FileNotFoundError(f"No {modality} volume found in {anat_dir}")
             
        target_remote = candidates[0]
        # target_remote is likely 'fcp-indi/...'
        # To download, we need to pass this to fs.get
        # fs.get usually expects 'bucket/key' or 's3://bucket/key'
        
        # Download
        local_dir_anat = self.local_dir / "anat"
        local_dir_anat.mkdir(exist_ok=True, parents=True)
        
        fname = os.path.basename(target_remote)
        local_path = local_dir_anat / fname
        
        if local_path.exists():
            print(f"[CACHE] Using existing MRI: {local_path}")
            return local_path
            
        print(f"[DOWNLOAD] Fetching MRI {fname} from {target_remote}...")
        self.fs.get(target_remote, str(local_path))
        return local_path
