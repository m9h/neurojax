"""
Wakeman-Henson (ds000117) Dataset Loader.
"""
import os
from pathlib import Path
import mne
import pandas as pd
import openneuro

class WakemanHensonLoader:
    """
    Loader for the Wakeman-Henson dataset (ds000117) from OpenNeuro.
    """
    DATASET_ID = "ds000117"
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory to store the dataset. Defaults to ~/mne_data/ds000117.
        """
        if data_dir is None:
            self.data_dir = Path(mne.get_config("MNE_DATA", default="~/mne_data")) / self.DATASET_ID
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir = self.data_dir.expanduser()
        
    def download(self, subject: str = "01"):
        """
        Download data for a specific subject using openneuro-py.
        """
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"Downloading {self.DATASET_ID} subject {subject} to {self.data_dir}...")
        # Note: openneuro.download returns None on success
        openneuro.download(
            dataset=self.DATASET_ID,
            target_dir=self.data_dir,
            include=[f"sub-{subject}/*"]
        )
        print("Download complete.")

    def load_subject(self, subject: str = "01", run: int = 1) -> mne.io.Raw:
        """
        Load raw MEG/EEG data for a subject and run.
        """
        subject_str = f"sub-{subject}"
        run_str = f"run-{run:02d}"
        
        # BIDS path construction
        meg_path = self.data_dir / subject_str / "ses-meg" / "meg" / \
                  f"{subject_str}_ses-meg_task-facerecognition_{run_str}_meg.fif"
                  
        if not meg_path.exists():
            # Try 3-digit run number (OpenNeuro format sometimes varies)
            run_str_3 = f"run-{run:03d}"
            meg_path_3 = self.data_dir / subject_str / "ses-meg" / "meg" / \
                  f"{subject_str}_ses-meg_task-facerecognition_{run_str_3}_meg.fif"
            if meg_path_3.exists():
                meg_path = meg_path_3
            else:
                raise FileNotFoundError(f"MEG file not found: {meg_path}. Did you run download()?")
            
        raw = mne.io.read_raw_fif(meg_path, preload=True)
        return raw

    def get_runs(self, subject: str = "01") -> list[int]:
        """
        Identify available runs for a subject.
        """
        subject_str = f"sub-{subject}"
        meg_dir = self.data_dir / subject_str / "ses-meg" / "meg"
        if not meg_dir.exists():
            return []
        
        runs = []
        for fname in meg_dir.glob("*_meg.fif"):
            # Extract run number
            parts = fname.name.split('_')
            for part in parts:
                if part.startswith('run-'):
                    try:
                        runs.append(int(part.split('-')[1]))
                    except ValueError:
                        pass
        return sorted(list(set(runs)))

    def load_events(self, raw: mne.io.Raw, subject: str = "01", run: int = 1):
        """
        Load events from the raw file or BIDS events.tsv.
        """
        # For this dataset, events are in the stimulus channel STI101
        events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002)
        return events
