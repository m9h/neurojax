import jax
import jax.numpy as jnp
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict, Optional

def load_ds004745(dataset_path: str, subject_id: str = "sub-001", session: str = "ses-01") -> Tuple[jax.Array, Dict]:
    """
    Loads one subject from the SSVEP with Artifact Trials dataset (ds004745).
    
    Args:
        dataset_path: Root path to the dataset.
        subject_id: Subject ID (e.g., 'sub-001').
        session: Session ID (e.g., 'ses-01').
        
    Returns:
        raw_data: JAX Array of shape (n_channels, n_timepoints)
        info: Dictionary containing 'sfreq', 'ch_names', 'events', 'event_id'
    """
    ds_path = Path(dataset_path)
    # Construct typical BIDS path: sub-001/ses-01/eeg/sub-001_ses-01_task-ssvep_eeg.vhdr
    # Note: We need to handle potential file naming variations, but this is standard.
    
    # Try finding the file
    subject_dir = ds_path / subject_id / session / 'eeg'
    if not subject_dir.exists():
        # Fallback for no-session structure if applicable, though BIDS says ses is optional.
        # Check if subject_dir without session exists
        subject_dir = ds_path / subject_id / 'eeg'
    
    if not subject_dir.exists():
         raise FileNotFoundError(f"Subject directory not found at {subject_dir}")

    # Find the EEG file (.set or .vhdr)
    # Datalad download revealed .set format for this dataset
    eeg_files = list(subject_dir.glob("*.set"))
    if not eeg_files:
        eeg_files = list(subject_dir.glob("*.vhdr"))
        
    if not eeg_files:
        raise FileNotFoundError(f"No .set or .vhdr file found in {subject_dir}")
    
    eeg_path = eeg_files[0]
    print(f"Loading {eeg_path}...")
    
    # Load with MNE
    if eeg_path.suffix == '.set':
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
    
    # Get data as numpy array (n_channels, n_times)
    data = raw.get_data()
    
    # Convert to JAX
    jax_data = jnp.array(data, dtype=jnp.float32)
    
    # Extract events if strictly needed, but raw object has them
    # We'll return basic metadata
    info = {
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
        'annotations': raw.annotations, # Important for artifact labeling
        # 'events': mne.events_from_annotations(raw)[0] if raw.annotations else None
    }
    
    return jax_data, info
