"""
Preprocessing pipeline based on Andersen 2018 (Frontiers in Neuroscience).
"""
import mne
import numpy as np
from typing import Tuple, Dict, Any

def maxwell_filter_wrapper(raw: mne.io.Raw, calibration_file: str = None, cross_talk_file: str = None) -> mne.io.Raw:
    """
    Apply Maxwell Filtering (SSS) using MNE.
    
    Args:
        raw: Raw data.
        calibration_file: Path to fine calibration file.
        cross_talk_file: Path to crosstalk compensation file.
    """
    # Auto-detect bad channels if not already done (simplified)
    # in Andersen 2018 they used bads from MaxFilter logs.
    # Here we assume user might provide them or we run without bads for now.
    
    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        calibration=calibration_file,
        cross_talk=cross_talk_file,
        st_duration=10.0, # tSSS buffer
        st_correlation=0.98,
        coord_frame="head"
    )
    return raw_sss

def temporal_filtering(raw: mne.io.Raw, l_freq: float = 1.0, h_freq: float = 40.0) -> mne.io.Raw:
    """
    Apply temporal filtering (FIR, zero-phase, Hamming).
    Andersen 2018: 1-40 Hz.
    """
    raw_filtered = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        phase='zero',
        fir_window='hamming',
        n_jobs=-1
    )
    return raw_filtered

def run_ica(raw: mne.io.Raw, n_components: int = 25, random_state: int = 42) -> mne.preprocessing.ICA:
    """
    Run Fit FastICA for artifact removal.
    """
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=random_state)
    # High-pass for ICA fitting usually recommended
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(raw_ica_fit)
    return ica

def epoch_data(raw: mne.io.Raw, events: np.ndarray, event_id: Dict[str, int], 
               tmin: float = -0.2, tmax: float = 2.9, baseline: Tuple[float, float] = (None, 0)) -> mne.Epochs:
    """
    Epoch the data.
    Andersen 2018: -200ms to 2900ms.
    """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None # Manual rejection preferred, or autoreject
    )
    return epochs

def run_pipeline(raw: mne.io.Raw, events: np.ndarray, event_id: Dict[str, int]) -> mne.Epochs:
    """
    Run full Andersen 2018 pipeline (Simplified).
    
    Steps:
    1. Filter 1-40Hz
    2. Epoch
    """
    # 1. Temporal Filtering
    # Note: MaxFilter usually done first, but often data comes pre-MaxFiltered in some distributions.
    # If raw.info['proc_history'] exists, it might be MaxFiltered.
    
    raw_filt = temporal_filtering(raw, l_freq=1.0, h_freq=40.0)
    
    # 2. Epoching
    epochs = epoch_data(raw_filt, events, event_id, tmin=-0.2, tmax=2.9)
    
    return epochs
