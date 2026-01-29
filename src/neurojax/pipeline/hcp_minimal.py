"""HCP Minimal Preprocessing Pipeline."""
import jax
import jax.numpy as jnp
from neurojax.preprocessing.filter import noncausal_highpass
from neurojax.preprocessing.resample import resample_minimal
from neurojax.preprocessing.interpolate import spherical_spline_interpolate

def hcp_minimal_preproc(
    raw_data, 
    sfreq, 
    layout_coords=None, 
    bad_channels=None, 
    target_fs=None,
    highpass_freq=0.5
):
    """
    Apply HCP 'Minimal Minimal' preprocessing steps.
    
    Steps:
    1. Highpass Filter (default 0.5Hz, Butterworth SOS) to remove drifts.
    2. Resample (if target_fs is provided and different from sfreq).
    3. Bad Channel Interpolation (if bad_channels and layout_coords provided).
    
    Args:
        raw_data: (n_channels, n_times) array.
        sfreq: Original sampling rate in Hz.
        layout_coords: (n_channels, 3) array of sensor coordinates on unit sphere.
                       Required for interpolation.
        bad_channels: List or array of bad channel indices to interpolate.
        target_fs: Target sampling rate. If None, no resampling is performed.
        highpass_freq: Highpass cutoff frequency. Default 0.5 Hz (HCP standard).
        
    Returns:
        cleaned_data: Processed data array.
    """
    
    # 1. Highpass Filtering
    # Remove slow drifts before resampling to avoid edge artifacts
    print(f"Applying Highpass Filter: {highpass_freq} Hz")
    data_hp = noncausal_highpass(raw_data, sfreq, cutoff=highpass_freq)
    
    # 2. Resampling
    current_data = data_hp
    if target_fs is not None and target_fs != sfreq:
        print(f"Resampling from {sfreq} Hz to {target_fs} Hz")
        current_data = resample_minimal(current_data, sfreq, target_fs)
    
    # 3. Bad Channel Interpolation
    if bad_channels is not None and len(bad_channels) > 0:
        if layout_coords is None:
            raise ValueError("layout_coords is required for interpolation")
            
        print(f"Interpolating {len(bad_channels)} bad channels")
        current_data = spherical_spline_interpolate(
            current_data, 
            bad_idx=jnp.array(bad_channels), 
            sensor_coords=layout_coords
        )
        
    return current_data
