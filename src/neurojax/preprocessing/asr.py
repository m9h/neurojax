"""Artifact Subspace Reconstruction (ASR) preprocessing module.

This module implements a JAX-native version of the ASR algorithm for
automated artifact removal. It uses a calibration dataset to learn
the valid signal subspace and then projects sliding windows of data
to identify and remove high-variance artifact components.
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax.scipy.linalg import eigh

class ASRState(NamedTuple):
    """State for Artifact Subspace Reconstruction."""
    mixing_matrix: jnp.ndarray  # (n_channels, n_channels) - PCA Eigenvectors (cols)
    component_stdevs: jnp.ndarray  # (n_channels,) - Std dev of components in calibration
    cutoff: float  # Standard deviation cutoff

def calibrate_asr(
    clean_data: jnp.ndarray, 
    cutoff: float = 5.0
) -> ASRState:
    """Calibrate ASR using clean/reference data.

    Computes the PCA mixing matrix and component statistics from the
    provided clean data.

    Args:
        clean_data: Array of shape (n_channels, n_times).
        cutoff: Rejection threshold in standard deviations.

    Returns:
        ASRState containing the calibration parameters.
    """
    n_channels, n_times = clean_data.shape
    
    # Center the data
    mean = jnp.mean(clean_data, axis=1, keepdims=True)
    centered = clean_data - mean
    
    # Compute Covariance: C = 1/(N-1) * X * X.T
    cov = jnp.dot(centered, centered.T) / (n_times - 1)
    
    # Eigen Decomposition (PCA)
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = eigh(cov)
    
    # Sort descending (optional, but good for interpretation)
    # JAX eigh returns ascending, so flip
    eigvals = jnp.flip(eigvals)
    eigvecs = jnp.flip(eigvecs, axis=1) # Columns are eigenvectors
    
    # Mixing matrix M = V
    # Projection to component space: Y = M.T @ X
    mixing_matrix = eigvecs
    
    # Project calibration data to get component statistics
    # In PCA, std of components is sqrt(eigenvalues), 
    # but let's compute it explicitly to be safe and robust (if we used robust cov later)
    # components = mixing_matrix.T @ centered
    # stdevs = jnp.std(components, axis=1)
    
    # Theoretical stdevs from eigenvalues (since C was computed from this data)
    stdevs = jnp.sqrt(jnp.maximum(eigvals, 1e-9)) # avoid nan
    
    return ASRState(
        mixing_matrix=mixing_matrix,
        component_stdevs=stdevs,
        cutoff=cutoff
    )

@jax.jit
def _process_window(
    window: jnp.ndarray,
    asr_state: ASRState
) -> jnp.ndarray:
    """Process a single window of data.
    
    Args:
        window: (n_channels, n_window_samples)
        asr_state: ASR calibration state
        
    Returns:
        Cleaned window of same shape.
    """
    # 1. Center window (local centering)
    # ASR usually centers using the window mean or block mean.
    # We'll use window mean.
    mean = jnp.mean(window, axis=1, keepdims=True)
    centered = window - mean
    
    # 2. Project to Component Space
    # Y = M.T @ X
    # shape: (n_comps, n_samples)
    components = jnp.dot(asr_state.mixing_matrix.T, centered)
    
    # 3. Compute Component RMS/Std in this window
    # shape: (n_comps,)
    t_axis = 1
    comp_rms = jnp.sqrt(jnp.mean(components**2, axis=t_axis))
    
    # 4. Identify Bad Components
    # If RMS > cutoff * calibration_std
    # thresholds: (n_comps,)
    thresholds = asr_state.cutoff * asr_state.component_stdevs
    
    # Mask: 1 for keep, 0 for reject
    # We want a smooth transition or hard threshold? Hard for now.
    keep_mask = (comp_rms < thresholds).astype(window.dtype)
    
    # 5. Reconstruct
    # Y_clean = mask * Y
    # X_clean = M @ Y_clean
    components_clean = components * keep_mask[:, None]
    
    reconstructed_centered = jnp.dot(asr_state.mixing_matrix, components_clean)
    
    # Add mean back
    return reconstructed_centered + mean

def apply_asr(
    raw_data: jnp.ndarray,
    asr_state: ASRState,
    window_size: int = 100,
    step_size: int = 50
) -> jnp.ndarray:
    """Apply ASR to raw data using sliding windows.

    Note: This is a basic implementation using overlap-add with a Hanning
    or boxcar window for fusion?
    For simplicity in JAX, we can start with non-overlapping or just
    center-sample stitching if we want to be fast.
    
    However, for artifact removal to be smooth, we usually want overlap.
    Let's implement a simple Overlap-Add (OLA) approach.
    
    Args:
        raw_data: (n_channels, n_times)
        asr_state: Calibration state
        window_size: Samples per window
        step_size: Stride samples (e.g. window_size // 2)
        
    Returns:
        Cleaned data of same shape.
    """
    n_channels, n_times = raw_data.shape
    
    # Pad to fit integer number of windows if needed
    # (Skip complex padding for this V1, just truncate or assume sufficient length)
    
    # Create windows
    # We will use jax.lax.scan or vmap if we can unfold.
    # Let's use a simpler approach: extract windows, vmap process, overlap-add.
    
    starts = jnp.arange(0, n_times - window_size + 1, step_size)
    n_windows = len(starts)
    
    # Extract windows: (n_windows, n_channels, window_size)
    # Using vmap gather
    def get_window(start_idx):
        return jax.lax.dynamic_slice(
            raw_data, 
            (0, start_idx), 
            (n_channels, window_size)
        )
        
    windows = jax.vmap(get_window)(starts)
    
    # Process windows
    cleaned_windows = jax.vmap(
        lambda w: _process_window(w, asr_state)
    )(windows)
    
    # Overlap-Add Reconstruction
    # We need an accumulation buffer and a weight buffer from the window function.
    # Use a Hanning window for smoothness? Or just boxcar?
    # ASR clean_rawdata usually does blending.
    # Let's use Hanning window for both analysis and synthesis to ensure smoothness.
    
    # Define window function (Hanning)
    win_func = jnp.hanning(window_size)
    # Reshape for broadcasting: (1, window_size) because it applies to time axis
    win_func_2d = win_func[None, :] 
    
    # Apply window to cleaned segments
    cleaned_windows_weighted = cleaned_windows * win_func_2d
    
    # Initialize buffers
    output = jnp.zeros_like(raw_data)
    norm = jnp.zeros((1, n_times))
    
    # Map back to full array
    # This is hard to do pure-JAX vectorised without scatter_add
    # We will use value_and_grad or scan style scatter_add
    
    # Flatten/Scattering indices
    # (n_windows, window_size) indices
    window_indices = starts[:, None] + jnp.arange(window_size)[None, :]
    
    output = output.at[:, window_indices.ravel()].add(
        cleaned_windows_weighted.transpose(1, 0, 2).reshape(n_channels, -1)
        # Wait, reshape is dangerous if windows overlap. 
        # .at[...].add() handles overlaps correctly by summing!
        # But we need flattened indices aligned with flattened data.
        # It's cleaner to scan projection.
    )
    
    # Re-do Overlap-Add using scan for correctness and memory efficiency
    # But scatter_add (index_add) is parallel enough.
    
    # Let's extract the scatter logic.
    # For each window, we add to (start:start+win)
    
    # Loop version (jit-compiled scan)
    def add_window(carrier, x):
        out_accum, norm_accum = carrier
        w_cleaned, start_idx = x
        
        # Weighted window
        w_weighted = w_cleaned * win_func_2d
        
        # Slice update
        # We need dynamic_update_slice, but that is Replace, not Add.
        # So we have to read, add, write? No, JAX has .at[...].add
        
        # But .at with dynamic slice is tricky.
        # Actually, for `scan`, the state must be fixed shape. 
        # Updating a full-size array in scan is efficient in JAX? 
        # Yes, if using unroll=1 it might be slow, but usually OK.
        
        # Indices
        idxs = jnp.arange(window_size) + start_idx
        
        out_accum = out_accum.at[:, idxs].add(w_weighted)
        norm_accum = norm_accum.at[:, idxs].add(win_func_2d)
        
        return (out_accum, norm_accum), None

    (output, norm), _ = jax.lax.scan(
        add_window,
        (jnp.zeros_like(raw_data), jnp.zeros((1, n_times))),
        (cleaned_windows, starts)
    )
    
    # Normalize
    # Avoid div by zero
    norm = jnp.maximum(norm, 1e-10)
    final_output = output / norm
    
    return final_output
