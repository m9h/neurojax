
import os
import sys
# Add src to path
sys.path.append(os.path.abspath("src"))

from neurojax.io.uci_loader import load_uci_eeg
import jax.numpy as jnp

def test_loader():
    dataset_path = "downloads/smni_eeg_data.tar.gz"
    if not os.path.exists(dataset_path):
        print(f"Skipping test: {dataset_path} not found.")
        return

    print(f"Loading data from {dataset_path}...")
    # Load only 2 subjects to be fast
    X, info = load_uci_eeg(dataset_path, max_subjects=2)
    
    print("Shape of X:", X.shape)
    print("Number of channels:", len(info['channel_names']))
    print("Sampling freq:", info['sfreq'])
    print("Trial info length:", len(info['trial_info']))
    
    if X.shape[0] > 0:
        print("First trial info:", info['trial_info'][0])
        
    # Validation
    assert X.ndim == 3, "Output should be 3D (Trials x Channels x Time)"
    assert X.shape[1] == 64, "Should have 64 channels"
    assert info['sfreq'] == 256.0
    
    print("Loader verification successful!")

if __name__ == "__main__":
    test_loader()
