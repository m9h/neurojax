
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neurojax.simulation.vbjax_wrapper import NeuralMassSimulator
from neurojax.analysis.rough import compute_signature, augment_path
from neurojax.analysis.timefreq import morlet_transform

def demo_burst_detection():
    print("\n--- HNN Beta Burst Detection with Rough Paths ---")
    
    # 1. Simulate HNN-like Beta Burst (Transient)
    sim = NeuralMassSimulator(dt=0.0001) # 10kHz
    
    # Generate 200ms of data (2000 points)
    # Burst 1: High Drive (Beta)
    print("Generating Burst (Beta)...")
    beta_lfp = sim.simulate_jansen_rit(duration=0.2, noise_std=1.0, u=150.0)[1]
    
    # Burst 2: Noise (No Drive)
    print("Generating Baseline (Noise)...")
    noise_lfp = sim.simulate_jansen_rit(duration=0.2, noise_std=1.0, u=0.0)[1]
    
    # Normalize amplitudes to force Shape-based detection (not power)
    beta_lfp = (beta_lfp - jnp.mean(beta_lfp)) / jnp.std(beta_lfp)
    noise_lfp = (noise_lfp - jnp.mean(noise_lfp)) / jnp.std(noise_lfp)
    
    # 2. Compute Signatures (Depth 3)
    # Augment with time is crucial for cyclic data
    sig_beta = compute_signature(augment_path(beta_lfp[:, None]), depth=3)
    sig_noise = compute_signature(augment_path(noise_lfp[:, None]), depth=3)
    
    print("\nSignature Comparison (Euclidean Distance):")
    dist = jnp.linalg.norm(sig_beta - sig_noise)
    print(f"Distance between Beta and Noise signatures: {dist:.4f}")
    
    # Compare two noise instances
    noise_lfp2 = (sim.simulate_jansen_rit(duration=0.2, noise_std=1.0, u=0.0)[1])
    noise_lfp2 = (noise_lfp2 - jnp.mean(noise_lfp2)) / jnp.std(noise_lfp2)
    sig_noise2 = compute_signature(augment_path(noise_lfp2[:, None]), depth=3)
    
    dist_noise = jnp.linalg.norm(sig_noise - sig_noise2)
    print(f"Distance between two Noise instances: {dist_noise:.4f}")
    
    ratio = dist / dist_noise
    print(f"Signal-to-Noise Separation Ratio: {ratio:.1f}x")
    
    if ratio > 5.0:
        print("[SUCCESS] Signatures clearly distinguish Structured Burst from Noise (invariant to amplitude).")
    else:
        print("[WARNING] Separation weak. Check simulation parameters.")

if __name__ == "__main__":
    demo_burst_detection()
