
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import time
import jax
import jax.numpy as jnp
import numpy as np
from neurojax.io.cmi import CMILoader
from neurojax.analysis.filtering import filter_fft, notch_filter_fft, robust_reference

SUBJECT_ID = "sub-NDARGU729WUR"

def benchmark():
    print(f"Benchmarking Denoising on {SUBJECT_ID}...")
    loader = CMILoader(SUBJECT_ID)
    
    # Load MNE Raw (CPU)
    t0 = time.time()
    raw = loader.load_task("RestingState")
    load_time = time.time() - t0
    print(f"Load Time (MNE): {load_time:.2f}s")
    
    data = raw.get_data() # (Channels, Time) float64
    sfreq = raw.info['sfreq']
    print(f"Data Shape: {data.shape} ({data.nbytes / 1e6:.1f} MB)")
    
    # 1. JAX Transfer
    print("Transferring to JAX Device...")
    t0 = time.time()
    data_jax = jnp.array(data, dtype=jnp.float32)
    t_transfer = time.time() - t0
    print(f"JAX Transfer: {t_transfer:.4f}s")
    
    # Warmup JIT
    print("Warming up JIT...")
    _ = filter_fft(data_jax[:, :1000], sfreq, 1.0, 100.0)
    _ = notch_filter_fft(data_jax[:, :1000], sfreq)
    _ = robust_reference(data_jax[:, :1000])
    
    # 2. Pipeline Execution
    print("\nExecuting Pipeline (Highpass 1Hz -> Notch 60Hz -> Robust Ref)...")
    t0 = time.time()
    
    # Chain functions
    d1 = filter_fft(data_jax, sfreq, f_low=1.0)
    d2 = notch_filter_fft(d1, sfreq, freq=60.0)
    d3 = robust_reference(d2)
    
    # Block until done
    d3.block_until_ready()
    
    t_pipeline = time.time() - t0
    
    print(f"Pipeline Time: {t_pipeline:.4f}s")
    
    # Throughput
    n_samples = data.shape[1]
    n_channels = data.shape[0]
    total_samples = n_samples * n_channels
    throughput = total_samples / t_pipeline
    
    print(f"Throughput: {throughput / 1e6:.2f} MegaSamples/s")
    print(f"Real-time Factor: {data.shape[1]/sfreq / t_pipeline:.1f}x faster than real-time")

if __name__ == "__main__":
    benchmark()
