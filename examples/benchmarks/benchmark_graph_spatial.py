
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.spatial.graph import EEGGraph

def benchmark_graph():
    print("--- Benchmarking Graph Spatial Smoothing (Jraph) ---")
    
    # 1. Setup Graph
    try:
        eg = EEGGraph(montage_name="GSN-HydroCel-129", adjacency_dist=0.05)
        print(f"Graph Created: {eg.n_node} nodes, {eg.n_edge} edges.")
    except Exception as e:
        print(f"Graph creation failed (likely missing template): {e}")
        # Fallback to standard 10-20
        eg = EEGGraph(montage_name="standard_1020", adjacency_dist=0.08)
        print(f"Fallback Graph (10-20): {eg.n_node} nodes, {eg.n_edge} edges.")

    # 2. Create Data
    # Smooth signal: same value everywhere (0.0) -> Laplacian should be 0
    # Add Outlier: Ch 10 = 5.0
    data = jnp.zeros((eg.n_node, 100))
    data = data.at[10, :].set(5.0) # Bad channel
    
    print(f"Original Outlier (Ch 10): {jnp.mean(data[10]):.2f}")
    
    # 3. Laplacian Smoothing (Diffusion)
    # alpha * L * X
    smoothed = eg.smooth(data, alpha=0.1)
    
    val_out = jnp.mean(smoothed[10]) # Should decrease (diffusion to neighbors)
    val_neighbor = jnp.mean(smoothed[11]) # Should increase (diffusion from outlier)
    
    print(f"Smoothed Outlier (Ch 10): {val_out:.2f}")
    
    if val_out < 5.0:
        print("[SUCCESS] Laplacian Smoothing reduced outlier amplitude.")
        
    # 4. Jraph Convolution (Message Passing)
    # Computes Sum of Neighbors + Self
    # If Ch 10 is 5.0, sum will be > 5.0 (if neighbors are 0, sum=5+0=5)
    # Actually wait: Jraph conv returns new feature.
    # Our simple implementation: node + sum(neighbors)
    # If neighbors are 0, it stays 5.
    # Let's see if neighbors get signal.
    
    convolved = eg.jraph_convolution(data)
    
    # Check a neighbor of 10.
    # Let's find neighbors of 10
    neighbors = eg.receivers[eg.senders == 10]
    if len(neighbors) > 0:
        nb_idx = neighbors[0]
        val_nb_orig = jnp.mean(data[nb_idx])
        val_nb_conv = jnp.mean(convolved[nb_idx])
        
        print(f"Neighbor {nb_idx} Original: {val_nb_orig:.2f}")
        print(f"Neighbor {nb_idx} Convolved (Received from 10): {val_nb_conv:.2f}")
        
        if val_nb_conv > val_nb_orig:
            print("[SUCCESS] Jraph propagated signal to neighbors (Contiguity).")

if __name__ == "__main__":
    benchmark_graph()
