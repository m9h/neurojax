
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neurojax.simulation.vbjax_wrapper import NeuralMassSimulator

def verify_simulator():
    print("Initializing Simulator...")
    sim = NeuralMassSimulator()
    
    print("Simulating Beta Burst (1s)...")
    try:
        lfp = sim.simulate_burst_beta(duration=1.0)
        print(f"LFP Shape: {lfp.shape}")
        print(f"LFP Range: [{jnp.min(lfp)}, {jnp.max(lfp)}]")
        
        # Check for NaNs
        if jnp.isnan(lfp).any():
            print("[FAILURE] Simulation produced NaNs!")
        else:
            print("[SUCCESS] Simulation ran without NaNs.")
            
        # Basic oscillation check (simple zero crossing or std)
        std = jnp.std(lfp)
        print(f"LFP Std Dev: {std}")
        if std < 1e-6:
             print("[WARNING] LFP seems flat/zero.")
        
    except Exception as e:
        print(f"[FAILURE] Simulation crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_simulator()
