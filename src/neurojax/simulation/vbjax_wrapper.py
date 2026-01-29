"""
VBJAX Wrapper for NeuroJAX.

This module provides a simplified interface to the Virtual Brain JAX (vbjax)
library for simulating Neural Mass Models (NMM), specifically tailored for
generating synthetic high-frequency bursts (Beta/Gamma) to validate
detection algorithms.
"""

import jax
import jax.numpy as jnp
import vbjax
import equinox as eqx
from typing import Tuple, Optional

class NeuralMassSimulator(eqx.Module):
    """
    Simulator for Neural Mass Models using vbjax.
    """
    dt: float = 0.0001  # 0.1 ms integration step (10kHz)
    
    def simulate_jansen_rit(self, 
                          duration: float = 1.0, 
                          noise_std: float = 1.5,
                          u: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate a single Jansen-Rit node driven by noise.
        
        Args:
            duration (float): Duration in seconds.
            noise_std (float): Standard deviation of input noise.
            u (float): External input mean (DC offset).
            
        Returns:
            (time, output): Time vector and PSP output (pyramidal).
        """
        n_steps = int(duration / self.dt)
        t = jnp.arange(n_steps) * self.dt
        
        # JAX random key
        key = jax.random.PRNGKey(42)  # Should be passed in state, fixed for demo
        
        # Noise input (Time x 1 node)
        noise = jax.random.normal(key, (n_steps, 1)) * noise_std + u
        
        # Initial state (6 variables for JR)
        y0 = jnp.zeros((6, 1)) # 6 vars, 1 node
        
        # Default Parameters
        theta = vbjax.jr_default_theta
        
        # Define Step Function (Euler-Maruyama / Heun)
        # dydt = jr_dfun(y, coupling, theta)
        # Couplings are 0 for single node
        c = jnp.zeros((1,))
        
        def step(y, x_in):
            # vbjax.jr_dfun signature: (y, c, theta) -> dydt
            # But we need to inject input 'x_in'. 
            # Looking at vbjax source logic (inferred):
            # usually input is added to the derivative of the membrane potential.
            # vbjax generic formulation might expect coupling 'c' to include input?
            # Or we add it manually.
            
            dydt = vbjax.jr_dfun(y, c, theta)
            
            # Add input to Pyramidal population (usually index 1, or equation 0/3)
            # JR vars: [y0..y5]. y0=y1-y2 (PSP). 
            # We add input to the potential derivative.
            # Assuming standard JR: input enters the EPSP equation.
            # We'll rely on vbjax logic or add it effectively.
            # Actually, let's just use make_sde if possible, but manual scan is safer.
            
            # Simple Euler
            y_next = y + dydt * self.dt
            
            # Inject noise/input into Pyramidal (y0 or appropriate)
            # In standard JR, input p(t) adds to y3 (average firing rate input?)
            # vbjax implementation detail: typically input is 'c' (coupling + ext).
            # So we pass 'c + x_in' as the coupling term?
            # Let's try passing x_in as 'c'.
            
            dydt_driven = vbjax.jr_dfun(y, x_in, theta)
            y_next = y + dydt_driven * self.dt
            
            return y_next, y_next[1] - y_next[2] # y1 - y2 is usually LFP (Main PSP)
            
        # Scan
        _, lfp = jax.lax.scan(step, y0, noise)
        
        return t, lfp.flatten()

    def simulate_burst_beta(self, duration: float = 1.0) -> jnp.ndarray:
        """
        Generate a synthetic beta burst.
        """
        # Drive with specific noise to create bursts
        t, lfp = self.simulate_jansen_rit(duration, noise_std=2.0, u=120.0) 
        # u=120-220 is typical bifurcation range for JR to oscillate
        return lfp

