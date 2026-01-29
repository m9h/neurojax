import jax.numpy as jnp
import equinox as eqx
import diffrax

class AbstractNeuralMass(eqx.Module):
    """Template for all biophysical kernels."""
    def vector_field(self, t, state, args):
        raise NotImplementedError

class WongWang(AbstractNeuralMass):
    """
    Reduced Wong-Wang Neural Mass Model.
    Represents the 'mean field' firing rate of a cortical region.
    """
    # Learnable parameters (e.g., global coupling G)
    G: float 
    
    # Constants (Time constants, etc.)
    tau_s: float = 100.0  # ms
    gamma: float = 0.641
    w: float = 0.6
    
    def vector_field(self, t, y, args):
        S = y  # Synaptic gating variable
        
        # In a network model, Input I would come from 'args' (Structural Connectivity)
        I_net, I_ext = args 
        
        # Transfer function (Sigmoid logic)
        x = self.w * I_net + I_ext
        H = (310 - 125*x) / (1 + jnp.exp(-2.6 * (125*x - 55)))
        
        # Differential Equation: dS/dt
        dS = -S/self.tau_s + (1 - S) * self.gamma * H
        return dS
