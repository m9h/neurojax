"""
SPM 25 Module: Differentiable Parametric Statistics and Random Field Theory.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Tuple, List

class GeneralLinearModel(eqx.Module):
    """
    JAX-native General Linear Model (GLM).
    
    Y = X * beta + epsilon
    """
    def __init__(self):
        pass

    def fit(self, Y: Float[Array, "n_samples n_features"], X: Float[Array, "n_samples n_regressors"]) -> Float[Array, "n_regressors n_features"]:
        """
        Fit GLM using OLS.
        """
        # beta = (X'X)^-1 X'Y
        # Use pseudo-inverse for stability
        beta = jnp.linalg.pinv(X) @ Y
        return beta

    def residuals(self, Y, X, beta):
        return Y - X @ beta

    def compute_stats(self, Y, X, beta, contrast: Float[Array, "n_regressors"], stat_type: str = 'T'):
        """
        Compute T/F statistics for a given contrast.
        """
        n, p = X.shape
        residuals = self.residuals(Y, X, beta)
        sigma_sq = jnp.sum(residuals**2, axis=0) / (n - p)
        
        if stat_type == 'T':
            # T = (c'beta) / sqrt(sigma_sq * c'(X'X)^-1 c)
            c = contrast
            XtX_inv = jnp.linalg.pinv(X.T @ X)
            denom = jnp.sqrt(sigma_sq * (c.T @ XtX_inv @ c))
            numerator = c.T @ beta
            t_stat = numerator / (denom + 1e-12) # Avoid div zero
            return t_stat
        else:
            raise NotImplementedError("F-statistics not yet implemented.")

class RandomFieldTheory(eqx.Module):
    """
    Random Field Theory (RFT) for topological inference.
    """
    def __init__(self):
        pass
        
    def euler_characteristic_density(self, t: Float[Array, "..."], df: float, D: int = 2) -> Float[Array, "..."]:
        """
        Compute Euler Characteristic (EC) density for T-field.
        
        Ref: Worsley et al. 1996.
        """
        # Simplified implementation for Gaussian fields (high DF approximation)
        # rho_0 = 1 - Phi(t) (Survival function)
        # rho_1 = (4 ln 2)^0.5 / 2pi * exp(-t^2/2)
        # rho_2 = (4 ln 2) / (2pi)^1.5 * exp(-t^2/2) * t
        
        # Using approximations for D=2
        const_1 = (4 * jnp.log(2))**0.5 / (2 * jnp.pi)
        const_2 = (4 * jnp.log(2)) / ((2 * jnp.pi)**1.5)
        
        exponent = jnp.exp(-0.5 * t**2)
        
        rho_0 = 0.5 * jax.scipy.special.erfc(t / jnp.sqrt(2))
        rho_1 = const_1 * exponent
        rho_2 = const_2 * exponent * t
        
        if D == 0:
            return rho_0
        elif D == 1:
            return rho_1
        elif D == 2:
            return rho_2
        else:
            raise NotImplementedError("D > 2 not implemented.")

    def cluster_threshold(self, resels: float, p_val: float = 0.05, D: int = 2):
        """
        Compute cluster size threshold.
        """
        # Placeholder for full RFT cluster thresholding logic
        # Typically involves inverse of EC density
        return 0.0 # TODO
        
    def correct_p_values(self, t_map: Float[Array, "W H"], resels: float, D: int = 2):
        """
        Apply RFT correction to a T-map.
        """
        # Compute max T
        max_t = jnp.max(t_map)
        
        # Expected EC
        # E[EC] = R * rho_D(t)
        ec_dens = self.euler_characteristic_density(max_t, df=100, D=D) # Assuming high DF
        expected_ec = resels * ec_dens
        
        # Under null, P(max > u) approx E[EC]
        p_corrected = expected_ec
        return jnp.minimum(p_corrected, 1.0)
