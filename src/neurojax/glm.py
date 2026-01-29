import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import distrax
from jaxtyping import Array, Float, PRNGKeyArray

class GeneralLinearModel(eqx.Module):
    """
    A JAX-native General Linear Model for MEG/EEG.
    Replaces osl-ephys GLM with a differentiable, GPU-accelerated version.
    """
    
    # Model parameters are stored as leaves in the PyTree
    design_matrix: Float[Array, "time regressors"]
    data: Float[Array, "time sensors"]
    betas: Float[Array, "regressors sensors"] | None = None
    residuals: Float[Array, "time sensors"] | None = None
    
    def __init__(self, design_matrix: Array, data: Array):
        self.design_matrix = design_matrix
        self.data = data
        self.betas = None
        self.residuals = None

    def fit(self):
        """
        Fits the GLM using Lineax for robust linear solves.
        Uses QR decomposition or SVD automatically for stability.
        """
        # Lineax solver is often faster and more robust to ill-conditioned 
        # matrices than standard numpy.linalg.solve
        operator = lx.MatrixLinearOperator(self.design_matrix)
        solver = lx.QR() # Robust choice for design matrices
        
        # Solve X * Beta = Y
        # We use vmap to solve for each sensor independently (sharing the operator)
        # operator maps [regressors] -> [time]
        # data is [time, sensors] -> we want to map over axis 1
        
        def solve_single(y):
            return lx.linear_solve(operator, y, solver=solver).value
            
        # vmap over sensors (axis 1 of data), return betas [regressors, sensors]
        betas = jax.vmap(solve_single, in_axes=1, out_axes=1)(self.data)
        
        residuals = self.data - (self.design_matrix @ betas)
        
        # Return a new instance of the model with updated state (functional style)
        return eqx.tree_at(
            lambda m: (m.betas, m.residuals), 
            self, 
            (betas, residuals),
            is_leaf=lambda x: x is None
        )

    def get_t_stats(self, contrast: Float[Array, "regressors"]):
        """Computes T-statistics for a specific contrast vector."""
        if self.betas is None:
            raise ValueError("Model must be fitted before computing stats.")
            
        n_obs = self.data.shape[0]
        n_regressors = self.design_matrix.shape[1]
        dof = n_obs - n_regressors
        
        # Variance estimate (residual sum of squares / dof)
        sigma_sq = jnp.sum(self.residuals ** 2, axis=0) / dof
        
        # Design variance: C.T @ (X.T @ X)^-1 @ C
        # We assume X is already decomposed or just invert X.T@X for stats
        # For small design matrices, pinv is fine. For larger ones, could reuse factorization.
        cov_matrix = jnp.linalg.pinv(self.design_matrix.T @ self.design_matrix)
        design_var = contrast.T @ cov_matrix @ contrast
        
        # Standard Error
        se = jnp.sqrt(sigma_sq * design_var)
        
        # T-statistic: (C.T @ Beta) / SE
        effect_size = contrast @ self.betas
        t_values = effect_size / se
        
        return t_values

    def log_likelihood(self):
        """
        Uses Distrax to compute the log-likelihood of the data given the model.
        Useful for model comparison (e.g., AIC/BIC).
        """
        if self.residuals is None:
            raise ValueError("Model must be fitted first.")
            
        # We model the residuals as Independent Normals
        # (A more complex model could use MultivariateNormal to capture sensor noise correlations)
        
        scale = jnp.std(self.residuals, axis=0)
        dist = distrax.Normal(loc=0.0, scale=scale)
        
        # Sum log_prob across time and sensors
        return jnp.sum(dist.log_prob(self.residuals))

# -------------------------------------------------------------------------
# THE ACCELERATOR: JIT-compiled Permutation Testing
# -------------------------------------------------------------------------

@eqx.filter_jit
def run_permutation_test(
    model: GeneralLinearModel, 
    contrast: Array, 
    key: PRNGKeyArray, 
    n_perms: int = 1000
):
    """
    Runs massive permutation testing on the GPU.
    Replaces the 'dask' loop in standard OSL.
    """
    
    # 1. Fit the true model
    fitted_model = model.fit()
    true_t = fitted_model.get_t_stats(contrast)
    
    # 2. Define a single permutation step
    def single_perm(k):
        # Shuffle the design matrix rows (preserve correlation structure of Y)
        # Or shuffle Y (common in some GLM implementations).
        # Here we shuffle Y rows to break X-Y relationship, preserving spatial structure.
        perm_idx = jax.random.permutation(k, model.data.shape[0])
        shuffled_data = model.data[perm_idx]
        
        # Create a temp model
        perm_model = GeneralLinearModel(model.design_matrix, shuffled_data)
        perm_model = perm_model.fit()
        return perm_model.get_t_stats(contrast)

    # 3. Vectorize (vmap) over random keys
    keys = jax.random.split(key, n_perms)
    perm_t_stats = jax.vmap(single_perm)(keys) # shape: [n_perms, n_sensors]

    # 4. Compute P-values (max-t correction for multiple comparisons across sensors)
    # We take the max T across sensors for each permutation to build the null distribution
    max_t_null = jnp.max(jnp.abs(perm_t_stats), axis=1)
    
    # Compare true T against the Max-T null distribution
    p_values = jnp.mean(max_t_null[:, None] >= jnp.abs(true_t)[None, :], axis=0)
    
    return true_t, p_values
