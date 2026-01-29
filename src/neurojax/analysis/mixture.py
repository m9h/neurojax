# © NeuroJAX developers
#
# License: BSD (3-clause)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.scipy.stats import norm, gamma
from functools import partial

class GaussianGammaMixture(eqx.Module):
    """
    Gaussian-Gamma Mixture Model for IC Thresholding.
    
    Model: P(x) = pi_g * N(x|mu, sigma) + 
                  pi_pos * Gamma(x-mu|alpha, beta) + 
                  pi_neg * Gamma(-(x-mu)|alpha, beta)
                  
    We use unconstrained parameters internally and transform them for computation.
    """
    # Unconstrained parameters
    # Weights (logits)
    weights_logits: jnp.ndarray # shape (3,) [Gaussian, PosGamma, NegGamma]
    
    # Gaussian
    mu: jnp.ndarray # Scalar (but array 0-dim)
    log_sigma: jnp.ndarray
    
    # Gamma (shared shape/rate for pos/neg tails usually, or separate?)
    # MELODIC often uses separate. Let's use separate for flexibility.
    # Pos Tail
    log_alpha_pos: jnp.ndarray
    log_beta_pos: jnp.ndarray
    
    # Neg Tail
    log_alpha_neg: jnp.ndarray
    log_beta_neg: jnp.ndarray
    
    def __init__(self, key):
        # Initialize
        # Gaussian ~ Standard Normal usually for ICs
        self.mu = jnp.array(0.0)
        self.log_sigma = jnp.array(0.0) # sigma=1
        
        # Gamma tails
        # alpha ~ 2.0, beta ~ 1.0?
        self.log_alpha_pos = jnp.array(0.0) # alpha=1
        self.log_beta_pos = jnp.array(0.0) # beta=1
        self.log_alpha_neg = jnp.array(0.0)
        self.log_beta_neg = jnp.array(0.0)
        
        # Weights: Start with dominant Gaussian
        # [Gaussian, Pos, Neg] -> [0.8, 0.1, 0.1]
        self.weights_logits = jnp.array([2.0, -1.0, -1.0])
        
    def get_params(self):
        """Transform to constrained space."""
        weights = jax.nn.softmax(self.weights_logits)
        sigma = jnp.exp(self.log_sigma)
        
        alpha_pos = jax.nn.softplus(self.log_alpha_pos) + 1.0 # Ensure > 1 for bell-ish shape? Or >0
        beta_pos = jax.nn.softplus(self.log_beta_pos) + 1e-6
        
        alpha_neg = jax.nn.softplus(self.log_alpha_neg) + 1.0
        beta_neg = jax.nn.softplus(self.log_beta_neg) + 1e-6
        
        return weights, self.mu, sigma, alpha_pos, beta_pos, alpha_neg, beta_neg
        
    def log_prob(self, x):
        weights, mu, sigma, a_pos, b_pos, a_neg, b_neg = self.get_params()
        
        # Gaussian Log Prob
        lp_g = jnp.log(weights[0]) + norm.logpdf(x, loc=mu, scale=sigma)
        
        # Pos Gamma Log Prob
        # Gamma pdf is defined for x > 0.
        # We model x - mu > 0 => x > mu
        # Shifted Gamma: Gamma(x - mu)
        # We need to handle the support. jax.scipy.stats.gamma returns nan/inf for x<0?
        # Actually it returns -inf for logpdf outside support?
        # Let's ensure strict support handling.
        
        # Safe shifted x
        # eps to avoid log(0) at exactly mu
        diff = x - mu
        
        # For Gamma(x|a, scale=1/b) -> beta is rate in JAX? No, scipy uses scale=theta.
        # jax.scipy.stats.gamma.logpdf(x, a, loc=0, scale=1)
        # parameterization: x^(a-1) * exp(-x/scale) ...
        # Standard notation: beta is rate (inverse scale). scale = 1/beta.
        
        scale_pos = 1.0 / b_pos
        scale_neg = 1.0 / b_neg
        
        # Use simple masking for stability, though logsumexp handles -inf well.
        
        # Pos Tail
        # We define it locally: if x > mu, P = Gamma. Else -inf.
        # But we want differentiable gradients?
        # Soft-masking? MELODIC assumes hard split?
        # GGM is a mixture over the whole real line? 
        # Usually Gamma is 0 for x<0.
        
        # lp_pos = log(w[1]) + gamma.logpdf(diff, a_pos, loc=0, scale=scale_pos)
        # But since diff can be negative, gamma logpdf will be NaN.
        # We must mask it.
        
        lp_pos_raw = gamma.logpdf(diff, a_pos, loc=0, scale=scale_pos)
        lp_pos = jnp.where(diff > 0, lp_pos_raw, -jnp.inf)
        lp_pos = jnp.log(weights[1]) + lp_pos
        
        # Neg Tail
        # Gamma(-(x-mu)) => Gamma(-diff) for -diff > 0 => diff < 0
        lp_neg_raw = gamma.logpdf(-diff, a_neg, loc=0, scale=scale_neg)
        lp_neg = jnp.where(diff < 0, lp_neg_raw, -jnp.inf)
        lp_neg = jnp.log(weights[2]) + lp_neg
        
        # Combine
        # P(x) = sum exp(lp)
        return jax.scipy.special.logsumexp(jnp.stack([lp_g, lp_pos, lp_neg]), axis=0)

    def posterior_prob(self, x):
        """Returns P(Active | x) = P(Pos|x) + P(Neg|x)."""
        weights, mu, sigma, a_pos, b_pos, a_neg, b_neg = self.get_params()
        
        scale_pos = 1.0 / b_pos
        scale_neg = 1.0 / b_neg
        diff = x - mu
        
        # Likelihoods
        p_g = weights[0] * norm.pdf(x, loc=mu, scale=sigma)
        
        p_pos = jnp.where(
            diff > 0, 
            weights[1] * gamma.pdf(diff, a_pos, loc=0, scale=scale_pos), 
            0.0
        )
        
        p_neg = jnp.where(
            diff < 0, 
            weights[2] * gamma.pdf(-diff, a_neg, loc=0, scale=scale_neg), 
            0.0
        )
        
        evidence = p_g + p_pos + p_neg + 1e-18
        
        p_active = (p_pos + p_neg) / evidence
        return p_active

    @staticmethod
    def loss(model, x):
        return -jnp.mean(model.log_prob(x))
        
    @staticmethod
    def fit(x, key, steps=1000, lr=0.01):
        model = GaussianGammaMixture(key)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(model)
        
        @eqx.filter_jit
        def step(model, opt_state, x):
            loss_val, grads = eqx.filter_value_and_grad(GaussianGammaMixture.loss)(model, x)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val
            
        for i in range(steps):
            model, opt_state, loss = step(model, opt_state, x)
            # if i % 100 == 0:
            #     print(f"Step {i}, Loss: {loss}")
                
        return model
