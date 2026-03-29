"""Multi-echo T2/T2* fitting in JAX."""

import jax
import jax.numpy as jnp
import optax
from functools import partial

from neurojax.qmri.steady_state import multiecho_signal_multi


def monoexp_t2star_fit(data: jnp.ndarray, TEs: jnp.ndarray,
                        n_iters: int = 50) -> dict:
    """Mono-exponential T2* fit: S(TE) = S0 * exp(-TE/T2*).

    Log-linear initialisation + gradient refinement.
    """
    # Log-linear init
    log_data = jnp.log(jnp.maximum(data, 1e-10))
    slope = (log_data[-1] - log_data[0]) / (TEs[-1] - TEs[0])
    T2star_init = jnp.clip(-1.0 / slope, 0.001, 0.500)
    S0_init = jnp.exp(log_data[0] + TEs[0] / T2star_init)

    params = jnp.array([S0_init, T2star_init])

    def loss(params):
        S0, T2s = params[0], jnp.clip(params[1], 0.001, 0.500)
        pred = multiecho_signal_multi(S0, T2s, TEs)
        return jnp.sum((pred - data)**2)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        l, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), l

    (params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=n_iters)

    S0 = params[0]
    T2star = jnp.clip(params[1], 0.001, 0.500)
    R2star = 1.0 / T2star
    pred = multiecho_signal_multi(S0, T2star, TEs)
    rmse = jnp.sqrt(jnp.mean((pred - data)**2))

    return {"S0": S0, "T2star": T2star, "R2star": R2star, "rmse": rmse}
