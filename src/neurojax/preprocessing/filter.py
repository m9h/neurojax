"""IIR filtering in pure JAX via jax.lax.scan.

Replaces the non-existent ``jax.scipy.signal.lfilter`` with a
Direct-Form II Transposed IIR filter implemented using ``jax.lax.scan``.
Fully differentiable.
"""

import jax
import jax.numpy as jnp


def lfilter(b: jnp.ndarray, a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Apply an IIR filter along the last axis using Direct-Form II Transposed.

    Parameters
    ----------
    b : (M+1,) numerator coefficients.
    a : (N+1,) denominator coefficients. ``a[0]`` must be non-zero.
    x : (..., T) input signal.

    Returns
    -------
    y : same shape as x, filtered signal.
    """
    # Normalize by a[0]
    a0 = a[0]
    b = b / a0
    a = a / a0

    # Pad to equal length
    n = max(len(a), len(b))
    b = jnp.pad(b, (0, n - len(b)))
    a = jnp.pad(a, (0, n - len(a)))

    def _filter_1d(x_1d):
        """Filter a single 1-D signal."""
        order = n - 1

        # Trivial case: b=[c], a=[1] → just scaling, no state
        if order == 0:
            return b[0] * x_1d

        def scan_fn(state, x_t):
            # Direct-Form II Transposed
            y_t = b[0] * x_t + state[0]
            new_state = jnp.zeros(order)
            for k in range(order - 1):
                new_state = new_state.at[k].set(
                    b[k + 1] * x_t - a[k + 1] * y_t + state[k + 1]
                )
            new_state = new_state.at[order - 1].set(
                b[order] * x_t - a[order] * y_t
            )
            return new_state, y_t

        init_state = jnp.zeros(order)
        _, y = jax.lax.scan(scan_fn, init_state, x_1d)
        return y

    # Handle batched input: vmap over all axes except the last
    if x.ndim == 1:
        return _filter_1d(x)
    else:
        # Flatten batch dims, filter, reshape back
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)
        y_flat = jax.vmap(_filter_1d)(x_flat)
        return y_flat.reshape(batch_shape + (T,))


@jax.jit
def filter_data(data: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Filter data along the last axis. Drop-in for the old API."""
    return lfilter(b, a, data)
