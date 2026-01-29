import jax.numpy as jnp
import jax.scipy.signal as jss
from jax import jit

@jit
def filter_data(data, b, a):
    return jss.lfilter(b, a, data, axis=-1)
