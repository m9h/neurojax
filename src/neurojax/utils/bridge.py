import jax.numpy as jnp
import numpy as np

def mne_to_jax(raw, picks=None, start=None, stop=None, return_times=False):
    """
    Extracts data from an MNE Raw object to a JAX Array.
    
    Args:
        raw: mne.io.Raw object
        picks: channels to include
        start: start sample index
        stop: stop sample index
        return_times: bool, whether to return the time vector
        
    Returns:
        data_jax: JAX Array [times, sensors] (Note transposed from MNE's [sensors, times])
        times: (optional) numpy array of time points
    """
    if start is None:
        start = 0
    data, times = raw.get_data(picks=picks, start=start, stop=stop, return_times=True)
    
    # MNE returns [channels, times]. We usually want [times, channels] for JAX/Equinox
    # standard batch dimension patterns.
    data_jax = jnp.array(data.T)
    
    if return_times:
        return data_jax, times
    else:
        return data_jax
