import mne
import jax.numpy as jnp
import numpy as np

def mne_to_jax(inst):
    data = inst.get_data()
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return jnp.array(data), inst.info["sfreq"]

def jax_to_mne(data, template):
    data_np = np.array(data)
    inst = template.copy()
    if isinstance(inst, mne.io.BaseRaw):
        inst._data = data_np
    return inst
