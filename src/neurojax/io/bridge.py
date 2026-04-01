"""Minimal MNE <-> JAX round-trip converters.

These two functions form the thinnest possible bridge between the
MNE-Python ecosystem (file I/O, preprocessing, visualisation) and the
JAX array world used for GPU-accelerated modelling in neurojax.

* :func:`mne_to_jax` — MNE ``Raw`` / ``Epochs`` -> ``(jax.Array, sfreq)``
* :func:`jax_to_mne` — ``jax.Array`` -> MNE object (copies metadata
  from a *template* instance so channel names, montage, etc. are
  preserved).

Data is cast to ``float32`` on the way in to match JAX's default
precision.

Example::

    from neurojax.io.bridge import mne_to_jax, jax_to_mne

    data_jax, sfreq = mne_to_jax(raw)
    # ... run JAX pipeline ...
    raw_out = jax_to_mne(result, template=raw)

See Also:
    neurojax.utils.bridge.mne_to_jax: Extended converter with channel
        picking and sample-range slicing.
"""

import mne
import jax.numpy as jnp
import numpy as np


def mne_to_jax(inst):
    """Convert an MNE data instance to a JAX array.

    Args:
        inst (mne.io.BaseRaw | mne.BaseEpochs): Any MNE object that
            exposes a :meth:`get_data` method.

    Returns:
        tuple[jax.Array, float]: A 2-tuple of:
            - **data** — JAX array with the same shape as
              ``inst.get_data()`` (typically ``(C, T)`` for Raw or
              ``(n_epochs, C, T)`` for Epochs), cast to ``float32``.
            - **sfreq** — sampling frequency in Hz.
    """
    data = inst.get_data()
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return jnp.array(data), inst.info["sfreq"]


def jax_to_mne(data, template):
    """Write a JAX array back into an MNE object, preserving metadata.

    The *template* instance is copied and its internal data buffer is
    replaced with the contents of *data*.  Channel names, montage,
    projectors, and all other metadata are inherited from the template.

    Args:
        data (jax.Array): Array with shape matching
            ``template.get_data().shape``.
        template (mne.io.BaseRaw): MNE Raw instance whose metadata
            (channel info, sampling rate, etc.) should be reused.

    Returns:
        mne.io.BaseRaw: A copy of *template* with the data buffer
            replaced by *data*.

    Warning:
        Currently only :class:`mne.io.BaseRaw` instances are supported
        as templates.  Epochs support may be added in a future release.
    """
    data_np = np.array(data)
    inst = template.copy()
    if isinstance(inst, mne.io.BaseRaw):
        inst._data = data_np
    return inst
