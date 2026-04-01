"""MNE-Python to JAX array bridge (extended version).

Provides :func:`mne_to_jax` for extracting continuous data from an
:class:`mne.io.Raw` object into a JAX array with the ``(T, C)`` layout
expected by neurojax analysis routines.  Unlike the minimal converter in
:mod:`neurojax.io.bridge`, this version supports channel picking and
sample-range slicing so that large recordings can be loaded incrementally.

Typical usage::

    from neurojax.utils.bridge import mne_to_jax

    data = mne_to_jax(raw, picks="meg", start=0, stop=30_000)
    # data.shape == (30000, n_meg_channels)

Note:
    The output array is transposed relative to MNE convention:
    MNE returns ``(channels, times)``; this function returns
    ``(times, channels)`` to match JAX/Equinox batch-dimension
    conventions.

See Also:
    neurojax.io.bridge.mne_to_jax: Simpler converter (no slicing).
    neurojax.io.bridge.jax_to_mne: Round-trip back to MNE objects.
"""

import jax.numpy as jnp
import numpy as np


def mne_to_jax(raw, picks=None, start=None, stop=None, return_times=False):
    """Extract data from an MNE Raw object into a JAX array.

    Args:
        raw (mne.io.Raw): MNE Raw object containing continuous data.
        picks (str | list[str] | None): Channel selection passed to
            :meth:`mne.io.Raw.get_data`.  Can be a channel type
            (e.g. ``"meg"``, ``"eeg"``) or a list of channel names.
        start (int | None): First sample index to read.  Defaults to 0.
        stop (int | None): One-past-last sample index.  ``None`` reads
            to the end of the recording.
        return_times (bool): If ``True``, also return the time vector
            (in seconds) as a NumPy array.

    Returns:
        jax.Array: Data array with shape ``(T, C)`` — time-points by
            channels, transposed from MNE's native ``(C, T)`` layout.
        numpy.ndarray: *(only when* ``return_times=True`` *)* — time
            vector of length *T*.
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
