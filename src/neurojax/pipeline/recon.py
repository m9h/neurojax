# © NeuroJAX developers
#
# License: BSD (3-clause)
"""EMEG "Recon" orchestrator — sensor M/EEG to leakage-aware source connectivity.

The differentiable, GPU-native analogue of an electrophysiological source-imaging
pipeline (cf. Valdés-Sosa's CiftiStorm): given a leadfield and sensor data, invert
to source space and read off directed connectivity, with the leakage diagnostic
that Valdés-Sosa insists on.

This is the inverse→connectivity *core*. The front-end that produces the
leadfield from a subject's anatomy — :func:`neurojax.geometry.bem.coregister_montage`
(EEG montage ↔ the FreeSurfer ``subjects_dir``) then a BEM/FEM forward — is the
data-staging driver; this function takes the resulting ``gain`` matrix.

The inverse operator is pluggable: the default is an MNE minimum-norm kernel, but
for the leakage-aware path Valdés-Sosa recommends, pass a HIGGS MAP kernel
(:func:`neurojax.source.higgs.higgs_source_estimate`) or a dSPM/eLORETA operator
via ``inverse_operator`` — connectivity then runs on better-separated sources.
"""

import jax.numpy as jnp

from neurojax.analysis.source_connectivity import (
    directed_source_connectivity,
    leakage_index,
)

__all__ = ["recon_directed_connectivity"]


def _mne_operator(gain: jnp.ndarray, reg: float) -> jnp.ndarray:
    """Minimum-norm (Tikhonov-regularised) inverse kernel ``W`` (n_sources, n_sensors)."""
    n = gain.shape[0]
    GGt = gain @ gain.T
    lam = reg * jnp.trace(GGt) / n
    return gain.T @ jnp.linalg.inv(GGt + lam * jnp.eye(n, dtype=gain.dtype))


def recon_directed_connectivity(gain: jnp.ndarray, sensor_data: jnp.ndarray,
                                freqs: jnp.ndarray, fs: float, *,
                                inverse_operator: jnp.ndarray | None = None,
                                reg: float = 0.05, order: int = 1) -> dict:
    """Sensor M/EEG → source estimate → directed source connectivity + leakage.

    Parameters
    ----------
    gain : (n_sensors, n_sources) leadfield from the forward model.
    sensor_data : (n_sensors, n_times) M/EEG.
    freqs, fs : frequency grid and sampling rate for the directed measures.
    inverse_operator : (n_sources, n_sensors) or None
        Explicit inverse kernel (e.g. HIGGS MAP / dSPM — the leakage-aware path).
        If ``None``, a minimum-norm kernel is built from ``gain``.
    reg : Tikhonov fraction for the default minimum-norm kernel.
    order : MVAR model order for PDC/DTF.

    Returns
    -------
    dict with ``sources`` (n_sources, n_times), ``pdc`` / ``dtf``
    (n_freqs, n_sources, n_sources), the MVAR ``A`` / ``Sigma``, and ``leakage``
    (n_sources, n_sources) from the resolution matrix ``W @ gain`` — high
    off-diagonal => that directed estimate is leakage-contaminated.
    """
    W = _mne_operator(gain, reg) if inverse_operator is None else inverse_operator
    sources = W @ sensor_data
    out = directed_source_connectivity(sources, freqs, fs, order=order)
    out["sources"] = sources
    out["leakage"] = leakage_index(W @ gain)
    return out
