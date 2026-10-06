# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Directed connectivity on source-space estimates, with a leakage diagnostic.

Wires the MVAR directed measures in :mod:`neurojax.analysis.mvar` (PDC, DTF) onto
*source* time courses rather than sensors. The methodological caveat (Valdés-Sosa;
the HIGGS rationale) is that connectivity must be computed on leakage-aware source
estimates — e.g. :func:`neurojax.source.higgs.higgs_source_estimate`, which jointly
estimates source activity and connectivity — and NOT on naively projected sources
(minimum-norm etc.), because spatial leakage injects ghost directed connectivity.

:func:`leakage_index` quantifies that risk from the resolution matrix
``M = W @ G`` (see :func:`neurojax.source.minimum_norm.resolution_matrix`): a large
normalised off-diagonal ``|M_ij|`` means any directed estimate between sources
``i`` and ``j`` is leakage-contaminated and should be distrusted (or the sources
re-estimated jointly, à la HIGGS).
"""

import jax.numpy as jnp

from neurojax.analysis.mvar import dtf, fit_mvar, pdc

__all__ = ["directed_source_connectivity", "leakage_index"]


def directed_source_connectivity(sources: jnp.ndarray, freqs: jnp.ndarray,
                                 fs: float, order: int = 1) -> dict:
    """PDC + DTF directed connectivity on source-space time courses.

    Parameters
    ----------
    sources : (n_sources, n_times)
        Source-space time courses. Prefer leakage-aware estimates
        (``higgs_source_estimate``) over naive min-norm projections.
    freqs : (n_freqs,) and fs : float — frequency grid and sampling rate.
    order : MVAR model order.

    Returns
    -------
    dict with ``pdc`` and ``dtf`` (each (n_freqs, n_sources, n_sources)) plus the
    fitted MVAR ``A`` and innovation covariance ``Sigma``.
    """
    A, Sigma = fit_mvar(sources.T, order)  # fit_mvar wants (n_times, n_sources)
    return {"pdc": pdc(A, freqs, fs), "dtf": dtf(A, freqs, fs), "A": A, "Sigma": Sigma}


def leakage_index(resolution_matrix: jnp.ndarray) -> jnp.ndarray:
    """Per-pair spatial-leakage coefficient from a resolution matrix ``M = W @ G``.

    ``leakage_{ij} = |M_{ij}| / sqrt(|M_{ii}| |M_{jj}|)`` — the cross-talk between
    sources ``i`` and ``j`` normalised by their self-resolution. Off-diagonal
    values near 1 mean strong leakage (directed connectivity untrustworthy);
    near 0 mean well-separated sources. Diagonal is 1.

    Parameters
    ----------
    resolution_matrix : (n_sources, n_sources) real or complex.

    Returns
    -------
    (n_sources, n_sources) real in [0, ~1].
    """
    M = jnp.abs(resolution_matrix)
    d = jnp.sqrt(jnp.maximum(jnp.diagonal(M), 1e-20))
    return M / (d[:, None] * d[None, :])
