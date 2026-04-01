"""Koopman operator and Dynamic Mode Decomposition (DMD).

Re-exports the :class:`KoopmanEstimator` from :mod:`jaxctrl` for
backwards compatibility.  Users should prefer importing from
:mod:`neurojax.dynamics` directly.

The Koopman operator provides a linear (but infinite-dimensional)
representation of nonlinear dynamics.  DMD approximates the leading
Koopman modes and eigenvalues from snapshot pairs, yielding dominant
spatial modes and their associated frequencies / growth rates.

Exported symbols:
    KoopmanEstimator: SVD-based DMD estimator.  Given snapshot matrices
        ``X`` and ``Y = F(X)``, computes the rank-truncated DMD
        approximation ``A_r``, its eigenvalues, and the DMD modes.

Example::

    from neurojax.dynamics.koopman import KoopmanEstimator

    est = KoopmanEstimator(rank=10)
    modes, eigenvalues, amplitudes = est.fit(X, Y)

References:
    Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
        experimental data. *J. Fluid Mech.*, 656, 5-28.
    Brunton, S. L., Budisic, M., Kaiser, E., & Kutz, J. N. (2021).
        Modern Koopman theory for dynamical systems. *SIAM Review*,
        64(2), 229-340.

See Also:
    neurojax.dynamics.windowed.windowed_dmd: Windowed DMD tracking
        dominant frequencies over time.
"""

from jaxctrl import KoopmanEstimator

__all__ = ["KoopmanEstimator"]
