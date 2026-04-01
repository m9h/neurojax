"""Sparse Identification of Nonlinear Dynamics (SINDy).

Re-exports the core SINDy implementation from :mod:`jaxctrl` for
backwards compatibility and convenience.  Users should prefer importing
from :mod:`neurojax.dynamics` directly.

Exported symbols:
    SINDyOptimizer: Sequentially-thresholded least-squares (STLS)
        optimiser that discovers sparse governing equations
        ``dX/dt = Theta(X) * Xi`` from data.
    polynomial_library: Constructs the polynomial feature library
        ``Theta(X)`` up to a given degree.

Example::

    from neurojax.dynamics.sindy import SINDyOptimizer, polynomial_library

    optimizer = SINDyOptimizer(threshold=0.05, max_iter=20)
    lib_fn = lambda X: polynomial_library(X, degree=3)
    Xi = optimizer.fit(X, dX, lib_fn)

References:
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
        Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems.
        *PNAS*, 113(15), 3932-3937.

See Also:
    neurojax.dynamics.windowed.windowed_sindy: Windowed SINDy tracking
        Jacobian eigenvalues over time.
"""

from jaxctrl import SINDyOptimizer, polynomial_library

__all__ = ["SINDyOptimizer", "polynomial_library"]
