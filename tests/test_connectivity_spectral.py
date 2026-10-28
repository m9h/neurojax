"""Partial coherence + imaginary coherency from a cross-spectral density.

neurojax already has CSD + magnitude-squared coherence (multitaper_cpsd,
coherence_from_cpsd). These add the two standard measures it lacked: partial
coherence (conditioning out the other channels, via the inverse CSD) and
imaginary coherency (Nolte et al. 2004 — the volume-conduction-robust measure
that is insensitive to zero-lag/instantaneous mixing). Both take a CSD
(n_freqs, C, C) like coherence_from_cpsd, so they compose with the existing
multitaper CSD path.
"""

import jax.numpy as jnp
import pytest

from neurojax.analysis.connectivity_spectral import (
    imaginary_coherency_from_cpsd,
    partial_coherence_from_cpsd,
)


def test_partial_coherence_identity_csd_has_zero_offdiagonal():
    # CSD = I per frequency: channels share no variance -> partial coherence is
    # the identity (diag 1, off-diag 0), and is bounded in [0, 1].
    C, nf = 3, 4
    cpsd = jnp.broadcast_to(jnp.eye(C, dtype=jnp.complex64)[None], (nf, C, C))
    pc = partial_coherence_from_cpsd(cpsd)
    assert pc.shape == (nf, C, C)
    offdiag = pc[:, ~jnp.eye(C, dtype=bool)]
    assert jnp.allclose(offdiag, 0.0, atol=1e-4)
    assert jnp.all(pc >= -1e-5) and jnp.all(pc <= 1.0 + 1e-5)


def test_imaginary_coherency_is_zero_for_real_csd():
    # A purely real CSD (zero-lag / instantaneous mixing, e.g. volume conduction)
    # must give zero imaginary coherency -- the defining robustness property.
    C, nf = 3, 4
    base = jnp.ones((C, C)) * 0.5 + jnp.eye(C) * 0.5
    cpsd = jnp.broadcast_to(base[None].astype(jnp.complex64), (nf, C, C))
    ic = imaginary_coherency_from_cpsd(cpsd)
    assert ic.shape == (nf, C, C)
    assert jnp.allclose(ic, 0.0, atol=1e-6)


def test_imaginary_coherency_recovers_phase_lagged_pair():
    # x,y with a 90-degree lag: S_xy = 0.5j, S_xx=S_yy=1 -> coherency = 0.5j,
    # so imaginary coherency = 0.5, and everything stays bounded in [-1, 1].
    Sxy = 0.5j
    M = jnp.array([[1.0 + 0j, Sxy], [jnp.conj(Sxy), 1.0 + 0j]])[None]
    ic = imaginary_coherency_from_cpsd(M)
    assert jnp.allclose(ic[0, 0, 1], 0.5, atol=1e-6)
    assert jnp.all(jnp.abs(ic) <= 1.0 + 1e-6)
