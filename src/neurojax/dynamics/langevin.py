"""Langevin / Fokker-Planck drift+diffusion estimation — re-exported from jaxctrl.

Implementation in :mod:`jaxctrl` (`_langevin`): recovers drift + diffusion from a
trajectory and splits the drift into dissipative (gradient) and solenoidal
(irreversible, cyclic) parts with the entropy-production rate — the data-driven
member of the Fokker-Planck family (Friston's (Γ+Q)∇log p; Ingber's SMNI).
NeuroJAX re-exports it to apply to source-MEG / connectome-harmonic latent
dynamics — e.g. testing whether a resting cycle is a non-equilibrium SOLENOIDAL
flow (nonzero entropy production) rather than a deterministic limit cycle.
See ``docs/LANGEVIN_FOKKER_PLANCK_NEUROIMAGING.md``.
"""

from jaxctrl import (
    LinearLangevin,
    fit_linear_langevin,
    langevin_gradient_part,
    langevin_solenoidal_part,
    langevin_entropy_production,
    langevin_solenoidal_frequency,
    transition_flux,
    discrete_entropy_production,
)

__all__ = [
    "LinearLangevin",
    "fit_linear_langevin",
    "langevin_gradient_part",
    "langevin_solenoidal_part",
    "langevin_entropy_production",
    "langevin_solenoidal_frequency",
    "transition_flux",
    "discrete_entropy_production",
]
